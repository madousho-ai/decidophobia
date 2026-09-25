#!/usr/bin/env python
"""synth-intents v3: 让一个参考模型把一个领域的题全部答一遍, 采集它在选项上的概率分布, 写成 <domain>.answer.json.
纯标准库. 推理走 OpenAI 兼容的远程端点, 不碰本机 GPU.

  export LLM_API_KEY=sk-...       # key 只从环境变量读, 不进任何文件
  python datasets/synth-intents-v3/answer.py telecom [--method sample] [--model M] [--threads 6]

题目与文本直接从 <domain>.py 读 (QUESTIONS / TEXTS), 先用 schema.problems 检查一遍, 不依赖生成出来的 JSON.
judged 为 false 的题 (如 256 个选项的意图题) 不答, 它的标签只在 stated 里.

## 分布怎么来

两种方法都关掉思考, 一道 k 个选项的题按循环移位摆 k 种顺序, 每种顺序得到一个分布 (映射回原顺序),
k 种顺序再取平均, 抵消模型的位置偏向.

  --method sample    每种顺序采 --samples 次, 分布是选中频率. 上游不回 logprobs 时只能用它 (如 OpenRouter).
                     回复里第一个整数落在 1..k 才算数, 否则重采, 一个样本最多 3 次; 3 次都不合格的不计数, 下次运行补采.
  --method logprobs  (默认) 每种顺序调一次, 读回复第一个 token 的 top_logprobs (最多 TOP_LOGPROBS 个),
                     取编号 1..k 的概率归一. 上游要回 logprobs (如 DeepSeek 官方). 返回的是温度作用之后的概率,
                     所以用 temperature 1. 选项多于 TOP_LOGPROBS 个时, 排不进前面的选项记 0.
                     一个编号都没有的回复不算数, 最多 3 次, 下次运行补.

提示 (与判题时同形):
  system  SYSTEM
  user    <LABEL>:\\n<文本; JSON state 缩进 2 格展开>\\n\\nQuestion: <ask[0]>\\nOptions:\\n1. ...\\nk. ...\\n\\n
          Answer with the option number only.

## <domain>.answer.json

  {"domain", "model", "params": {方法与参数}, "answers": {"<文本 id>": {"<题 id>": 一道题, ...}, ...}}

一道题, 按方法二选一:
  sample    {"hash", "p", "counts": [[第 s 种顺序下每个选项被选中的次数] * k]}
  logprobs  {"hash", "p", "dists": [[第 s 种顺序下每个选项的概率] 或 null * k],
             "mass": [第 s 种顺序下编号 1..k 在第一个 token 上的概率总和, 归一之前 或 null * k]}
文本在 <domain>.py 里把这道题列进 unknown (文本对它没给线索) 时, 不问模型, 写成
  {"hash", "unknown": true, "p": [1/k] * k}
去掉 unknown 标记再跑, 这道题会重答.

一行一份文本. p、counts、dists 都按 <domain>.py 里选项的原顺序; 第 s 项是选项循环移位 s 位摆放时的结果.
null 是还没答到的顺序. mass 低说明模型没在答编号, 那一格的分布靠不住.
hash 是提示内容 (标签、文本、ask[0]、选项) 的摘要. 改了 <domain>.py 之后再跑, hash 变了的题丢掉旧结果重答,
没变的只补还缺的顺序. 文件里的 model 或方法参数与这次不同时拒绝运行, 加 --restart 才从头来;
sample 下只把 --samples 调大是允许的, 会补采.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import os
import pathlib
import re
import statistics as st
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from schema import applies, problems  # noqa: E402

DEFAULT_BASE = "https://llm-api.usnj.uuzdream.cn/v1"
DEFAULT_MODEL = "deepseek/deepseek-v4.1-flash"
SYSTEM = "You answer multiple-choice questions about a text. Reply with only the number of the option you choose."
SAVE_EVERY = 30  # 秒: 采到一半断掉最多丢这么久的样本
TOP_LOGPROBS = 20  # DeepSeek 官方允许的上限


def load_domain(domain: str):
    path = HERE / f"{domain}.py"
    if not path.exists():
        sys.exit(f"没有 {path}")
    spec = importlib.util.spec_from_file_location(f"synth_v3_{domain}", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    errs = problems(m.DOMAIN, m.LABEL, m.QUESTIONS, m.TEXTS)
    if m.DOMAIN != domain:
        errs.insert(0, f"{path.name} 里的 DOMAIN 是 {m.DOMAIN!r}")
    if errs:
        sys.exit(f"{path.name} 有 {len(errs)} 个问题, 先用它自己跑一遍看清楚:\n  " + "\n  ".join(errs))
    return m


def state_text(text) -> str:
    return text if isinstance(text, str) else json.dumps(text, indent=2, ensure_ascii=False)


def prompt(label: str, text, question: str, options: list[str], shift: int) -> tuple[list[int], str]:
    """第 shift 种顺序的提示. 返回 (order, user): 第 j 个显示位放原来的第 order[j] 项."""
    n = len(options)
    order = [(j + shift) % n for j in range(n)]
    menu = "\n".join(f"{j + 1}. {options[o]}" for j, o in enumerate(order))
    return order, (f"{label}:\n{state_text(text)}\n\nQuestion: {question}\nOptions:\n{menu}\n\n"
                   "Answer with the option number only.")


def item_hash(label: str, text, question: str, options: list[str]) -> str:
    blob = json.dumps([label, text, question, options], ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def parse(raw: str, n: int) -> int | None:
    """回复里第一个独立的整数, 在 1..n 之内才算. 返回显示位 (0 起), 否则 None."""
    m = re.search(r"\b(\d+)\b", raw)
    return int(m.group(1)) - 1 if m and 1 <= int(m.group(1)) <= n else None


def first_token_dist(choice: dict, n: int) -> tuple[list[float], float] | None:
    """回复第一个 token 的 top_logprobs 里编号 1..n 的概率. 返回 (按显示位归一的分布, 归一前的总和 mass);
    一个编号都没有就是 None. 编号外的 token 丢掉, 同一个编号的几种写法 (如 "3" 与 " 3") 相加."""
    content = (choice.get("logprobs") or {}).get("content") or []
    if not content:
        return None
    probs = [0.0] * n
    for t in content[0].get("top_logprobs") or []:
        m = re.fullmatch(r"\s*(\d+)\s*", t.get("token", ""))
        if m and 1 <= int(m.group(1)) <= n:
            probs[int(m.group(1)) - 1] += math.exp(t["logprob"])
    mass = sum(probs)
    return ([x / mass for x in probs], mass) if mass > 0 else None


def chat(base: str, key: str, model: str, params: dict, user: str) -> dict | str:
    """返回 choices[0]; 网关一直出错就返回 "ERROR ..." 字符串."""
    body = {"model": model, "temperature": params["temperature"], "max_tokens": params["max_tokens"],
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]}
    if not params["reasoning"]:
        # 两家上游关思考的写法不同: OpenRouter 认 reasoning, DeepSeek 官方认 thinking (实测忽略 reasoning).
        # OpenRouter 收到 thinking 会怎样没测过.
        body["reasoning"] = {"enabled": False}
        body["thinking"] = {"type": "disabled"}
    if params["method"] == "logprobs":
        body["logprobs"] = True
        body["top_logprobs"] = params["top_logprobs"]
    last = None
    for attempt in range(8):
        try:
            req = urllib.request.Request(f"{base}/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                d = json.load(r)
            return d["choices"][0]
        except urllib.error.HTTPError as e:
            last = e
            e.close()
            time.sleep(min(30 + 30 * attempt if e.code == 429 else 3 * 2 ** min(attempt, 5), 180))
        except Exception as e:  # noqa: BLE001 —— 网关偶发断连 / 超时, 退避重试
            last = e
            time.sleep(3 * 2 ** min(attempt, 5))
    return f"ERROR {last}"


def reply_text(choice: dict | str) -> str:
    return choice if isinstance(choice, str) else (choice.get("message", {}).get("content") or "").strip()


def order_dists(entry: dict) -> list[list[float]]:
    """已答到的每种顺序在原选项顺序上的分布. unknown 的题没有."""
    if entry.get("unknown"):
        return []
    if "dists" in entry:
        return [d for d in entry["dists"] if d]
    return [[x / sum(c) for x in c] for c in entry["counts"] if sum(c)]


def finish(entry: dict) -> None:
    """算 p: 已答到的顺序之间取平均 (sample 先把每种顺序的计数归一). unknown 的题 p 固定是均匀分布, 不动."""
    if entry.get("unknown"):
        return
    rows = order_dists(entry)
    n = len(entry["dists"] if "dists" in entry else entry["counts"])
    entry["p"] = [round(sum(r[i] for r in rows) / len(rows), 3) for i in range(n)] if rows else [0.0] * n


def save(path: pathlib.Path, doc: dict) -> None:
    """一行一份文本, 先写临时文件再改名, 中途被杀也不会留下半个文件."""
    head = json.dumps({k: doc[k] for k in ("domain", "model", "params")}, ensure_ascii=False)[:-1]
    body = ",\n".join(f"{json.dumps(tid)}: {json.dumps(a, ensure_ascii=False, separators=(', ', ': '))}"
                      for tid, a in doc["answers"].items())
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(f'{head}, "answers": {{\n{body}\n}}}}\n', encoding="utf-8")
    tmp.replace(path)


def build(m, doc: dict) -> list[tuple]:
    """按当前 <domain>.py 整理 doc["answers"]: 删掉已不存在的文本 / 题, hash 变了或方法不同的清零.
    返回还要答的 (文本 id, 题 id, shift, 缺几次)."""
    logprobs = doc["params"]["method"] == "logprobs"
    samples = doc["params"].get("samples_per_order", 0)
    field = "dists" if logprobs else "counts"
    old, new, todo = doc["answers"], {}, []
    for t in m.TEXTS:
        for q in m.QUESTIONS:
            if not q.judged or not applies(q, t):
                continue
            k = len(q.options)
            h = item_hash(m.LABEL, t.text, q.ask[0], q.options)
            if q.id in t.unknown:
                new.setdefault(t.id, {})[q.id] = {"hash": h, "unknown": True, "p": [round(1 / k, 4)] * k}
                continue
            e = old.get(t.id, {}).get(q.id)
            if not e or e.get("hash") != h or len(e.get(field, [])) != k:
                e = ({"hash": h, "p": [], "dists": [None] * k, "mass": [None] * k} if logprobs
                     else {"hash": h, "p": [], "counts": [[0] * k for _ in range(k)]})
            new.setdefault(t.id, {})[q.id] = e
            finish(e)
            if logprobs:
                todo += [(t.id, q.id, s, 1) for s, d in enumerate(e["dists"]) if d is None]
            else:
                todo += [(t.id, q.id, s, samples - sum(c)) for s, c in enumerate(e["counts"]) if sum(c) < samples]
    doc["answers"] = new
    return todo


def summary(m, doc: dict) -> str:
    """写明的答案命中几题、平均概率; 没写的题最高项平均多少; 不同顺序之间差多少; logprobs 下编号拿到多少概率.
    标成 unknown 的题只计数, 不进其余统计."""
    stated_p, hit, unstated_max, tvs, masses, unknown = [], 0, [], [], [], 0
    for t in m.TEXTS:
        for qid, e in doc["answers"].get(t.id, {}).items():
            if e.get("unknown"):
                unknown += 1
                continue
            p = e["p"]
            if not any(p):
                continue
            if qid in t.stated:
                i = t.stated[qid]
                stated_p.append(p[i])
                hit += max(range(len(p)), key=p.__getitem__) == i
            else:
                unstated_max.append(max(p))
            rows = order_dists(e)
            tvs += [0.5 * sum(abs(x - y) for x, y in zip(a, b)) for a, b in itertools.combinations(rows, 2)]
            masses += [x for x in e.get("mass", []) if x is not None]
    out = [f"模型 {doc['model']}  参数 {doc['params']}"]
    if stated_p:
        out.append(f"写明的答案 {len(stated_p)} 个: 首选命中 {hit}, 平均概率 {st.mean(stated_p):.3f}, 最低 {min(stated_p):.2f}")
    if unstated_max:
        out.append(f"没写的题 {len(unstated_max)} 道: 最高项平均 {st.mean(unstated_max):.3f}")
    if unknown:
        out.append(f"文本没给线索 (unknown, 均匀分布) {unknown} 道")
    if tvs:
        out.append(f"不同选项顺序之间的总变差: 平均 {st.mean(tvs):.3f}, 最大 {max(tvs):.2f}")
    if masses:
        out.append(f"第一个 token 落在编号上的概率: 平均 {st.mean(masses):.3f}, 最低 {min(masses):.3f}, "
                   f"低于 0.9 的 {sum(x < 0.9 for x in masses)} 格 / {len(masses)}")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="让参考模型答一个领域的全部题, 写 <domain>.answer.json")
    ap.add_argument("domain")
    ap.add_argument("--method", choices=["sample", "logprobs"], default="logprobs",
                    help="sample: 每种顺序采 --samples 次数频率; logprobs: 每种顺序调一次读编号的概率 (上游要回 logprobs)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--base-url", default=DEFAULT_BASE)
    ap.add_argument("--samples", type=int, default=8, help="sample 下每种选项顺序采几次")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--reasoning", action="store_true", help="不关思考 (默认关, logprobs 下必须关); 开了要把 --max-tokens 调大")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--restart", action="store_true", help="丢掉已有答案从头答 (换模型、方法或参数时要加)")
    ap.add_argument("--summary", action="store_true", help="不调模型, 只按现有答案打印统计")
    args = ap.parse_args()

    m = load_domain(args.domain)
    path = HERE / f"{args.domain}.answer.json"
    logprobs = args.method == "logprobs"
    if logprobs and args.reasoning:
        sys.exit("logprobs 读的是回复第一个 token, 要关掉思考, 去掉 --reasoning")
    params = {"method": args.method, "temperature": args.temperature, "max_tokens": args.max_tokens,
              "reasoning": args.reasoning}
    params.update({"top_logprobs": TOP_LOGPROBS} if logprobs else {"samples_per_order": args.samples})
    doc = {"domain": args.domain, "model": args.model, "params": params, "answers": {}}
    if path.exists() and not args.restart:
        old = json.loads(path.read_text(encoding="utf-8"))
        same = {k: v for k, v in old["params"].items() if k != "samples_per_order"} == \
               {k: v for k, v in params.items() if k != "samples_per_order"}
        if not args.summary and (old["model"] != args.model or not same):
            sys.exit(f"{path.name} 是 {old['model']} {old['params']} 答的, 这次是 {args.model} {params}; "
                     "要换就加 --restart")
        doc = old
        if not args.summary and not logprobs:  # 只改了采样次数: 调大就补采, 调小不删已有的样本
            doc["params"]["samples_per_order"] = max(old["params"]["samples_per_order"], args.samples)
    todo = build(m, doc)
    if args.summary or not todo:
        if not args.summary:  # 删掉的题 / 新标的 unknown 也要落盘
            save(path, doc)
        print(summary(m, doc) if args.summary else f"没有要补的, 已全部答完\n{summary(m, doc)}")
        return

    key = os.environ.get("LLM_API_KEY") or sys.exit("先 export LLM_API_KEY")
    T = {t.id: t for t in m.TEXTS}
    Q = {q.id: q for q in m.QUESTIONS}
    jobs = [(tid, qid, s) for tid, qid, s, need in todo for _ in range(need)]
    print(f"{len(jobs)} 次调用要做, 涉及 {len({(a, b) for a, b, _ in jobs})} 道题", flush=True)
    lock, last_save, invalid = threading.Lock(), [time.time()], [0]

    def run(job):
        tid, qid, s = job
        q = Q[qid]
        order, user = prompt(m.LABEL, T[tid].text, q.ask[0], q.options, s)
        print(f"start {tid} {qid} shift {s}", flush=True)
        got = None
        for _ in range(3):
            choice = chat(args.base_url, key, args.model, params, user)
            if isinstance(choice, str):
                got = None
            elif logprobs:
                got = first_token_dist(choice, len(order))
            else:
                got = parse(reply_text(choice), len(order))
            if got is not None:
                break
        with lock:
            e = doc["answers"][tid][qid]
            if got is None:
                invalid[0] += 1
            elif logprobs:
                dist, mass = got
                back = [0.0] * len(order)
                for j, o in enumerate(order):  # 第 j 个显示位放的是原来的第 o 项
                    back[o] = round(dist[j], 4)
                e["dists"][s], e["mass"][s] = back, round(mass, 4)
            else:
                e["counts"][s][order[got]] += 1
            if time.time() - last_save[0] > SAVE_EVERY:
                for x in (x for a in doc["answers"].values() for x in a.values()):
                    finish(x)
                save(path, doc)
                last_save[0] = time.time()
        status = "ok" if got is not None else f"INVALID {reply_text(choice)[:40]!r}"
        print(f"done  {tid} {qid} shift {s} {status}", flush=True)

    with ThreadPoolExecutor(args.threads) as ex:
        list(ex.map(run, jobs))
    for e in (e for a in doc["answers"].values() for e in a.values()):
        finish(e)
    save(path, doc)
    print(f"无效回复 {invalid[0]} 次 (下次运行会补采)\n{summary(m, doc)}\n→ {path}", flush=True)


if __name__ == "__main__":
    main()
