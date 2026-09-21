#!/usr/bin/env python
"""Banking77 baseline: 77 路意图分类, 选项作为菜单进 context, 用单 token 码读出.

部署形状与 Jev 的 API 一致: 调用方在请求里定义选项, 槽本身匿名. 于是
  - 77 个意图各绑一个两字母大写码 (' QX' 与 'QX' 都是单 token), 码<->意图的分配由 seed 决定
  - 提示 = 指令 + 77 行菜单 ("QX. card arrival") + 用户句 + "Intent:"
  - 一次 forward, 在最后位置读 77 个码的 logit, 归一成决策分布
  - 零样本档存在两种污染: 模型对特定码的先验偏好、对菜单位置的偏好.
    两套不同 perm-seed 的差是偏好噪声地板; 空 query 的先验用于 contextual calibration.
  - --mode bare 只给用户句不给菜单, 存 h_bare 作探针的上界参照.

产物格式沿用 baseline-boolq.py: json (指标+元数据) + npz (h, 槽 logit, logsumexp, top-k).
"""

from __future__ import annotations

import math
import random
import string

# --------------------------------------------------------------------------
# 纯函数 (tests/test_banking77_metrics.py 覆盖)
# --------------------------------------------------------------------------


def pick_code_tokens(tokenizer, n: int, seed: int) -> list[str]:
    """从 'AA'..'ZZ' 里挑 n 个码. 两个条件, 都对着 tokenizer 实测:

    1. ' XY' 与 'XY' 都是单 token —— 答案位置读带空格的, 菜单行里出现不带空格的
    2. ' xy' 与 ' Xy' 不能也都是单 token —— 那样它是个英文词或常见缩写, 带内容无关的先验

    候选按字母序遍历, 再由 seed 抽样, 所以同 seed 同结果.
    """

    def single(s: str) -> bool:
        return len(tokenizer.encode(s, add_special_tokens=False)) == 1

    pool: list[str] = []
    for a in string.ascii_uppercase:
        for b in string.ascii_uppercase:
            code = a + b
            if not (single(" " + code) and single(code)):
                continue
            if single(" " + code.lower()) and single(" " + code.capitalize()):
                continue
            pool.append(code)
    if len(pool) < n:
        raise ValueError(f"only {len(pool)} qualifying codes, need {n}")
    return random.Random(seed).sample(pool, n)


def assign_codes(labels: list[str], codes: list[str], seed: int) -> dict[str, str]:
    """码 <-> label 的一个随机双射. 同 seed 同置换; 换 seed 就是另一套分配."""
    if len(labels) != len(codes):
        raise ValueError(f"{len(labels)} labels vs {len(codes)} codes")
    shuffled = list(codes)
    random.Random(seed).shuffle(shuffled)
    return dict(zip(labels, shuffled))


def humanize_label(raw: str) -> str:
    """'Refund_not_showing_up' -> 'refund not showing up'; 'reverted_card_payment?' -> 'reverted card payment'."""
    return raw.replace("_", " ").rstrip("?").strip().lower()


MENU_HEADER = (
    "Classify the customer message into exactly one of the intents below. "
    "Answer with the two-letter code only.\n\nIntents:\n"
)


def render_menu_prompt(query: str, code_by_label: dict[str, str]) -> str:
    """指令 + 按 label 字母序排的菜单 + 用户句, 冒号收尾.

    菜单行以 'XY. ' 开头 (码不带前导空格): 这样菜单里的 token 与答案位置的 ' XY'
    是两个不同的 id, 模型没法靠从上下文拷贝 id 绕过映射. 空 query 也保留消息行,
    这份提示用来量 contextual calibration 的先验.
    """
    lines = [f"{code_by_label[lab]}. {humanize_label(lab)}" for lab in sorted(code_by_label)]
    return MENU_HEADER + "\n".join(lines) + f"\n\nCustomer message: {query}\nIntent:"


def render_bare_prompt(query: str) -> str:
    """无菜单版, 只用来取 h_bare: 用户句 + 'Intent:'."""
    return f"Customer message: {query}\nIntent:"


def renormalize_k(p: list[list[float]]) -> list[list[float]]:
    """每行除以行和, 落在 k 个码上的质量归一成决策分布."""
    out = []
    for row in p:
        s = sum(row)
        out.append([v / s for v in row])
    return out


def _argsort_desc(row: list[float]) -> list[int]:
    return sorted(range(len(row)), key=lambda k: -row[k])


def topk_accuracy(q: list[list[float]], y: list[int], k: int) -> float:
    """正确类落在前 k 名的比例."""
    hits = sum(1 for row, t in zip(q, y) if t in _argsort_desc(row)[:k])
    return hits / len(y)


def brier_multiclass(q: list[list[float]], y: list[int]) -> float:
    """mean_i sum_k (q_ik - y_ik)^2. 无 1/K 因子; 二元情形下等于 brier_binary 的 2 倍."""
    total = 0.0
    for row, t in zip(q, y):
        total += sum((v - (1.0 if k == t else 0.0)) ** 2 for k, v in enumerate(row))
    return total / len(y)


def calibrate_by_prior(q: list[list[float]], prior: list[float]) -> list[list[float]]:
    """contextual calibration: 每列除以空 query 下的先验, 再按行归一."""
    return renormalize_k([[v / p for v, p in zip(row, prior)] for row in q])


NLL_EPS = 1e-12


def nll_multiclass(q: list[list[float]], y: list[int], eps: float = NLL_EPS) -> float:
    """正确类概率的平均负对数似然, 夹到 eps 防 log(0)."""
    return -sum(math.log(max(row[t], eps)) for row, t in zip(q, y)) / len(y)


def count_saturated_multiclass(q: list[list[float]], y: list[int], eps: float = NLL_EPS) -> int:
    """正确类概率低于 eps 的条数, 这些条目的 NLL 由夹取值决定."""
    return sum(1 for row, t in zip(q, y) if row[t] < eps)


def ece(conf: list[float], correct: list[bool], n_bins: int = 10) -> float:
    """等宽分箱的期望校准误差, 与 baseline-boolq.py 同一实现."""
    n = len(conf)
    sums = [0.0] * n_bins
    hits = [0] * n_bins
    counts = [0] * n_bins
    for c, ok in zip(conf, correct):
        b = min(int(c * n_bins), n_bins - 1)
        sums[b] += c
        hits[b] += 1 if ok else 0
        counts[b] += 1
    total = 0.0
    for b in range(n_bins):
        if counts[b] == 0:
            continue
        total += counts[b] / n * abs(sums[b] / counts[b] - hits[b] / counts[b])
    return total


def ece_multiclass(q: list[list[float]], y: list[int], n_bins: int = 10) -> float:
    """置信度取每行 max, 对错取 argmax == y, 然后走等宽分箱."""
    conf = [max(row) for row in q]
    correct = [_argsort_desc(row)[0] == t for row, t in zip(q, y)]
    return ece(conf, correct, n_bins=n_bins)


def result_paths(out_dir, model_id: str, mode: str, perm_seed: int, n_items: int, ts: str):
    """产物路径. 不用 Path.with_suffix —— 模型名里的小数点会被它当成扩展名切掉."""
    import pathlib

    stem = f"{ts}-banking77-{model_id.replace('/', '_')}-{mode}-perm{perm_seed}-n{n_items}"
    base = pathlib.Path(out_dir)
    return base / (stem + ".json"), base / (stem + ".npz")


# --------------------------------------------------------------------------
# 数据 (钉在上游仓库的唯一一次 commit 上, 校验 md5)
# --------------------------------------------------------------------------

DATA_COMMIT = "9d081458ff52e53cf7e848f414e6e9344e4e6696"
DATA_FILES = {
    "train": ("cec64185f4197906aabce0781ef9a19b", 10003),
    "test": ("8dcd9dc31b686c75ec1f24bf23c140cb", 3080),
}


def load_split(split: str, cache_dir) -> tuple[list[str], list[str]]:
    """返回 (texts, raw_labels). 本地没有就从 GitHub 拉, 拉完必须过 md5."""
    import csv
    import hashlib
    import pathlib
    import urllib.request

    want_md5, want_n = DATA_FILES[split]
    path = pathlib.Path(cache_dir) / f"{split}.csv"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = (
            f"https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
            f"{DATA_COMMIT}/banking_data/{split}.csv"
        )
        urllib.request.urlretrieve(url, path)
    got_md5 = hashlib.md5(path.read_bytes()).hexdigest()
    if got_md5 != want_md5:
        raise RuntimeError(f"{path}: md5 {got_md5} != {want_md5}")
    with path.open(encoding="utf-8") as f:
        rows = [(r["text"], r["category"]) for r in csv.DictReader(f)]
    if len(rows) != want_n:
        raise RuntimeError(f"{path}: {len(rows)} rows != {want_n}")
    return [t for t, _ in rows], [c for _, c in rows]


# --------------------------------------------------------------------------
# 运行
# --------------------------------------------------------------------------


def last_position_logits(lm, batch):
    """左填充下取最后一个真实 token 的隐状态, 只在该位置过输出头 (同 baseline-boolq.py)."""
    import torch

    with torch.no_grad():
        h = lm.model(**batch).last_hidden_state[:, -1, :]
        logits = lm.lm_head(h)
    return h.float(), logits.float()


def selfcheck_manual_head(lm, batch) -> float:
    """手动过头必须等于模型自带的 causal LM 输出头. 返回最大绝对偏差."""
    import torch

    _, manual = last_position_logits(lm, batch)
    with torch.no_grad():
        builtin = lm(**batch).logits[:, -1, :].float()
    return (manual - builtin).abs().max().item()


def selfcheck_padding(lm, tok, text: str, batch, slot_ids: list[int], max_length: int) -> float:
    """批内左填充的结果必须等于单条不填充的结果. 返回 77 路归一分布的最大绝对偏差.

    实测 1.7B-Base 在 fp32 下恰好 0.0, bf16 下均值 0.016 / 最大 0.074, 与填充长度无关
    (pad=0 的条目同样有 0.017). 即 bf16 在不同 batch 形状下累加顺序不同带来的舍入噪声,
    掩码本身没漏. BoolQ 的二元 softmax 把同样的 logit 抖动吞掉了, 77 路小概率彼此靠近就放大成
    可见偏差; 16 条里 1 条 argmax 翻转. 这是零样本档在 bf16 下的实测噪声地板, 当作已知量记录.
    """
    import torch

    solo = tok(text, return_tensors="pt", truncation=True, max_length=max_length).to(lm.device)
    _, l_solo = last_position_logits(lm, dict(solo))
    _, l_batch = last_position_logits(lm, batch)
    q_solo = torch.softmax(l_solo[0, slot_ids], -1)
    q_batch = torch.softmax(l_batch[0, slot_ids], -1)
    return (q_solo - q_batch).abs().max().item()


def selfcheck_answer_token_boundary(tok, prompt: str, code: str) -> bool:
    """提示 + ' XY' 编码后末 token 必须恰是 ' XY' 的单 token id, 否则读出位置对不上."""
    ids = tok.encode(prompt + " " + code, add_special_tokens=False)
    return ids[-1] == tok.encode(" " + code, add_special_tokens=False)[0]


NULL_QUERIES = ["", "N/A", "[MASK]"]


def main() -> None:
    import argparse
    import json
    import pathlib
    import time

    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--mode", default="menu", choices=["menu", "bare"])
    ap.add_argument("--split", default="test", choices=sorted(DATA_FILES))
    ap.add_argument("--code-seed", type=int, default=0, help="决定用哪 77 个码")
    ap.add_argument("--perm-seed", type=int, default=0, help="决定码<->意图的分配")
    ap.add_argument("--limit", type=int, default=0, help="0 = 全量; 否则按固定 seed 打乱后取前 N")
    ap.add_argument("--shuffle-seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--top-k", type=int, default=200)
    ap.add_argument("--n-bins", type=int, default=10)
    ap.add_argument("--data-dir", default="data/banking77")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    lm = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda").eval()

    texts_raw, raw_labels = load_split(args.split, args.data_dir)
    label_names = sorted(set(raw_labels))
    label_idx = {lab: i for i, lab in enumerate(label_names)}
    y_all = [label_idx[c] for c in raw_labels]
    # test.csv 按 label 成块排 (前 40 条全是 card_arrival), 取子集前必须打乱
    order = list(range(len(texts_raw)))
    if args.limit:
        random.Random(args.shuffle_seed).shuffle(order)
        order = sorted(order[: args.limit])
    queries = [texts_raw[i] for i in order]
    y = np.array([y_all[i] for i in order])

    codes = pick_code_tokens(tok, n=len(label_names), seed=args.code_seed)
    code_by_label = assign_codes(label_names, codes, seed=args.perm_seed)
    # 槽按 label 序排: 第 k 列就是第 k 个 label 的码, 与 y 的编号对齐
    slot_ids = [tok.encode(" " + code_by_label[lab], add_special_tokens=False)[0] for lab in label_names]
    code_ids_bare = [tok.encode(code_by_label[lab], add_special_tokens=False)[0] for lab in label_names]

    if args.mode == "menu":
        texts = [render_menu_prompt(q, code_by_label) for q in queries]
    else:
        texts = [render_bare_prompt(q) for q in queries]

    checks: dict = {}
    checks["answer_token_boundary_ok"] = all(
        selfcheck_answer_token_boundary(tok, texts[0], code_by_label[lab]) for lab in label_names
    )

    def run(batch_texts: list[str]):
        H, SL, LSE, TOPI, TOPL = [], [], [], [], []
        for s in range(0, len(batch_texts), args.batch_size):
            chunk = batch_texts[s : s + args.batch_size]
            batch = dict(
                tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=args.max_length).to("cuda")
            )
            if s == 0 and "manual_head_max_abs_logit_diff" not in checks:
                checks["manual_head_max_abs_logit_diff"] = selfcheck_manual_head(lm, batch)
                checks["padding_max_abs_q_diff"] = selfcheck_padding(
                    lm, tok, chunk[0], batch, slot_ids, args.max_length
                )
            h, logits = last_position_logits(lm, batch)
            lse = torch.logsumexp(logits, dim=-1)
            tv, ti = logits.topk(args.top_k, dim=-1)
            H.append(h.cpu().numpy().astype(np.float16))
            SL.append(logits[:, slot_ids].cpu().numpy())
            LSE.append(lse.cpu().numpy())
            TOPI.append(ti.cpu().numpy().astype(np.int32))
            TOPL.append(tv.cpu().numpy().astype(np.float16))
        return (np.concatenate(H), np.concatenate(SL), np.concatenate(LSE), np.concatenate(TOPI), np.concatenate(TOPL))

    H, SL, LSE, TOPI, TOPL = run(texts)
    P = np.exp(SL - LSE[:, None])  # 每个码的全词表概率 (N, 77)
    m_answer = P.sum(axis=1)
    Q = np.array(renormalize_k(P.tolist()))

    report: dict = {}
    prior_sl = None
    if args.mode == "menu":
        # 空 query 先验: 三种空输入各一次 forward, 概率取平均
        null_texts = [render_menu_prompt(nq, code_by_label) for nq in NULL_QUERIES]
        _, prior_sl, prior_lse, prior_topi, _ = run(null_texts)
        prior_p = np.exp(prior_sl - prior_lse[:, None])
        prior = np.array(renormalize_k(prior_p.tolist())).mean(axis=0)
        Q_cal = np.array(calibrate_by_prior(Q.tolist(), prior.tolist()))

        def metrics(q: np.ndarray) -> dict:
            ql, yl = q.tolist(), y.tolist()
            pred = q.argmax(axis=1)
            per_class = [float((pred[y == k] == k).mean()) if (y == k).any() else None for k in range(len(label_names))]
            return {
                "accuracy": topk_accuracy(ql, yl, k=1),
                "top5_accuracy": topk_accuracy(ql, yl, k=5),
                "nll": nll_multiclass(ql, yl),
                "n_saturated": count_saturated_multiclass(ql, yl),
                "brier": brier_multiclass(ql, yl),
                "ece": ece_multiclass(ql, yl, n_bins=args.n_bins),
                "conf_mean": float(q.max(axis=1).mean()),
                "pred_histogram_max_frac": float(np.bincount(pred, minlength=len(label_names)).max() / len(pred)),
                "per_class_accuracy": per_class,
            }

        report["raw"] = metrics(Q)
        report["prior_calibrated"] = metrics(Q_cal)
        report["shared"] = {
            "m_answer_mean": float(m_answer.mean()),
            "m_answer_p05": float(np.percentile(m_answer, 5)),
            "m_answer_p95": float(np.percentile(m_answer, 95)),
            "prior_max": float(prior.max()),
            "prior_argmax_label": label_names[int(prior.argmax())],
            "prior_entropy_bits": float(-(prior * np.log2(prior)).sum()),
            "prior_uniform_entropy_bits": float(math.log2(len(label_names))),
        }
        # 自检: 正确码进 top-k 的比例; top1 直方图
        in_topk = [slot_ids[t] in set(row.tolist()) for t, row in zip(y.tolist(), TOPI)]
        checks["gold_code_in_topk_rate"] = float(np.mean(in_topk))
        checks["top1_is_some_code_rate"] = float(np.mean([int(r[0]) in set(slot_ids) for r in TOPI]))
    top1 = TOPI[:, 0]
    uniq, cnt = np.unique(top1, return_counts=True)
    order_ = np.argsort(-cnt)[:8]
    checks["top1_token_histogram"] = [[tok.decode([int(uniq[i])]), int(cnt[i])] for i in order_]

    meta = {
        "model_id": args.model,
        "model_revision": pathlib.Path(snapshot_download(args.model, local_files_only=True)).name,
        "dtype": "bfloat16",
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "dataset": "PolyAI-LDN/task-specific-datasets banking_data",
        "dataset_commit": DATA_COMMIT,
        "split": args.split,
        "n_items": int(len(y)),
        "n_labels": len(label_names),
        "limit": args.limit,
        "shuffle_seed": args.shuffle_seed,
        "mode": args.mode,
        "code_seed": args.code_seed,
        "perm_seed": args.perm_seed,
        "codes_by_label": code_by_label,
        "null_queries": NULL_QUERIES if args.mode == "menu" else None,
        "template_rendered_example": texts[0],
        "prompt_tokens_example": len(tok.encode(texts[0], add_special_tokens=False)),
        "top_k": args.top_k,
        "ece_bins": args.n_bins,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "elapsed_sec": round(time.time() - t0, 1),
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
    }
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    json_path, npz_path = result_paths(
        out, args.model, args.mode, args.perm_seed, len(y), time.strftime("%Y%m%d-%H%M%S")
    )
    arrays = dict(
        h=H, slot_logits=SL, slot_ids=np.array(slot_ids), code_ids_bare=np.array(code_ids_bare),
        logsumexp=LSE, topk_ids=TOPI, topk_logits=TOPL, labels=y, item_index=np.array(order),
    )
    if prior_sl is not None:
        arrays["prior_slot_logits"] = prior_sl
    np.savez_compressed(npz_path, **arrays)
    payload = {"meta": meta, "selfchecks": checks, "metrics": report, "label_names": label_names}
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    print(json.dumps({k: v for k, v in payload.items() if k != "label_names"}, indent=2, ensure_ascii=False))
    print(f"\n→ {json_path}")


if __name__ == "__main__":
    main()
