#!/usr/bin/env python
"""BoolQ baseline: Qwen3-0.6B-Base 零训练、全词表读出.

做一件事: 给 passage 和 question, 一次 forward, 在最后位置记下
  - 全词表的 logsumexp (一个数, 用它可还原任意 token 的精确全词表概率)
  - 一批候选 token 的原始 logit (yes / no / unknown 等的各种大小写与空格变体)
  - top-k (模型实际最想说什么)
然后离线算指标. 概率不在候选之间强制归一 —— 剩余质量本身是弃权信号.
"""

from __future__ import annotations

import math

# --------------------------------------------------------------------------
# 纯函数 (tests/test_baseline_metrics.py 覆盖)
# --------------------------------------------------------------------------


def prob_from_logsumexp(logit: float, lse: float) -> float:
    """由单个 logit 与全向量的 logsumexp 还原精确的全词表概率."""
    return math.exp(logit - lse)


def renormalize_binary(p_yes: float, p_no: float) -> tuple[float, float]:
    """把落在 {yes, no} 上的质量重新归一, 得到二元决策分布."""
    total = p_yes + p_no
    return p_yes / total, p_no / total


NLL_EPS = 1e-12


def nll(q_correct: list[float], eps: float = NLL_EPS) -> float:
    """正确类概率序列的平均负对数似然. 概率夹到下限 eps, 避免饱和到 0 时 log(0) 报错."""
    return -sum(math.log(max(q, eps)) for q in q_correct) / len(q_correct)


def count_saturated(q_correct: list[float], eps: float = NLL_EPS) -> int:
    """低于 eps 的条目数. 这些条目的 NLL 由夹取值决定, 是该指标失真程度的可审计信号."""
    return sum(1 for q in q_correct if q < eps)


def brier_binary(q_yes: list[float], y: list[int]) -> float:
    """二元 Brier 分数, 约定为 mean((q_yes - y)^2), y=1 表示 yes."""
    return sum((q - t) ** 2 for q, t in zip(q_yes, y)) / len(q_yes)


def ece(conf: list[float], correct: list[bool], n_bins: int = 10) -> float:
    """等宽分箱的期望校准误差. 空箱不贡献数值."""
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


def build_slot_table(tokenizer, spec: dict[str, list[str]]) -> dict[str, dict[str, int]]:
    """把候选写法编成 token id, 只保留单 token 的变体.

    多 token 的写法必须剔除: 取它首个子词的 logit 是一个静默的错误读数.
    """
    table: dict[str, dict[str, int]] = {}
    for name, variants in spec.items():
        kept: dict[str, int] = {}
        for v in variants:
            ids = tokenizer.encode(v, add_special_tokens=False)
            if len(ids) == 1:
                kept[v] = ids[0]
        table[name] = kept
    return table


# --------------------------------------------------------------------------
# 候选槽与模板 (跑之前冻结, 改动必须改 ID)
# --------------------------------------------------------------------------

SLOT_SPEC: dict[str, list[str]] = {
    # 提示里明确给出的三个答案
    "yes": ["yes", " yes", "Yes", " Yes", "YES", " YES"],
    "no": ["no", " no", "No", " No", "NO", " NO"],
    "unknown": ["unknown", " unknown", "Unknown", " Unknown"],
    # 对照组: 提示里没出现, 用来看剩余质量流去了哪
    "maybe": ["maybe", " maybe", "Maybe", " Maybe"],
    "true": ["true", " true", "True", " True"],
    "false": ["false", " false", "False", " False"],
    "none": ["none", " none", "None", " None"],
}

TEMPLATE_BODY = (
    "{passage}\n\n"
    "Question: {question}?\n"
    "Answer with one word - yes, no, or unknown"
)

TEMPLATES: dict[str, dict[str, str]] = {
    # 补全式: 裸文本, 以冒号收尾 -> 延续带前导空格 -> 读空格变体
    "passage-question-oneword-v1": {"kind": "completion", "suffix": ":"},
    # 对话式: 走模型自带 chat template 且强制关闭思考模式.
    # 关思考后提示以 "</think>\n\n" 收尾 -> 延续无前导空格 -> 读无空格变体.
    # 开着思考的话下一个 token 是 <think>, 读出取到的会是与答案无关的数.
    "chat-oneword-v1": {"kind": "chat", "suffix": "."},
}


def render_prompt(template_id: str, tokenizer, passage: str, question: str) -> str:
    """按模板 id 渲染提示. 对话式模板需要 tokenizer 来套 chat template."""
    spec = TEMPLATES[template_id]
    body = TEMPLATE_BODY.format(passage=passage.strip(), question=question.strip()) + spec["suffix"]
    if spec["kind"] == "completion":
        return body
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": body}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


# --------------------------------------------------------------------------
# 运行
# --------------------------------------------------------------------------


def result_paths(out_dir, model_id: str, template_id: str, n_items: int, ts: str):
    """产物路径. 不用 Path.with_suffix —— 模型名里的小数点会被它当成扩展名切掉."""
    import pathlib

    stem = f"{ts}-boolq-{model_id.replace('/', '_')}-{template_id}-n{n_items}"
    base = pathlib.Path(out_dir)
    return base / (stem + ".json"), base / (stem + ".npz")


def last_position_logits(lm, batch):
    """左填充下取最后一个真实 token 的隐状态, 只在该位置过输出头.

    绕开 `lm(**batch).logits` 的 (B, L, V) 张量: B=16 L=768 V=151936 的 bf16
    是 3.7 GB, 而这里只有 (B, V) = 4.9 MB.
    """
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


def selfcheck_padding(lm, tok, text: str, batch) -> float:
    """批内左填充的结果必须等于单条不填充的结果. 返回 P(yes) 的绝对偏差.

    比较归一化后的概率而非原始 logit: bf16 在 logit 量级 25 上的分辨率约 0.1,
    原始值本来就对不齐, 而我们真正依赖的是概率.
    """
    import torch

    yes_id, no_id = tok.encode(" yes", add_special_tokens=False)[0], tok.encode(" no", add_special_tokens=False)[0]
    solo = tok(text, return_tensors="pt", truncation=True, max_length=768).to(lm.device)
    _, l_solo = last_position_logits(lm, dict(solo))
    _, l_batch = last_position_logits(lm, batch)

    def p_yes(logits, row):
        py = math.exp(logits[row, yes_id].item() - torch.logsumexp(logits[row], -1).item())
        pn = math.exp(logits[row, no_id].item() - torch.logsumexp(logits[row], -1).item())
        return py / (py + pn)

    return abs(p_yes(l_solo, 0) - p_yes(l_batch, 0))


def main() -> None:
    import argparse
    import json
    import pathlib
    import time

    import numpy as np
    import torch
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from sklearn.metrics import roc_auc_score
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--template", default="passage-question-oneword-v1", choices=sorted(TEMPLATES))
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=768)
    ap.add_argument("--top-k", type=int, default=200)
    ap.add_argument("--n-bins", type=int, default=10)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    lm = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda").eval()

    slots = build_slot_table(tok, SLOT_SPEC)
    slot_ids = sorted({i for v in slots.values() for i in v.values()})
    id_pos = {i: p for p, i in enumerate(slot_ids)}

    ds = load_dataset("google/boolq", split="validation").select(range(args.limit))
    texts = [render_prompt(args.template, tok, p, q) for p, q in zip(ds["passage"], ds["question"])]
    labels = [1 if a else 0 for a in ds["answer"]]

    H, SL, LSE, TOPI, TOPL = [], [], [], [], []
    checks: dict[str, float] = {}
    for s in range(0, len(texts), args.batch_size):
        chunk = texts[s : s + args.batch_size]
        batch = dict(
            tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=args.max_length).to("cuda")
        )
        if s == 0:
            checks["manual_head_max_abs_logit_diff"] = selfcheck_manual_head(lm, batch)
            checks["padding_max_abs_pyes_diff"] = selfcheck_padding(lm, tok, chunk[0], batch)
        h, logits = last_position_logits(lm, batch)
        lse = torch.logsumexp(logits, dim=-1)
        tv, ti = logits.topk(args.top_k, dim=-1)
        H.append(h.cpu().numpy().astype(np.float16))
        SL.append(logits[:, slot_ids].cpu().numpy())
        LSE.append(lse.cpu().numpy())
        TOPI.append(ti.cpu().numpy().astype(np.int32))
        TOPL.append(tv.cpu().numpy().astype(np.float16))
    H = np.concatenate(H)
    SL = np.concatenate(SL)
    LSE = np.concatenate(LSE)
    TOPI = np.concatenate(TOPI)
    TOPL = np.concatenate(TOPL)

    def p_of(name: str, variants: list[str] | None = None) -> np.ndarray:
        """指定名字(可限定变体)的全词表概率之和."""
        ids = [slots[name][v] for v in (variants or list(slots[name]))]
        cols = [id_pos[i] for i in ids]
        return np.exp(SL[:, cols] - LSE[:, None]).sum(axis=1)

    readouts = {
        "space_lower": {"yes": [" yes"], "no": [" no"], "unknown": [" unknown"]},
        "nospace_lower": {"yes": ["yes"], "no": ["no"], "unknown": ["unknown"]},
        "marginal_all": {"yes": None, "no": None, "unknown": None},
    }

    y = np.array(labels)
    report: dict[str, dict] = {}
    for rname, cfg in readouts.items():
        p_yes, p_no = p_of("yes", cfg["yes"]), p_of("no", cfg["no"])
        p_unk = p_of("unknown", cfg["unknown"])
        m = p_yes + p_no
        q_yes = p_yes / m
        pred = (q_yes >= 0.5).astype(int)
        conf = np.maximum(q_yes, 1 - q_yes)
        ok = pred == y
        q_correct = [float(q) for q in np.where(y == 1, q_yes, 1 - q_yes)]
        report[rname] = {
            "accuracy": float(ok.mean()),
            "nll": nll(q_correct),
            "n_saturated": count_saturated(q_correct),
            "brier": brier_binary([float(v) for v in q_yes], [int(v) for v in y]),
            "ece": ece([float(v) for v in conf], [bool(v) for v in ok], n_bins=args.n_bins),
            "m_answer_mean": float(m.mean()),
            "m_answer_p05": float(np.percentile(m, 5)),
            "m_answer_p95": float(np.percentile(m, 95)),
            "p_unknown_mean": float(p_unk.mean()),
            "residual_mean": float(1.0 - m.mean() - p_unk.mean()),
            "auroc_m_answer_vs_correct": float(roc_auc_score(ok, m)) if 0 < ok.mean() < 1 else None,
            "auroc_conf_vs_correct": float(roc_auc_score(ok, conf)) if 0 < ok.mean() < 1 else None,
        }

    # 自检三: 正确答案的任一变体必须进 top-k, 否则模板与读出对不上
    want = [set(slots["yes"].values()) if t == 1 else set(slots["no"].values()) for t in labels]
    in_topk = [len(w & set(row.tolist())) > 0 for w, row in zip(want, TOPI)]
    checks["gold_token_in_topk_rate"] = float(np.mean(in_topk))
    top1 = TOPI[:, 0]
    uniq, cnt = np.unique(top1, return_counts=True)
    order = np.argsort(-cnt)[:8]
    checks["top1_token_histogram"] = [
        [tok.decode([int(uniq[i])]), int(cnt[i])] for i in order
    ]

    meta = {
        "model_id": args.model,
        "model_revision": pathlib.Path(snapshot_download(args.model, local_files_only=True)).name,
        "dtype": "bfloat16",
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "dataset": "google/boolq",
        "split": "validation",
        "n_items": int(len(labels)),
        "label_yes_frac": float(y.mean()),
        "template_id": args.template,
        "template_kind": TEMPLATES[args.template]["kind"],
        "template_rendered_example": texts[0],
        "slot_table": slots,
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
        out, args.model, args.template, len(labels), time.strftime("%Y%m%d-%H%M%S")
    )
    np.savez_compressed(
        npz_path, h=H, slot_logits=SL, slot_ids=np.array(slot_ids),
        logsumexp=LSE, topk_ids=TOPI, topk_logits=TOPL, labels=y,
    )
    payload = {"meta": meta, "selfchecks": checks, "metrics": report}
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\n→ {json_path}")


if __name__ == "__main__":
    main()
