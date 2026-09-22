#!/usr/bin/env python
"""训练 256 槽决策头 (LoRA + D-token 嵌入), Banking77 上按类留出测泛化.

  PYTHONPATH=src .venv/bin/python scripts/train.py --steps 2000 --held-out 17 --trainable attn
  .venv/bin/tensorboard --logdir runs

产出 (--out 目录):
  log.jsonl     每次评估一行 (step 0 是训练前基线)
  result.json   配置 + 类切分 + 全部评估记录
  trained.pt    LoRA 权重 + 256 个 D 行嵌入
  tb/           TensorBoard 事件
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import time

import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer

from decidophobia.banking77 import load_banking77
from decidophobia.data import build_examples, class_split
from decidophobia.model import LORA_TARGETS, prepare_model
from decidophobia.thermal import ThermalGuard
from decidophobia.tokens import install_d_tokens
from decidophobia.train import TrainConfig, save_trained, train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--trainable", default="attn", choices=sorted(LORA_TARGETS),
                    help="放开的范围: d-only 只训 D 行; attn 加 attention LoRA; attn-mlp 再加 MLP LoRA")
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lr-lora", type=float, default=1e-4)
    ap.add_argument("--lr-embed", type=float, default=1e-3)
    ap.add_argument("--lr-schedule", default="cosine", choices=["constant", "cosine"])
    ap.add_argument("--warmup", type=int, default=100, help="线性 warmup 步数")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=10)
    ap.add_argument("--k-eval", type=int, default=10, help="评估菜单长度 (固定)")
    ap.add_argument("--held-out", type=int, default=17, help="留出的类数, 训练里完全不出现")
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-limit", type=int, default=0, help="每个评估集最多用几条 (0 = 全部)")
    ap.add_argument("--temp-max", type=float, default=85.0, help="CPU Tctl 超过就暂停 (°C)")
    ap.add_argument("--temp-cooldown", type=float, default=20.0, help="每次暂停多少秒")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", default="data/banking77")
    ap.add_argument("--out", default=None, help="默认 runs/<时间戳>-<trainable>")
    args = ap.parse_args()

    out = pathlib.Path(args.out or f"runs/{time.strftime('%Y%m%d-%H%M%S')}-{args.trainable}-{args.lr_schedule}")
    out.mkdir(parents=True, exist_ok=True)

    tr, te = load_banking77(args.data_dir)
    split = class_split(len(tr.names), args.held_out, seed=args.seed)

    # 评估集固定 seed 组一次, 训练全程用同一份
    erng = random.Random(args.seed + 1)
    kr = (args.k_eval, args.k_eval)
    eval_sets = {
        "seen": build_examples(te.queries, te.labels, split.train, kr, erng),
        "unseen": build_examples(te.queries, te.labels, split.held_out, kr, erng),
    }
    if args.eval_limit:
        for k in eval_sets:
            random.Random(args.seed + 2).shuffle(eval_sets[k])
            eval_sets[k] = eval_sets[k][: args.eval_limit]

    tok = AutoTokenizer.from_pretrained(args.model)
    d_ids = install_d_tokens(tok)
    lm = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda")
    m = prepare_model(lm, d_ids, args.lora_r, args.lora_alpha, args.lora_dropout, trainable=args.trainable)

    cfg = TrainConfig(
        steps=args.steps, batch_size=args.batch_size, k_range=(args.k_min, args.k_max),
        k_max=max(args.k_max, args.k_eval), lr_lora=args.lr_lora, lr_embed=args.lr_embed,
        weight_decay=args.weight_decay, lr_schedule=args.lr_schedule, warmup_steps=args.warmup,
        eval_every=args.eval_every, seed=args.seed,
    )
    guard = ThermalGuard(max_c=args.temp_max, cooldown_s=args.temp_cooldown)
    writer = SummaryWriter(log_dir=str(out / "tb"))
    writer.add_text("args", json.dumps(vars(args), indent=2), 0)
    n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"trainable={args.trainable}  params {n_train:,}  train classes {len(split.train)}  "
          f"held-out {len(split.held_out)}  eval seen {len(eval_sets['seen'])}  "
          f"unseen {len(eval_sets['unseen'])}  tctl {guard.read()}  → {out}", flush=True)
    history = train(
        m, tok, tr.names, d_ids, tr.queries, tr.labels, split.train, eval_sets, cfg,
        log_path=out / "log.jsonl", writer=writer, guard=guard,
    )
    writer.close()
    save_trained(m, d_ids, cfg, out / "trained.pt")
    (out / "result.json").write_text(json.dumps({
        "args": vars(args),
        "trainable_params": n_train,
        "split": {"train": split.train, "held_out": split.held_out,
                  "held_out_names": [tr.names[c] for c in split.held_out]},
        "history": history,
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
    }, indent=2))
    print(f"→ {out}")


if __name__ == "__main__":
    main()
