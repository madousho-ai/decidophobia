#!/usr/bin/env python
"""训练 256 槽决策头 (LoRA + D-token 嵌入). 数据集: banking77 (按类留出测泛化) / boolq / both.

  PYTHONPATH=src .venv/bin/python scripts/train.py --dataset banking77 --steps 2000
  PYTHONPATH=src .venv/bin/python scripts/train.py --dataset boolq --init runs/<b77>/trained.pt --steps 0   # 跨任务零训练评估
  PYTHONPATH=src .venv/bin/python scripts/train.py --dataset both --steps 2000                            # 联合训练
  .venv/bin/tensorboard --logdir runs

产出 (--out 目录):
  log.jsonl     每次评估一行 (step 0 是训练前 / 加载后的基线)
  result.json   配置 + 类切分 + 全部评估记录
  trained.pt    LoRA 权重 + 256 个 D 行嵌入 (--steps 0 时不写, 保住 --init 那份)
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

from decidophobia.data import class_split
from decidophobia.model import LORA_TARGETS, prepare_model
from decidophobia.prompt import DEFAULT_LAYOUT, LAYOUTS
from decidophobia.thermal import ThermalGuard
from decidophobia.tokens import install_d_tokens, install_type_tokens
from decidophobia.train import EvalSet, TrainConfig, load_trained, save_trained, train

DATASETS = ("banking77", "boolq", "both")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="banking77", choices=DATASETS)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--init", default=None, help="从这份 trained.pt 加载 LoRA + D 行再开始 (或配 --steps 0 只评估)")
    ap.add_argument("--trainable", default="attn", choices=sorted(LORA_TARGETS),
                    help="放开的范围: d-only 只训 D 行; attn 加 attention LoRA; attn-mlp 再加 MLP LoRA")
    ap.add_argument("--layout", default=DEFAULT_LAYOUT, choices=LAYOUTS,
                    help="context-first: 上下文在前, 前缀可作 KV cache 共享 (默认); menu-first: 菜单在前, 对照组")
    ap.add_argument("--type-marker", action="store_true",
                    help="问句标签写成 'Question (<|bool|>):', 类型 token 随 D 行一起训")
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lr-lora", type=float, default=1e-4)
    ap.add_argument("--lr-embed", type=float, default=1e-3)
    ap.add_argument("--lr-schedule", default="cosine", choices=["constant", "cosine"])
    ap.add_argument("--warmup", type=int, default=100, help="线性 warmup 步数")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=300, help="0 = 只评估")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=512, help="超长提示从左截, BoolQ passage p95 约 256 token")
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=10)
    ap.add_argument("--k-eval", type=int, default=10, help="Banking77 评估菜单长度 (固定); BoolQ 恒为 2")
    ap.add_argument("--held-out", type=int, default=17, help="Banking77 留出的类数, 训练里完全不出现")
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-batch-size", type=int, default=16)
    ap.add_argument("--eval-limit", type=int, default=0, help="每个评估集最多用几条 (0 = 全部)")
    ap.add_argument("--temp-max", type=float, default=85.0, help="CPU Tctl 超过就暂停 (°C)")
    ap.add_argument("--temp-cooldown", type=float, default=20.0, help="每次暂停多少秒")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", default="data/banking77")
    ap.add_argument("--out", default=None, help="默认 runs/<时间戳>-<dataset>-<trainable>-<schedule>-<layout>")
    args = ap.parse_args()

    tag = "-qtype" if args.type_marker else ""
    out = pathlib.Path(args.out or f"runs/{time.strftime('%Y%m%d-%H%M%S')}-{args.dataset}-{args.trainable}-{args.lr_schedule}-{args.layout}{tag}")
    out.mkdir(parents=True, exist_ok=True)

    # ---- 数据: 每个数据集给一个 sampler 和若干评估集 ------------------------------
    erng = random.Random(args.seed + 1)
    samplers, eval_sets, split_info = [], {}, {}
    if args.dataset in ("banking77", "both"):
        from decidophobia.banking77 import load_banking77

        tr, te = load_banking77(args.data_dir)
        split = class_split(len(tr.names), args.held_out, seed=args.seed)
        kr = (args.k_eval, args.k_eval)
        eval_sets["seen"] = EvalSet(te.build_examples(split.train, kr, erng), args.eval_batch_size)
        eval_sets["unseen"] = EvalSet(te.build_examples(split.held_out, kr, erng), args.eval_batch_size)
        samplers.append(lambda n, rng: tr.sample_examples(split.train, (args.k_min, args.k_max), n, rng))
        split_info = {"train": split.train, "held_out": split.held_out,
                      "held_out_names": [tr.names[c] for c in split.held_out]}
    if args.dataset in ("boolq", "both"):
        from decidophobia.boolq import load_boolq

        btr, bva = load_boolq()
        eval_sets["boolq"] = EvalSet(bva.build_examples([0, 1], (2, 2), erng), max(1, args.eval_batch_size // 2), pos_class=1)
        samplers.append(lambda n, rng: btr.sample_examples([0, 1], (2, 2), n, rng))
    if args.eval_limit:
        for k, es in eval_sets.items():
            exs = list(es.examples)
            random.Random(args.seed + 2).shuffle(exs)
            eval_sets[k] = EvalSet(exs[: args.eval_limit], es.batch_size, es.pos_class)

    def sample_fn(n, rng):
        """both 时一批里一半一半 (第一个 sampler 拿零头), 再打乱."""
        parts = [n // len(samplers)] * len(samplers)
        parts[0] += n - sum(parts)
        out = [ex for s, c in zip(samplers, parts) for ex in s(c, rng)]
        rng.shuffle(out)
        return out

    # ---- 模型 ---------------------------------------------------------------------
    tok = AutoTokenizer.from_pretrained(args.model)
    d_ids = install_d_tokens(tok)
    t_ids = install_type_tokens(tok)
    train_ids = d_ids + t_ids  # 类型行永远放开; 不带 --type-marker 时它们不出现在提示里, 梯度为零、原地不动
    lm = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda")
    m = prepare_model(lm, train_ids, args.lora_r, args.lora_alpha, args.lora_dropout, trainable=args.trainable)
    init_cfg = load_trained(m, train_ids, args.init) if args.init else None

    cfg = TrainConfig(
        steps=args.steps, batch_size=args.batch_size, k_max=max(args.k_max, args.k_eval), max_length=args.max_length,
        lr_lora=args.lr_lora, lr_embed=args.lr_embed, weight_decay=args.weight_decay,
        lr_schedule=args.lr_schedule, warmup_steps=args.warmup, layout=args.layout, type_marker=args.type_marker,
        eval_every=args.eval_every, seed=args.seed,
    )
    guard = ThermalGuard(max_c=args.temp_max, cooldown_s=args.temp_cooldown)
    writer = SummaryWriter(log_dir=str(out / "tb"))
    writer.add_text("args", json.dumps(vars(args), indent=2), 0)
    n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"dataset={args.dataset} trainable={args.trainable} layout={args.layout} type_marker={args.type_marker} params {n_train:,}  "
          f"init={args.init or '-'}  eval " + " ".join(f"{k}={len(v.examples)}" for k, v in eval_sets.items())
          + f"  tctl {guard.read()}  → {out}", flush=True)
    history = train(m, tok, d_ids, sample_fn, eval_sets, cfg, log_path=out / "log.jsonl", writer=writer, guard=guard)
    writer.close()
    if args.steps > 0:
        save_trained(m, train_ids, cfg, out / "trained.pt")
    (out / "result.json").write_text(json.dumps({
        "args": vars(args),
        "trainable_params": n_train,
        "init_config": init_cfg,
        "split": split_info,
        "history": history,
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
    }, indent=2))
    print(f"→ {out}")


if __name__ == "__main__":
    main()
