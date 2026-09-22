"""训练循环与评估. 每一步的菜单都是现组的: 同一条 query 每次见到的选项集和位置都不同."""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass

import torch

from decidophobia.batch import collate
from decidophobia.data import MenuExample, compose_menu
from decidophobia.loss import gather_slot_logits
from decidophobia.metrics import summarize
from decidophobia.model import last_logits, trainable_param_groups
from decidophobia.schedule import lr_scale


@dataclass
class TrainConfig:
    steps: int = 300
    batch_size: int = 8
    k_range: tuple[int, int] = (2, 10)
    k_max: int = 16
    lr_lora: float = 1e-4
    lr_embed: float = 1e-3
    weight_decay: float = 0.0
    lr_schedule: str = "constant"  # constant | cosine
    warmup_steps: int = 0
    eval_every: int = 100
    eval_batch_size: int = 16
    log_every: int = 20
    seed: int = 0


def sample_examples(
    queries: list[str], labels: list[int], classes: list[int],
    k_range: tuple[int, int], n: int, rng: random.Random,
) -> list[MenuExample]:
    """从 (queries, labels) 里抽 n 条 label 落在 classes 内的, 各配一个现组的菜单."""
    allowed = set(classes)
    pool = [i for i, lab in enumerate(labels) if lab in allowed]
    out = []
    for i in rng.sample(pool, n):
        k = rng.randint(*k_range)
        opts, gi = compose_menu(labels[i], classes, k, rng)
        out.append(MenuExample(query=queries[i], options=opts, gold_idx=gi, label=labels[i]))
    return out


@torch.no_grad()
def evaluate(m, tok, names, d_ids, examples: list[MenuExample], k_max: int, batch_size: int) -> dict:
    """在给定样本上算 summarize() 那组指标. 概率只在各自菜单的 k 个槽上归一."""
    was_training = m.training
    m.eval()
    Q, Y = [], []
    for s in range(0, len(examples), batch_size):
        chunk = examples[s : s + batch_size]
        b = collate(chunk, tok, names, d_ids, k_max)
        b = {k: v.to("cuda") for k, v in b.items()}
        logits = last_logits(m, b["input_ids"], b["attention_mask"])
        q = torch.softmax(gather_slot_logits(logits, b["slot_ids"]), dim=-1)  # pad 槽 exp(-inf)=0
        Q.extend(q.cpu().tolist())
        Y.extend(b["gold"].tolist())
    if was_training:
        m.train()
    out = summarize(Q, Y)
    out["n"] = len(Y)
    return out


def train(
    m, tok, names: dict[int, str], d_ids: list[int],
    train_queries: list[str], train_labels: list[int], train_classes: list[int],
    eval_sets: dict[str, list[MenuExample]],
    cfg: TrainConfig, log_path=None, writer=None, guard=None,
) -> list[dict]:
    """跑 cfg.steps 步. 返回评估记录 (含 step 0 的训练前基线). 每条记录也追加写到 log_path.

    writer: torch.utils.tensorboard.SummaryWriter, 可选. 标量分三组:
      train/loss, train/lr_*        每 log_every 步
      eval/<set>/<metric>           每次评估
      sys/tctl_c, sys/thermal_waits 温度与被温度闸拦下的次数
    guard: ThermalGuard, 可选. 每步之前和每次评估之前各问一次.
    """
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)
    opt = torch.optim.AdamW(
        trainable_param_groups(m, cfg.lr_lora, cfg.lr_embed), weight_decay=cfg.weight_decay
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_scale(s, cfg.warmup_steps, cfg.steps, cfg.lr_schedule)
    )
    history: list[dict] = []
    log_f = open(log_path, "a") if log_path else None
    waits = 0

    def tctl() -> float | None:
        return guard.read() if guard else None

    def do_eval(step: int, train_loss: float | None):
        nonlocal waits
        if guard:
            waits += guard.wait()
        rec = {"step": step, "train_loss": train_loss, "t": round(time.time() - t0, 1),
               "tctl_c": tctl(), "thermal_waits": waits}
        for name, exs in eval_sets.items():
            rec[name] = evaluate(m, tok, names, d_ids, exs, cfg.k_max, cfg.eval_batch_size)
            if writer:
                for k, v in rec[name].items():
                    writer.add_scalar(f"eval/{name}/{k}", v, step)
        if writer:
            if rec["tctl_c"] is not None:
                writer.add_scalar("sys/tctl_c", rec["tctl_c"], step)
            writer.add_scalar("sys/thermal_waits", waits, step)
            writer.flush()
        history.append(rec)
        line = json.dumps(rec)
        print(line, flush=True)
        if log_f:
            log_f.write(line + "\n")
            log_f.flush()

    t0 = time.time()
    m.train()
    do_eval(0, None)
    running = 0.0
    for step in range(1, cfg.steps + 1):
        if guard:
            waits += guard.wait()
        exs = sample_examples(train_queries, train_labels, train_classes, cfg.k_range, cfg.batch_size, rng)
        b = collate(exs, tok, names, d_ids, cfg.k_max)
        b = {k: v.to("cuda") for k, v in b.items()}
        logits = last_logits(m, b["input_ids"], b["attention_mask"])
        loss = torch.nn.functional.cross_entropy(gather_slot_logits(logits, b["slot_ids"]), b["gold"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        running += loss.item()
        if step % cfg.log_every == 0:
            avg = running / cfg.log_every
            t = tctl()
            print(f"step {step:5d}  loss {avg:.4f}  {time.time() - t0:.0f}s"
                  + (f"  tctl {t:.0f}°C" if t is not None else ""), flush=True)
            if writer:
                writer.add_scalar("train/loss", avg, step)
                for i, g in enumerate(opt.param_groups):
                    writer.add_scalar(f"train/lr_group{i}", g["lr"], step)
                if t is not None:
                    writer.add_scalar("sys/tctl_c", t, step)
            running = 0.0
        if step % cfg.eval_every == 0 or step == cfg.steps:
            do_eval(step, loss.item())
    if log_f:
        log_f.close()
    return history


def save_trained(m, d_ids: list[int], cfg: TrainConfig, path) -> None:
    """只存会变的部分: LoRA 权重 + 256 个 D 行 + 配置. 基模照 model_id 重新加载."""
    emb = m.get_input_embeddings().weight
    state = {n: p.detach().cpu() for n, p in m.named_parameters() if p.requires_grad and "lora_" in n}
    torch.save(
        {"lora": state, "d_embed": emb[d_ids].detach().cpu(), "d_ids": d_ids, "config": asdict(cfg)},
        path,
    )
