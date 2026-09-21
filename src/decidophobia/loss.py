"""训练目标: 只在这条样本给出的 k 个槽上做 softmax, 交叉熵到正确槽.

与推理形状一致 —— 调用方给几个选项就在几个上归一. 词表里其余 15 万个 token
不进分母: 这里训的是「选哪个」, 「要不要说 D-token 而非别的」由嵌入行的更新顺带解决.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def gather_slot_logits(logits: torch.Tensor, slot_ids: torch.Tensor) -> torch.Tensor:
    """(B, V) x (B, K) -> (B, K); slot_ids == -1 的位置填 -inf."""
    mask = slot_ids >= 0
    got = logits.gather(1, slot_ids.clamp_min(0))
    return got.masked_fill(~mask, float("-inf"))


def slot_cross_entropy(logits: torch.Tensor, slot_ids: torch.Tensor, gold: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(gather_slot_logits(logits, slot_ids), gold)
