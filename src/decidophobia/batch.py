"""把一批 MenuExample 变成张量. 左填充, 答案位置永远是最后一列."""

from __future__ import annotations

import torch

from decidophobia.data import MenuExample
from decidophobia.prompt import DEFAULT_LAYOUT, render_menu


def collate(
    examples: list[MenuExample],
    tokenizer,
    names: dict[int, str],
    d_ids: list[int],
    k_max: int,
    layout: str = DEFAULT_LAYOUT,
) -> dict[str, torch.Tensor]:
    """返回
      input_ids      (B, L)  左填充
      attention_mask (B, L)
      slot_ids       (B, k_max)  第 j 列是 <|Dj|> 的 id, 超出该样本菜单长度的位置填 -1
      gold           (B,)        正确选项在菜单里的位置
    """
    texts = [render_menu(ex, names, layout) for ex in examples]
    pad = tokenizer.pad_token_id
    encs = [tokenizer.encode(t, add_special_tokens=False) for t in texts]
    L = max(len(e) for e in encs)
    input_ids = torch.full((len(encs), L), pad, dtype=torch.long)
    attn = torch.zeros((len(encs), L), dtype=torch.long)
    for i, e in enumerate(encs):
        input_ids[i, L - len(e) :] = torch.tensor(e)
        attn[i, L - len(e) :] = 1
    slot_ids = torch.full((len(examples), k_max), -1, dtype=torch.long)
    for i, ex in enumerate(examples):
        k = len(ex.options)
        slot_ids[i, :k] = torch.tensor(d_ids[:k])
    gold = torch.tensor([ex.gold_idx for ex in examples], dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attn, "slot_ids": slot_ids, "gold": gold}
