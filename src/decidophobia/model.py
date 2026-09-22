"""可训练的部分: attention 上的 LoRA + 嵌入矩阵里 256 个 D 行. 其余冻结.

改的是路由 (哪一行匹配) 和槽标记 (D-token 的向量), 知识那部分不碰.

D 行的放开用梯度 hook 实现: 整个嵌入矩阵 requires_grad, 反传后把非 D 行的梯度
清零. peft 有 trainable_token_indices 做同一件事, 但它在 tied embedding 上的
处理路径这几个版本变过好几次, 一个 8 行的 hook 比追它稳.
"""

from __future__ import annotations

import torch
from peft import LoraConfig, get_peft_model

ATTN_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP_TARGETS = ["gate_proj", "up_proj", "down_proj"]

# 放开的范围, 三档. 全参不在其中: 0.6B 全参 AdamW 的优化器状态 8GB 卡放不下,
# 更要紧的是它让模型有能力记住数据集的事实, 留出类的成绩就不再说明泛化.
LORA_TARGETS: dict[str, list[str]] = {
    "d-only": [],  # 基模全冻, 只训 256 个 D 行 —— 纯读出
    "attn": ATTN_TARGETS,
    "attn-mlp": ATTN_TARGETS + MLP_TARGETS,
}


def prepare_model(
    lm, d_ids: list[int], lora_r: int, lora_alpha: int, lora_dropout: float, trainable: str = "attn"
):
    targets = LORA_TARGETS[trainable]
    if targets:
        cfg = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=targets, bias="none", task_type="CAUSAL_LM",
        )
        m = get_peft_model(lm, cfg)
    else:
        m = lm
        for p in m.parameters():
            p.requires_grad_(False)
    emb = m.get_input_embeddings().weight
    emb.requires_grad_(True)
    keep = torch.zeros(emb.shape[0], dtype=torch.bool, device=emb.device)
    keep[d_ids] = True
    emb.register_hook(lambda g: g * keep[:, None].to(g.dtype))
    # 空行的初值是随机的 (基模从没训过它们); 从已有嵌入的均值起步, 让第一步就在合理的尺度上
    with torch.no_grad():
        mu = emb[: min(d_ids)].mean(0)
        emb[d_ids] = mu + 0.01 * torch.randn(len(d_ids), emb.shape[1], device=emb.device, dtype=emb.dtype)
    return m


def trainable_param_groups(m, lr_lora: float, lr_embed: float) -> list[dict]:
    lora, embed = [], []
    for n, p in m.named_parameters():
        if not p.requires_grad:
            continue
        (embed if "embed_tokens" in n else lora).append(p)
    return [{"params": lora, "lr": lr_lora}, {"params": embed, "lr": lr_embed}]


def last_logits(m, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """只算最后一个位置的 logits, (B, V). logits_to_keep=1 绕开 (B, L, V) 的大张量."""
    out = m(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=1)
    return out.logits[:, -1, :].float()
