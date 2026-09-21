"""256 个决策槽 token. 它们既是菜单里的行首标记, 也是答案位置上要预测的 token.

Qwen3 的 tokenizer 有 151669 个 token 而 vocab_size 是 151936, 空 267 行,
256 个 D-token 装进去不用 resize 嵌入, checkpoint 形状不变.
tie_word_embeddings=True 时嵌入行和输出头行是同一个向量: 输出位置上 D_k 的
分数 = h · E[D_k], 与 nn.Linear(hidden, 256) 的第 k 行等价, 区别只是这个向量
还会在菜单里作为输入出现.
"""

from __future__ import annotations

N_SLOTS = 256
D_TOKENS = [f"<|D{i}|>" for i in range(N_SLOTS)]


def install_d_tokens(tokenizer) -> list[int]:
    """把 256 个 D-token 加进 tokenizer (已有则跳过), 返回它们的 id, 顺序与 D_TOKENS 一致."""
    tokenizer.add_tokens(D_TOKENS, special_tokens=True)
    return tokenizer.convert_tokens_to_ids(D_TOKENS)
