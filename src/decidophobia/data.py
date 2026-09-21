"""菜单样本的采样: 类切分、组菜单、批量生成. 纯函数, 不碰文本与 tokenizer.

一条样本 = 一个用户句 + 一个选项菜单 (类 id 的有序列表) + 正确选项在菜单里的位置.
菜单每次都重新随机组, 同一个类在不同样本里落到不同位置 —— 模型学的是
「在给出的选项里指出匹配的那个」, 而非「某个类永远对应某个槽」.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class ClassSplit:
    train: list[int]
    held_out: list[int]


def class_split(n_classes: int, n_held_out: int, seed: int) -> ClassSplit:
    """把 0..n-1 随机切成 train / held_out. held_out 的类在训练里完全不出现,
    用来测「学到的是方法还是记住了标签」."""
    ids = list(range(n_classes))
    random.Random(seed).shuffle(ids)
    return ClassSplit(train=sorted(ids[n_held_out:]), held_out=sorted(ids[:n_held_out]))


@dataclass(frozen=True)
class MenuExample:
    query: str
    options: list[int]  # 类 id, 顺序就是菜单顺序, 第 i 个绑 <|Di|>
    gold_idx: int  # 正确选项在 options 里的位置
    label: int  # 正确选项的类 id (== options[gold_idx])


def compose_menu(gold: int, pool: list[int], k: int, rng: random.Random) -> tuple[list[int], int]:
    """从 pool 里抽 k-1 个干扰项加上 gold, 打乱. 返回 (options, gold 的位置).
    k 大于 pool 大小时取整个 pool."""
    others = [c for c in pool if c != gold]
    k = min(k, len(others) + 1)
    opts = rng.sample(others, k - 1) + [gold]
    rng.shuffle(opts)
    return opts, opts.index(gold)


def build_examples(
    queries: list[str],
    labels: list[int],
    classes: list[int],
    k_range: tuple[int, int],
    rng: random.Random,
) -> list[MenuExample]:
    """给 label 落在 classes 里的每条 query 组一个菜单, 菜单选项只从 classes 里取,
    菜单长度在 k_range 内均匀抽."""
    allowed = set(classes)
    out = []
    for q, lab in zip(queries, labels):
        if lab not in allowed:
            continue
        k = rng.randint(k_range[0], k_range[1])
        opts, gi = compose_menu(lab, classes, k, rng)
        out.append(MenuExample(query=q, options=opts, gold_idx=gi, label=lab))
    return out
