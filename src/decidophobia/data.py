"""菜单样本的采样: 类切分、组菜单、批量生成. 纯函数, 不碰文本与 tokenizer.

一条样本 = 上下文 (query) + 可选的问句 + 一个选项菜单 (类 id 的有序列表) + 正确选项在菜单里的位置.
菜单每次都重新随机组, 同一个类在不同样本里落到不同位置 —— 模型学的是
「在给出的选项里指出匹配的那个」, 而非「某个类永远对应某个槽」.

LabeledSet 是数据集适配层交上来的统一形状: 一列上下文、一列类 id、类名表,
以及这个数据集里上下文叫什么 (Customer message / Passage) 和每条各自的问句 (可无).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


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
    query: str  # 上下文: 用户句 / passage
    options: list[int]  # 类 id, 顺序就是菜单顺序, 第 i 个绑 <|Di|>
    gold_idx: int  # 正确选项在 options 里的位置
    label: int  # 正确选项的类 id (== options[gold_idx])
    option_names: list[str]  # 与 options 平行, 菜单里显示的名字
    context_label: str = "Customer message"  # 上下文前面的标签
    question: str | None = None  # 这条样本的问句; None 表示数据集用固定的默认问句
    qtype: str = "choice"  # choice | bool | score, 见 tokens.QTYPES


def compose_menu(gold: int, pool: list[int], k: int, rng: random.Random) -> tuple[list[int], int]:
    """从 pool 里抽 k-1 个干扰项加上 gold, 打乱. 返回 (options, gold 的位置).
    k 大于 pool 大小时取整个 pool."""
    others = [c for c in pool if c != gold]
    k = min(k, len(others) + 1)
    opts = rng.sample(others, k - 1) + [gold]
    rng.shuffle(opts)
    return opts, opts.index(gold)


@dataclass(frozen=True)
class LabeledSet:
    queries: list[str]
    labels: list[int]  # 索引进 names
    names: dict[int, str]  # 类 id -> 给模型看的名字
    context_label: str = "Customer message"
    questions: list[str] | None = None  # 与 queries 平行; None 表示没有逐条问句
    question_default: str | None = field(default=None)  # 没有逐条问句时统一用它
    qtype: str = "choice"

    def make_example(self, i: int, options: list[int], gold_idx: int) -> MenuExample:
        q = self.questions[i] if self.questions is not None else self.question_default
        return MenuExample(
            query=self.queries[i], options=options, gold_idx=gold_idx, label=self.labels[i],
            option_names=[self.names[c] for c in options], context_label=self.context_label, question=q,
            qtype=self.qtype,
        )

    def build_examples(self, classes: list[int], k_range: tuple[int, int], rng: random.Random) -> list[MenuExample]:
        """给 label 落在 classes 里的每条各组一个菜单, 菜单选项只从 classes 里取. 用于固定的评估集."""
        allowed = set(classes)
        out = []
        for i, lab in enumerate(self.labels):
            if lab not in allowed:
                continue
            k = rng.randint(k_range[0], k_range[1])
            opts, gi = compose_menu(lab, classes, k, rng)
            out.append(self.make_example(i, opts, gi))
        return out

    def sample_examples(self, classes: list[int], k_range: tuple[int, int], n: int, rng: random.Random) -> list[MenuExample]:
        """随机抽 n 条 label 落在 classes 内的, 各配一个现组的菜单. 用于训练批."""
        allowed = set(classes)
        pool = [i for i, lab in enumerate(self.labels) if lab in allowed]
        out = []
        for i in rng.sample(pool, n):
            k = rng.randint(k_range[0], k_range[1])
            opts, gi = compose_menu(self.labels[i], classes, k, rng)
            out.append(self.make_example(i, opts, gi))
        return out
