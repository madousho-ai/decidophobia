"""把一条菜单样本渲染成提示文本. 模板是被训练的对象, 改动等于换任务.

两种布局:
  context-first  用户句 + 指令 + 菜单 + Answer:   —— 前缀只含用户句, N 个问题共用它的 KV cache,
                                                   每个问题只喂自己那段菜单. 默认.
  menu-first     指令 + 菜单 + 用户句 + Answer:   —— 答案位置读用户句时已经知道选项. 对照组.

默认定为 context-first 的依据 (attn LoRA + cosine, 2000 步, 17 个留出类, 10 选 1):
unseen acc 0.903 vs 0.888, NLL 0.310 vs 0.341, 9 个评估点全部领先, 后半程 sd 0.0012 vs 0.0083,
step 250 即到 0.88 (menu-first 要 1000 步). 与冻结探针 h_bare 0.880 > h_menu 0.857 同向.
唯一输的一项是 ECE 0.041 vs 0.033 (过自信 +3.8 vs +2.9 点), 标定项加进来之后再看.

split_prompt 给出 context-first 的两段, 分界处两边都以换行收尾, BPE 不会跨界合并.
"""

from __future__ import annotations

from decidophobia.data import MenuExample
from decidophobia.tokens import D_TOKENS

LAYOUTS = ("context-first", "menu-first")
DEFAULT_LAYOUT = "context-first"

HEADER = "Classify the customer message into exactly one of the options below.\n\nOptions:\n"
HEADER_CTX = "Classify the customer message above into exactly one of the options below.\n\nOptions:\n"


def _menu(ex: MenuExample, names: dict[int, str]) -> str:
    return "\n".join(f"{D_TOKENS[i]}. {names[c]}" for i, c in enumerate(ex.options))


def split_prompt(ex: MenuExample, names: dict[int, str], layout: str = DEFAULT_LAYOUT) -> tuple[str, str]:
    """(context, question). menu-first 下 context 为空串 —— 那种布局没有可共享的前缀."""
    if layout == "menu-first":
        return "", HEADER + _menu(ex, names) + f"\n\nCustomer message: {ex.query}\nAnswer:"
    if layout == "context-first":
        return f"Customer message: {ex.query}\n\n", HEADER_CTX + _menu(ex, names) + "\n\nAnswer:"
    raise ValueError(f"unknown layout {layout!r}; expected one of {LAYOUTS}")


def render_menu(ex: MenuExample, names: dict[int, str], layout: str = DEFAULT_LAYOUT) -> str:
    """整条提示. 第 i 行 '<|Di|>. <名字>'; 以 'Answer:' 收尾, 答案 token 紧跟冒号之后, 中间无空格."""
    ctx, q = split_prompt(ex, names, layout)
    return ctx + q
