"""把一条菜单样本渲染成提示文本. 模板是被训练的对象, 改动等于换任务."""

from __future__ import annotations

from decidophobia.data import MenuExample
from decidophobia.tokens import D_TOKENS

HEADER = "Classify the customer message into exactly one of the options below.\n\nOptions:\n"


def render_menu(ex: MenuExample, names: dict[int, str]) -> str:
    """第 i 行 '<|Di|>. <名字>'; 以 'Answer:' 收尾, 答案 token 紧跟冒号之后, 中间无空格."""
    lines = [f"{D_TOKENS[i]}. {names[c]}" for i, c in enumerate(ex.options)]
    return HEADER + "\n".join(lines) + f"\n\nCustomer message: {ex.query}\nAnswer:"
