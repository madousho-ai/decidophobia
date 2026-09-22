"""decidophobia.data + decidophobia.prompt 的测试.

跑:  PYTHONPATH=src .venv/bin/python tests/test_data.py
"""

import random

from _runner import run
from decidophobia.data import LabeledSet, MenuExample, class_split, compose_menu
from decidophobia.prompt import render_menu, split_prompt

NAMES = {i: f"n{i}" for i in range(10)}


def _ex(query="Q", options=(0, 1), gold_idx=0, **kw):
    return MenuExample(query=query, options=list(options), gold_idx=gold_idx, label=options[gold_idx],
                       option_names=[NAMES[c] for c in options], **kw)


# --------------------------------------------------------------------------
# class_split / compose_menu
# --------------------------------------------------------------------------


def test_class_split_partitions_all_classes():
    s = class_split(n_classes=10, n_held_out=3, seed=0)
    assert len(s.train) == 7 and len(s.held_out) == 3, s
    assert sorted(s.train + s.held_out) == list(range(10)), s


def test_class_split_is_deterministic_by_seed():
    a, b, c = class_split(10, 3, seed=1), class_split(10, 3, seed=1), class_split(10, 3, seed=2)
    assert a == b and a.held_out != c.held_out


def test_compose_menu_includes_gold_and_has_k_distinct_options_from_pool():
    opts, gi = compose_menu(gold=5, pool=list(range(10)), k=4, rng=random.Random(0))
    assert len(opts) == 4 and len(set(opts)) == 4 and set(opts) <= set(range(10)) and opts[gi] == 5


def test_compose_menu_gold_position_varies_with_rng():
    rng = random.Random(0)
    positions = {compose_menu(0, list(range(6)), 3, rng)[1] for _ in range(40)}
    assert positions == {0, 1, 2}, positions


def test_compose_menu_clamps_k_to_pool_size():
    """pool 只有 2 个类 (BoolQ 的 yes/no) 时 k 夹到 2, 两个位置都要出现."""
    rng = random.Random(0)
    outs = [compose_menu(1, [0, 1], 10, rng) for _ in range(20)]
    assert all(sorted(o) == [0, 1] for o, _ in outs)
    assert {gi for _, gi in outs} == {0, 1}


# --------------------------------------------------------------------------
# LabeledSet
# --------------------------------------------------------------------------


def _set(n=30, n_cls=6, questions=None):
    return LabeledSet(queries=[f"q{i}" for i in range(n)], labels=[i % n_cls for i in range(n)],
                      names={i: f"n{i}" for i in range(n_cls)}, context_label="Customer message",
                      questions=questions)


def test_build_examples_only_uses_queries_and_menus_from_given_classes():
    s = _set(n=4, n_cls=4)
    ex = s.build_examples(classes=[1, 3], k_range=(2, 2), rng=random.Random(0))
    assert [e.query for e in ex] == ["q1", "q3"], ex
    for e in ex:
        assert set(e.options) <= {1, 3} and e.options[e.gold_idx] == e.label
        assert e.option_names == [s.names[c] for c in e.options]
        assert e.context_label == "Customer message" and e.question is None


def test_sample_examples_draws_n_with_varied_k():
    s = _set()
    ex = s.sample_examples(classes=list(range(6)), k_range=(2, 4), n=20, rng=random.Random(0))
    assert len(ex) == 20 and {len(e.options) for e in ex} <= {2, 3, 4} and len({len(e.options) for e in ex}) > 1


def test_labeled_set_carries_per_item_question():
    """BoolQ 每条各有问句; 通过 questions 列表按索引带进样本."""
    s = _set(n=3, n_cls=2, questions=["is it a?", "is it b?", "is it c?"])
    ex = s.build_examples(classes=[0, 1], k_range=(2, 2), rng=random.Random(0))
    assert [e.question for e in ex] == ["is it a?", "is it b?", "is it c?"]


# --------------------------------------------------------------------------
# prompt
# --------------------------------------------------------------------------


def test_render_menu_lists_options_with_d_tokens_in_order_and_ends_at_answer():
    s = render_menu(_ex("I lost my card", options=(7, 2), gold_idx=1))
    assert "\n<|D0|>. n7\n<|D1|>. n2\n" in s, repr(s)
    assert s.endswith("Answer:") and "<|D2|>" not in s, repr(s)


def test_context_first_puts_query_before_menu_and_splits_at_the_newline():
    ex = _ex("I lost my card", options=(7, 2), gold_idx=1)
    s = render_menu(ex, layout="context-first")
    assert s.index("I lost my card") < s.index("<|D0|>"), repr(s)
    ctx, q = split_prompt(ex, layout="context-first")
    assert ctx + q == s and ctx.endswith("\n\n") and ctx.startswith("Customer message: I lost my card")
    assert "<|D0|>" not in ctx and "I lost my card" not in q


def test_context_prefix_is_identical_across_questions():
    a = _ex("q", options=(0, 1), gold_idx=0)
    b = _ex("q", options=(3, 4, 5), gold_idx=2)
    assert split_prompt(a, layout="context-first")[0] == split_prompt(b, layout="context-first")[0]


def test_menu_first_has_no_shared_prefix():
    ctx, q = split_prompt(_ex("hello"), layout="menu-first")
    assert ctx == "" and "hello" in q and q.endswith("Answer:")


def test_question_goes_between_context_and_menu():
    """BoolQ 形状: passage 是 context, 每条自带问句, 选项 yes/no.
    context-first 下 passage 在前缀里, 问句在分支里、菜单之前."""
    ex = _ex("The sky is blue because of Rayleigh scattering.", options=(1, 0), gold_idx=0,
             context_label="Passage", question="is the sky blue because of scattering?")
    ex = MenuExample(**{**ex.__dict__, "option_names": ["yes", "no"]})
    ctx, q = split_prompt(ex, layout="context-first")
    assert ctx.startswith("Passage: The sky is blue") and "scattering?" not in ctx, repr(ctx)
    assert q.index("Question: is the sky blue") < q.index("<|D0|>. yes"), repr(q)
    assert "\n<|D0|>. yes\n<|D1|>. no\n" in q and q.endswith("Answer:"), repr(q)


if __name__ == "__main__":
    run(globals())
