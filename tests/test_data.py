"""decidophobia.data + decidophobia.prompt 的测试.

跑:  PYTHONPATH=src .venv/bin/python tests/test_data.py
"""

import random
import sys

from decidophobia.data import MenuExample, build_examples, class_split, compose_menu
from decidophobia.prompt import render_menu

# --------------------------------------------------------------------------
# class_split
# --------------------------------------------------------------------------


def test_class_split_partitions_all_classes():
    """10 类留出 3: train 7 + held_out 3, 不重不漏."""
    s = class_split(n_classes=10, n_held_out=3, seed=0)
    assert len(s.train) == 7 and len(s.held_out) == 3, s
    assert sorted(s.train + s.held_out) == list(range(10)), s


def test_class_split_is_deterministic_by_seed():
    """同 seed 同切分; 不同 seed 应给不同的留出集."""
    a = class_split(10, 3, seed=1)
    b = class_split(10, 3, seed=1)
    c = class_split(10, 3, seed=2)
    assert a == b, (a, b)
    assert a.held_out != c.held_out, (a, c)


# --------------------------------------------------------------------------
# compose_menu
# --------------------------------------------------------------------------


def test_compose_menu_includes_gold_and_has_k_distinct_options_from_pool():
    """gold=5, pool=0..9, k=4 -> 4 个不同的类, 都在 pool 里, 含 5."""
    opts, gi = compose_menu(gold=5, pool=list(range(10)), k=4, rng=random.Random(0))
    assert len(opts) == 4 and len(set(opts)) == 4, opts
    assert set(opts) <= set(range(10)), opts
    assert opts[gi] == 5, (opts, gi)


def test_compose_menu_gold_position_varies_with_rng():
    """同一 gold, 多次采样, gold 应出现在不同位置 —— 模型不能靠位置记答案."""
    positions = set()
    rng = random.Random(0)
    for _ in range(40):
        _, gi = compose_menu(gold=0, pool=list(range(6)), k=3, rng=rng)
        positions.add(gi)
    assert positions == {0, 1, 2}, positions


def test_compose_menu_clamps_k_to_pool_size():
    """pool 只有 3 个类, 要 k=5 -> 给 3 个."""
    opts, _ = compose_menu(gold=1, pool=[0, 1, 2], k=5, rng=random.Random(0))
    assert sorted(opts) == [0, 1, 2], opts


# --------------------------------------------------------------------------
# build_examples
# --------------------------------------------------------------------------


def test_build_examples_only_uses_queries_and_menus_from_given_classes():
    """queries 的 label 落在 classes 之外的不出样本; 菜单里的选项也只来自 classes."""
    queries = ["q0", "q1", "q2", "q3"]
    labels = [0, 1, 2, 3]
    ex = build_examples(queries, labels, classes=[1, 3], k_range=(2, 2), rng=random.Random(0))
    assert [e.query for e in ex] == ["q1", "q3"], ex
    for e in ex:
        assert set(e.options) <= {1, 3}, e
        assert e.options[e.gold_idx] == e.label, e


def test_build_examples_respects_k_range():
    """k_range=(2, 4) 时每个菜单长度都在 [2, 4] 里."""
    queries = [f"q{i}" for i in range(30)]
    labels = [i % 6 for i in range(30)]
    ex = build_examples(queries, labels, classes=list(range(6)), k_range=(2, 4), rng=random.Random(0))
    ks = {len(e.options) for e in ex}
    assert ks <= {2, 3, 4} and len(ks) > 1, ks


# --------------------------------------------------------------------------
# render_menu
# --------------------------------------------------------------------------


def test_render_menu_lists_options_with_d_tokens_in_order_and_ends_at_answer():
    """菜单第 i 行以 <|Di|> 开头, 按 options 顺序; 提示以 'Answer:' 收尾, 答案 token 紧跟其后."""
    ex = MenuExample(query="I lost my card", options=[7, 2], gold_idx=1, label=2)
    names = {2: "change pin", 7: "card arrival"}
    s = render_menu(ex, names)
    assert "\n<|D0|>. card arrival\n<|D1|>. change pin\n" in s, repr(s)
    assert "I lost my card" in s, repr(s)
    assert s.endswith("Answer:"), repr(s[-30:])
    assert "<|D2|>" not in s, repr(s)


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{failed} failed")
    sys.exit(1 if failed else 0)
