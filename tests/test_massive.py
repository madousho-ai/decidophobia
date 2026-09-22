"""decidophobia.massive 的测试: 只做评估的留出数据集, 训练里一条都不出现.

跑:  PYTHONPATH=src .venv/bin/python tests/test_massive.py
需要 data/massive/amazon-massive-dataset-1.0.tar.gz 在盘上 (39.5MB, 不自动下载).
"""

import random

from _runner import run
from decidophobia.massive import humanize_intent, load_massive


def test_humanize_intent_reads_as_scenario_then_action():
    """名字形如 'scenario: action'. 粘连的复合词要拆开, 品牌名 (hue / wemo) 保留."""
    assert humanize_intent("alarm_set") == "alarm: set"
    assert humanize_intent("iot_hue_lightchange") == "iot: hue light change"
    assert humanize_intent("lists_createoradd") == "lists: create or add"
    assert humanize_intent("email_querycontact") == "email: query contact"
    assert humanize_intent("iot_wemo_off") == "iot: wemo off"
    assert humanize_intent("qa_maths") == "qa: maths"


def test_test_split_has_2974_utterances_over_60_intents():
    te = load_massive()
    assert len(te.queries) == 2974
    assert len(te.names) == 60
    assert set(te.labels) == set(range(60)) - {10}, "cooking_query (id 10) 在 test 里 0 条、train 里 4 条; 其余 59 类都有"
    assert te.context_label != "Customer message", "上下文标签要换成这个数据集自己的, 别把 Banking77 的框架带过来"
    assert te.qtype == "choice"


def test_intent_ids_follow_sorted_raw_names():
    """类 id 按原始 intent 名字母序, 与 banking77 同一约定; 名字表是人话化之后的."""
    te = load_massive()
    assert te.names[0] == "alarm: query"
    assert te.names[59] == "weather: query"


def test_full_menu_lists_every_intent_exactly_once():
    te = load_massive()
    exs = te.build_examples(list(range(60)), (60, 60), random.Random(0))
    assert len(exs) == 2974
    assert all(sorted(ex.options) == list(range(60)) for ex in exs[:50])
    assert all(ex.options[ex.gold_idx] == ex.label for ex in exs[:50])


if __name__ == "__main__":
    run(globals())
