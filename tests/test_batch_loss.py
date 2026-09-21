"""decidophobia.tokens / batch / loss / metrics 的测试.

跑:  PYTHONPATH=src .venv/bin/python tests/test_batch_loss.py
"""

import math
import sys

import torch

from decidophobia.batch import collate
from decidophobia.data import MenuExample
from decidophobia.loss import slot_cross_entropy
from decidophobia.metrics import brier_multiclass, ece_multiclass, nll_multiclass, topk_accuracy
from decidophobia.tokens import D_TOKENS, install_d_tokens

MODEL = "Qwen/Qwen3-0.6B-Base"


def _tok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL)


# --------------------------------------------------------------------------
# tokens
# --------------------------------------------------------------------------


def test_d_tokens_are_256_named_slots():
    assert len(D_TOKENS) == 256 and D_TOKENS[0] == "<|D0|>" and D_TOKENS[255] == "<|D255|>", D_TOKENS[:2]


def test_install_d_tokens_fills_spare_rows_without_resize():
    """Qwen3 tokenizer 151669 个, vocab_size 151936, 空 267 行. 256 个 D-token 应落在 151669..151924."""
    tok = _tok()
    ids = install_d_tokens(tok)
    assert ids == list(range(151669, 151669 + 256)), (ids[0], ids[-1])
    assert ids[-1] < 151936


def test_install_d_tokens_is_idempotent():
    """同一 tokenizer 装两次给同一批 id (add_tokens 第二次返回 0, 不能因此返回空表)."""
    tok = _tok()
    a = install_d_tokens(tok)
    b = install_d_tokens(tok)
    assert a == b and len(b) == 256


def test_d_token_is_single_token_right_after_colon():
    """'Answer:<|D3|>' 必须切成 [..., ':', <|D3|>], 中间没有空格 token —— 读出位置是冒号之后紧邻的那一位."""
    tok = _tok()
    ids = install_d_tokens(tok)
    enc = tok.encode("Answer:<|D3|>", add_special_tokens=False)
    assert enc[-1] == ids[3], tok.convert_ids_to_tokens(enc)
    assert tok.convert_ids_to_tokens(enc[-2]) == ":", tok.convert_ids_to_tokens(enc)


# --------------------------------------------------------------------------
# collate
# --------------------------------------------------------------------------


def test_collate_left_pads_so_answer_position_is_last():
    """两条长短不一的提示, 左填充后每行最后一个 token 都是 ':'; 短的那条前面是 pad."""
    tok = _tok()
    d_ids = install_d_tokens(tok)
    exs = [
        MenuExample(query="short", options=[0, 1], gold_idx=0, label=0),
        MenuExample(query="a much longer customer message here", options=[2, 0, 1], gold_idx=2, label=1),
    ]
    names = {0: "a", 1: "b", 2: "c"}
    b = collate(exs, tok, names, d_ids, k_max=4)
    colon = tok.encode(":", add_special_tokens=False)[0]
    assert b["input_ids"].shape[0] == 2
    assert (b["input_ids"][:, -1] == colon).all(), b["input_ids"][:, -1]
    assert b["attention_mask"][0, 0] == 0 and b["attention_mask"][1, 0] == 1, b["attention_mask"][:, 0]


def test_collate_builds_slot_ids_by_position_and_pads_with_minus_one():
    """slot_ids[i, j] = D_j 的 id (j < k_i), 其余 -1; gold 是位置索引."""
    tok = _tok()
    d_ids = install_d_tokens(tok)
    exs = [
        MenuExample(query="x", options=[5, 9], gold_idx=1, label=9),
        MenuExample(query="y", options=[1, 2, 3], gold_idx=0, label=1),
    ]
    names = {i: f"n{i}" for i in range(10)}
    b = collate(exs, tok, names, d_ids, k_max=4)
    assert b["slot_ids"].tolist() == [
        [d_ids[0], d_ids[1], -1, -1],
        [d_ids[0], d_ids[1], d_ids[2], -1],
    ], b["slot_ids"]
    assert b["gold"].tolist() == [1, 0]


# --------------------------------------------------------------------------
# loss
# --------------------------------------------------------------------------


def test_slot_cross_entropy_masks_padding_and_averages():
    """V=10.
    行 0: 槽 [5,6,7] 的 logit [1,2,3], gold=2 -> -ln(e^3/(e+e^2+e^3)) = 0.40760596
    行 1: 槽 [8,9,-1], logit [0,0,(第 3 位是 pad, 其 id 位置 logit 设 99 也不得参与)], gold=0 -> ln 2 = 0.69314718
    mean = 0.55037657
    """
    logits = torch.zeros(2, 10)
    logits[0, 5], logits[0, 6], logits[0, 7] = 1.0, 2.0, 3.0
    logits[1, 8], logits[1, 9] = 0.0, 0.0
    logits[1, 0] = 99.0  # pad 槽 clamp 到 id 0 之后若没 mask 会读到它
    slot_ids = torch.tensor([[5, 6, 7], [8, 9, -1]])
    gold = torch.tensor([2, 0])
    got = slot_cross_entropy(logits, slot_ids, gold).item()
    assert abs(got - 0.55037657) < 1e-6, got


# --------------------------------------------------------------------------
# metrics (与 baseline-banking77.py 同一套手算期望)
# --------------------------------------------------------------------------


def test_topk_accuracy():
    q = [[0.1, 0.2, 0.7], [0.3, 0.6, 0.1], [0.5, 0.1, 0.4]]
    y = [2, 0, 1]
    assert abs(topk_accuracy(q, y, k=1) - 1 / 3) < 1e-12
    assert abs(topk_accuracy(q, y, k=2) - 2 / 3) < 1e-12


def test_nll_multiclass_clamps():
    assert abs(nll_multiclass([[0.9, 0.1], [0.2, 0.8]], [0, 1]) - 0.1642520330) < 1e-9
    assert abs(nll_multiclass([[1.0, 0.0]], [1]) - 27.631021115) < 1e-6


def test_brier_multiclass():
    assert abs(brier_multiclass([[0.9, 0.1], [0.2, 0.8]], [0, 0]) - 0.65) < 1e-12


def test_ece_multiclass():
    q = [[0.6, 0.4], [0.3, 0.7], [0.8, 0.2], [0.1, 0.9]]
    assert abs(ece_multiclass(q, [0, 0, 0, 1], n_bins=4) - 0.15) < 1e-12


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
