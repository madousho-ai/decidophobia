"""decidophobia.model 的测试. 要加载 0.6B, 走 GPU.

跑:  PYTHONPATH=src .venv/bin/python tests/test_model.py
"""

import sys

import torch

from decidophobia.batch import collate
from decidophobia.data import MenuExample
from decidophobia.loss import slot_cross_entropy
from decidophobia.model import last_logits, prepare_model, trainable_param_groups
from decidophobia.tokens import install_d_tokens

MODEL = "Qwen/Qwen3-0.6B-Base"


def _load():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    d_ids = install_d_tokens(tok)
    lm = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda")
    return tok, d_ids, lm


def test_prepare_model_trains_only_lora_and_embedding():
    """可训参数名里只应有 lora_ 和 embed_tokens; 其余全部冻结."""
    tok, d_ids, lm = _load()
    m = prepare_model(lm, d_ids, lora_r=4, lora_alpha=8, lora_dropout=0.0)
    names = [n for n, p in m.named_parameters() if p.requires_grad]
    assert names, "nothing trainable"
    bad = [n for n in names if "lora_" not in n and "embed_tokens" not in n]
    assert not bad, bad[:5]
    assert any("embed_tokens" in n for n in names), names[:5]


def test_embedding_gradient_is_zero_outside_d_rows():
    """一步反传后, 嵌入矩阵的梯度只在 D 行非零 —— 基模的词嵌入不许动."""
    tok, d_ids, lm = _load()
    m = prepare_model(lm, d_ids, lora_r=4, lora_alpha=8, lora_dropout=0.0)
    exs = [MenuExample(query="I lost my card", options=[0, 1], gold_idx=0, label=0)]
    b = collate(exs, tok, {0: "card lost", 1: "change pin"}, d_ids, k_max=2)
    b = {k: v.to("cuda") for k, v in b.items()}
    logits = last_logits(m, b["input_ids"], b["attention_mask"])
    slot_cross_entropy(logits, b["slot_ids"], b["gold"]).backward()
    emb = m.get_input_embeddings().weight
    g = emb.grad
    assert g is not None
    d = torch.zeros(g.shape[0], dtype=torch.bool, device=g.device)
    d[d_ids] = True
    assert g[~d].abs().max().item() == 0.0, g[~d].abs().max().item()
    assert g[d].abs().max().item() > 0.0


def test_trainable_param_groups_split_lora_from_embedding():
    """两组: lora 一组, 嵌入一组, 学习率各自给."""
    tok, d_ids, lm = _load()
    m = prepare_model(lm, d_ids, lora_r=4, lora_alpha=8, lora_dropout=0.0)
    groups = trainable_param_groups(m, lr_lora=1e-4, lr_embed=1e-3)
    assert len(groups) == 2, len(groups)
    lrs = sorted(g["lr"] for g in groups)
    assert lrs == [1e-4, 1e-3], lrs
    assert all(len(g["params"]) > 0 for g in groups)


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
        torch.cuda.empty_cache()
    print(f"\n{failed} failed")
    sys.exit(1 if failed else 0)
