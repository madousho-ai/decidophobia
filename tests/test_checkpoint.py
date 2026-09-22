"""train.save_trained / load_trained 的测试. 用一个假模型, 不碰 GPU.

跑:  PYTHONPATH=src .venv/bin/python tests/test_checkpoint.py
"""

import tempfile

import torch
from torch import nn

from _runner import run
from decidophobia.train import TrainConfig, load_trained, save_trained


class _Emb(nn.Module):
    def __init__(self, n, h):
        super().__init__()
        self.rows = nn.Parameter(torch.zeros(n, h))


class _Fake(nn.Module):
    """只提供 load_trained 会碰的两样: named_parameters 里的 lora_ 张量, 和 get_input_embeddings().rows."""

    def __init__(self, n_rows, h=4):
        super().__init__()
        self.emb = _Emb(n_rows, h)
        self.lora_A = nn.Parameter(torch.zeros(2, h))

    def get_input_embeddings(self):
        return self.emb


def _save(m, ids):
    f = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    save_trained(m, ids, TrainConfig(), f.name)
    return f.name


def test_load_trained_accepts_checkpoint_with_a_prefix_of_the_model_rows():
    """类型 token 加进来之前存的档只有 256 个 D 行. 那 3 行当时不在提示里、梯度为零,
    留在初始化就是那次训练的真实状态. 所以 ids 是前缀就该装得进去, 多出的行原样不动."""
    old = _Fake(256)
    with torch.no_grad():
        old.emb.rows.fill_(1.0)
        old.lora_A.fill_(2.0)
    path = _save(old, list(range(256)))

    new = _Fake(259)
    with torch.no_grad():
        new.emb.rows.fill_(-1.0)
    load_trained(new, list(range(259)), path)
    assert torch.all(new.emb.rows[:256] == 1.0)
    assert torch.all(new.emb.rows[256:] == -1.0), "档里没有的行必须保持原样"
    assert torch.all(new.lora_A == 2.0)


def test_load_trained_rejects_ids_that_are_not_a_prefix():
    old = _Fake(256)
    path = _save(old, list(range(1, 257)))  # 同样 256 个, 但内容对不上
    new = _Fake(259)
    try:
        load_trained(new, list(range(259)), path)
    except ValueError:
        return
    raise AssertionError("ids 不是前缀却没有报错")


def test_load_trained_rejects_checkpoint_with_more_rows_than_the_model():
    old = _Fake(259)
    path = _save(old, list(range(259)))
    new = _Fake(256)
    try:
        load_trained(new, list(range(256)), path)
    except ValueError:
        return
    raise AssertionError("档比模型多行却没有报错")


if __name__ == "__main__":
    run(globals())
