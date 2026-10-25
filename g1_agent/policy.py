"""unitree_rl_gym の G1 歩行ポリシー（motion.pt）を PyTorch なしで動かす
TorchScript の zip から重みだけを読み、LSTM(47->64) + MLP(64->32->12) を NumPy で計算する"""
from __future__ import annotations

import pickle
import zipfile

import numpy as np


class _Storage:
    def __init__(self, key: str):
        self.key = key


class _TorchScriptModule(dict):
    def __setstate__(self, state):
        self.update(state)


class _Unpickler(pickle.Unpickler):
    def __init__(self, fp, zf: zipfile.ZipFile, prefix: str):
        super().__init__(fp)
        self.zf = zf
        self.prefix = prefix

    def find_class(self, module, name):
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return self._rebuild_tensor
        if module == "torch" and name.endswith("Storage"):
            return name
        if module == "collections" and name == "OrderedDict":
            return dict
        if module.startswith("__torch__"):
            return _TorchScriptModule
        if module == "torch.jit._pickle" and name == "restore_type_tag":
            return lambda value, _type: value
        if module == "torch.jit._pickle" and name.startswith("build_"):
            return lambda value: value
        raise pickle.UnpicklingError(f"unexpected global {module}.{name}")

    def persistent_load(self, pid):
        return _Storage(pid[2])

    def _rebuild_tensor(self, storage, offset, size, stride, *_):
        raw = self.zf.read(f"{self.prefix}/data/{storage.key}")
        flat = np.frombuffer(raw, dtype="<f4")
        if len(size) == 0:
            return flat[offset].copy()
        strides = tuple(s * 4 for s in stride)
        return np.lib.stride_tricks.as_strided(flat[offset:], shape=size, strides=strides).copy()


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _elu(x):
    return np.where(x > 0, x, np.expm1(np.minimum(x, 0)))


class LSTMPolicy:
    """legged_gym の PolicyExporterLSTM と同じ計算"""

    def __init__(self, path: str):
        with zipfile.ZipFile(path) as zf:
            prefix = zf.namelist()[0].split("/")[0]
            with zf.open(f"{prefix}/data.pkl") as fp:
                root = _Unpickler(fp, zf, prefix).load()
        mem = root["memory"]
        self.w_ih = mem["weight_ih_l0"]
        self.w_hh = mem["weight_hh_l0"]
        self.b = mem["bias_ih_l0"] + mem["bias_hh_l0"]
        actor = root["actor"]
        self.l0_w, self.l0_b = actor["0"]["weight"], actor["0"]["bias"]
        self.l2_w, self.l2_b = actor["2"]["weight"], actor["2"]["bias"]
        self.hidden = self.w_hh.shape[1]
        self.reset()

    def reset(self):
        self.h = np.zeros(self.hidden, dtype=np.float32)
        self.c = np.zeros(self.hidden, dtype=np.float32)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        g = self.w_ih @ obs + self.w_hh @ self.h + self.b
        i, f, gg, o = np.split(g, 4)  # PyTorch のゲート順: input, forget, cell, output
        self.c = _sigmoid(f) * self.c + _sigmoid(i) * np.tanh(gg)
        self.h = _sigmoid(o) * np.tanh(self.c)
        x = _elu(self.l0_w @ self.h + self.l0_b)
        return (self.l2_w @ x + self.l2_b).astype(np.float32)


class TorchPolicy:
    def __init__(self, path: str):
        import torch

        self.torch = torch
        self.policy = torch.jit.load(path)

    def reset(self):
        self.policy.reset_memory()

    def __call__(self, obs):
        with self.torch.no_grad():
            return self.policy(self.torch.from_numpy(obs).unsqueeze(0)).numpy().squeeze()


def load_policy(path: str, use_torch: bool = False):
    return TorchPolicy(path) if use_torch else LSTMPolicy(path)
