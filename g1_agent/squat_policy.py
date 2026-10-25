"""Isaac Lab で学習したしゃがむポリシー（lab/）を NumPy で動かす"""
from __future__ import annotations

import json
import os

import numpy as np

DEFAULT_PATH = os.environ.get(
    "G1_SQUAT_POLICY",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lab/exported/squat_policy.npz"))


def available(path: str = DEFAULT_PATH) -> bool:
    return os.path.exists(path)


class SquatPolicy:
    def __init__(self, path: str = DEFAULT_PATH):
        z = np.load(path)
        self.meta = json.loads(str(z["meta"]))
        self.layers = []
        i = 0
        while f"actor_{i}_weight" in z:
            self.layers.append((z[f"actor_{i}_weight"], z[f"actor_{i}_bias"]))
            i += 2  # actor.0, actor.2, ... の間は ELU
        self.last_action = np.zeros(len(self.meta["policy_joints"]), dtype=np.float32)

    def reset(self):
        self.last_action[:] = 0.0

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = obs.astype(np.float32)
        for k, (w, b) in enumerate(self.layers):
            x = w @ x + b
            if k < len(self.layers) - 1:
                x = np.where(x > 0, x, np.exp(np.minimum(x, 0)) - 1)
        self.last_action = x
        return x
