"""学習したしゃがむポリシーを NumPy の .npz に書き出す（lab サービスの中で実行）
  python export.py <model_XXXX.pt> [出力先]"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g1_squat import spec  # noqa: E402

ckpt = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else "/project/lab/exported/squat_policy.npz"
sd = torch.load(ckpt, map_location="cpu", weights_only=False)["model_state_dict"]
arrays = {k.replace(".", "_"): v.numpy().astype(np.float32) for k, v in sd.items() if k.startswith("actor.")}
assert arrays["actor_0_weight"].shape[1] == spec.OBS_DIM, "観測の次元が spec.py と合いません"
meta = dict(checkpoint=os.path.relpath(ckpt, "/project"), policy_joints=spec.POLICY_JOINTS, obs_joints=spec.OBS_JOINTS,
            default=spec.DEFAULT, action_scale=[spec.ACTION_SCALES[n] for n in spec.POLICY_JOINTS], ang_vel_scale=spec.ANG_VEL_SCALE,
            joint_vel_scale=spec.JOINT_VEL_SCALE, stand_height=spec.STAND_HEIGHT, height_range=spec.HEIGHT_RANGE)
os.makedirs(os.path.dirname(out), exist_ok=True)
np.savez(out, meta=json.dumps(meta), **arrays)
print(f"書き出しました: {out}（{ckpt}）")
