"""歩行ポリシーからしゃがむポリシーへ切り替える瞬間の状態を MuJoCo で集める（学習の初期状態に使う）
  python scripts/collect_handoff_states.py [出力先] [本数]"""
import math
import os
import sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _load_spec():
    """spec.py だけを読む（パッケージの __init__ は Isaac Lab を読み込むため）"""
    import importlib.util

    sp = importlib.util.spec_from_file_location("squat_spec", os.path.join(ROOT, "lab", "g1_squat", "spec.py"))
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    return mod


OBS_JOINTS = _load_spec().OBS_JOINTS


def quat_to_rpy(q):
    w, x, y, z = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def run(seed, sim_seconds=90.0):
    from g1_agent import sim as simmod
    from g1_agent import tasks

    rng = np.random.default_rng(seed)
    s = tasks.scene_shelf()
    s.start_pose = (0.0, 0.0, float(rng.uniform(-math.pi, math.pi)))
    sim = simmod.G1Sim(simmod.DEFAULT_RL_GYM, scene=s)
    sim.safety.stop_distance = sim.safety.slow_distance = 0.0
    m, d = sim.model, sim.data
    qa = np.array([m.joint(n).qposadr[0] for n in OBS_JOINTS])
    va = np.array([m.joint(n).dofadr[0] for n in OBS_JOINTS])
    out = {k: [] for k in ("height", "roll_pitch", "lin_vel_b", "ang_vel_b", "joint_pos", "joint_vel")}
    sim.step((0, 0, 0), 1.0)
    while sim.time < sim_seconds:
        kind = rng.choice(["walk", "turn", "creep", "stop"], p=[0.3, 0.25, 0.3, 0.15])
        if kind == "walk":
            cmd = (rng.uniform(-0.3, 0.5), rng.uniform(-0.2, 0.2), rng.uniform(-0.3, 0.3))
        elif kind == "turn":
            cmd = (0.0, 0.0, rng.choice([-1, 1]) * rng.uniform(0.15, 0.8))
        elif kind == "creep":
            cmd = (rng.choice([0.0, 0.08, -0.14, 0.12]), rng.choice([0.0, 0.08, -0.08]), rng.choice([0.0, 0.15, -0.15]))
        else:
            cmd = (0.0, 0.0, 0.0)
        sim.step(cmd, float(rng.uniform(0.3, 2.0)))
        sim.step((0, 0, 0), float(rng.uniform(0.0, 1.2)))
        t_end = sim.time + 0.8
        while sim.time < t_end and not sim.feet_on_ground():
            sim.step((0, 0, 0), 0.02)
        if sim.state.fallen:
            break
        roll, pitch, yaw = quat_to_rpy(d.qpos[3:7])
        c, s_ = math.cos(yaw), math.sin(yaw)
        v = d.qvel[0:3]  # 世界座標系
        out["height"].append(d.qpos[2])
        out["roll_pitch"].append((roll, pitch))
        out["lin_vel_b"].append((c * v[0] + s_ * v[1], -s_ * v[0] + c * v[1], v[2]))
        out["ang_vel_b"].append(d.qvel[3:6].copy())  # 体の座標系
        out["joint_pos"].append(d.qpos[qa].copy())
        out["joint_vel"].append(d.qvel[va].copy())
    return {k: np.asarray(v, dtype=np.float32) for k, v in out.items()}


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "lab", "handoff_states.npz")
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 48
    with Pool(min(16, os.cpu_count() or 4)) as pool:
        parts = pool.map(run, range(n))
    data = {k: np.concatenate([p[k] for p in parts if len(p[k])]) for k in parts[0]}
    np.savez(path, joint_names=np.array(OBS_JOINTS), **data)
    print(f"{len(data['height'])} 個の状態を保存しました: {path}")
    for k in ("height", "lin_vel_b", "ang_vel_b"):
        print(f"  {k}: 平均 {np.round(data[k].mean(0), 3)}  標準偏差 {np.round(data[k].std(0), 3)}")


if __name__ == "__main__":
    main()
