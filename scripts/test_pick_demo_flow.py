"""撮影と同じ流れで、棚の下の段の物をしゃがんで取って受付に置けるかを試す（AI なし）
  python scripts/test_pick_demo_flow.py [ポリシー.npz] [回数]"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if len(sys.argv) > 1 and sys.argv[1].endswith(".npz"):
    os.environ["G1_SQUAT_POLICY"] = sys.argv[1]
from g1_agent import sim as simmod, tasks, world  # noqa: E402
from g1_agent.skills import Robot  # noqa: E402

n_each = int(sys.argv[2]) if len(sys.argv) > 2 else 2
locs = {l.name: l for l in world.LOCATIONS}
results = {}
for color in ("blue", "yellow"):
    for name in ("受付", "休憩スペース", "充電ステーション"):
        loc = locs[name]
        for k in range(n_each):
            s = tasks.scene_shelf()
            s.start_pose = (loc.x, loc.y, math.radians(-90 + 70 * k))
            sim = simmod.G1Sim(simmod.DEFAULT_RL_GYM, scene=s)
            r = Robot(sim)
            r.auto_detour = True
            sim.step((0, 0, 0), 1.0)
            steps = []
            for f in (lambda: r.walk_to(location="休憩スペース"), lambda: r.pick_up(color),
                      lambda: r.walk_to(location="受付"), lambda: r.put_down()):
                res = f()
                steps.append(res["success"])
                if not res["success"]:
                    steps.append(res["message"][:40])
                    break
            ok = steps == [True, True, True, True] and not sim.state.fallen
            results.setdefault(color, []).append(ok)
            print(f"{color} {name} 向き{-90 + 70 * k}°: {'成功' if ok else '失敗'} {steps}", flush=True)
            r.close()
for color, v in results.items():
    print(f"{color}: {sum(v)}/{len(v)} 成功")
