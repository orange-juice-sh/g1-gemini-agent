"""しゃがむタスクの学習・再生（lab サービスの中で実行）
  bash /project/lab/run.sh train --num_envs 4096 --max_iterations 2000
  bash /project/lab/run.sh play --checkpoint <model_XXXX.pt>"""
import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g1_squat  # noqa: E402,F401

mode = sys.argv.pop(1) if len(sys.argv) > 1 else "train"
script = f"/workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/{mode}.py"
if "--task" not in sys.argv:
    sys.argv += ["--task", "G1-Squat-v0" if mode == "train" else "G1-Squat-Play-v0"]
sys.argv[0] = script
sys.path.insert(0, os.path.dirname(script))
runpy.run_path(script, run_name="__main__")
