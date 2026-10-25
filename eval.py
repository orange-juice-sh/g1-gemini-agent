"""複数のモデルで評価タスクを実行し、成功率・遅延・費用を比べる

  python eval.py --models argon er2 flash qwen rule --trials 3 --workers 4
  python eval.py --models rule --tasks T1 T2

モデル: argon / er2 / flash / qwen / rule（run.py と同じ gemini:<ID> なども可）
結果: results/eval-<日時>/（runs.csv, summary.md, logs/）"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

ALIASES = {
    "argon": "gemini:gemini-4-argon",
    "er2": "gemini:gemini-robotics-er-2-preview",
    "flash": "gemini:gemini-3.8-flash",
    "qwen": "local",
    "rule": "mock",
}
# USD / 100万トークン（入力, 出力, キャッシュ入力）、2026-10-17 時点の公式料金
# ER 2 のキャッシュ単価は記載がないため入力の 1/10 と仮定
PRICES = {
    "gemini-4-argon": (2.00, 10.00, 0.10),
    "gemini-robotics-er-2-preview": (1.00, 5.00, 0.10),
    "gemini-3.8-flash": (0.75, 3.75, 0.075),
}


def cost_usd(model_id, input_tokens, output_tokens, cache_read_tokens):
    """キャッシュから読んだ入力は安い単価で数える"""
    p_in, p_out, p_cache = PRICES.get(model_id, (0.0, 0.0, 0.0))
    fresh = max(0, input_tokens - cache_read_tokens)
    return (fresh * p_in + cache_read_tokens * p_cache + output_tokens * p_out) / 1e6


def resolve(name):
    return ALIASES.get(name, name)


def short(name):
    for k, v in ALIASES.items():
        if resolve(name) == v:
            return k
    return name


def wilson(k, n, z=1.96):
    """Wilson の 95% 信頼区間"""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def run_task(model, task_id, trial, log_dir, max_turns):
    """1試行を実行する（別プロセスで呼ばれる）"""
    from g1_agent.agent import G1Agent, make_client, summarize_usage
    from g1_agent.sim import DEFAULT_RL_GYM, G1Sim
    from g1_agent.skills import Robot
    from g1_agent.tasks import TASKS_BY_ID

    task = TASKS_BY_ID[task_id]
    sim = G1Sim(DEFAULT_RL_GYM, scene=task.scene())
    robot = Robot(sim)
    robot.auto_detour = task.assist
    sim.step((0, 0, 0), 1.0)
    row = {"model": short(model), "task": task.id, "kind": task.kind, "trial": trial}
    try:
        client, name = make_client(resolve(model))
        agent = G1Agent(robot, client=client, model=name, verbose=False, max_turns=max_turns)
        log = agent.run(task.text)
        ok = bool(task.check(sim, log)) and not sim.state.fallen
        usage = summarize_usage(log)
        cost = cost_usd(name, usage["input_tokens"], usage["output_tokens"], usage["cache_read_tokens"])
        row.update(model_id=name, success=ok, sim_time_s=log.sim_time_s, wall_time_s=log.wall_time_s,
                   fallen=sim.state.fallen, safety_stops=sim.state.safety_stops, cost_usd=round(cost, 5),
                   final_text=log.final_text, **usage)
        with open(os.path.join(log_dir, f"{short(model)}_{task.id}_{trial}.json"), "w", encoding="utf-8") as f:
            json.dump(log.to_dict() | {"success": ok, "usage": usage, "cost_usd": cost,
                                       "final_pose": sim.pose(), "events": sim.state.events},
                      f, ensure_ascii=False, indent=1, default=str)
    except Exception as e:
        row.update(success=False, error=f"{type(e).__name__}: {str(e)[:300]}")
    finally:
        robot.close()
    return row


def summarize(rows, models, tasks):
    lines = ["| モデル | " + " | ".join(f"{t.id}({t.kind})" for t in tasks)
             + " | 全体 [95%CI] | 平均API遅延[s] | 平均ツール呼び出し | 費用[$] | 成功1回あたり[$] |",
             "|---|" + "---|" * (len(tasks) + 5)]
    for m in models:
        mr = [r for r in rows if r["model"] == short(m)]
        cells = []
        for t in tasks:
            tr = [r for r in mr if r["task"] == t.id]
            cells.append(f"{sum(r['success'] for r in tr)}/{len(tr)}")
        k, n = sum(r["success"] for r in mr), len(mr)
        lo, hi = wilson(k, n)
        lat = [r["mean_latency_s"] for r in mr if r.get("mean_latency_s")]
        tc = [r["tool_calls"] for r in mr if r.get("tool_calls") is not None]
        cost = sum(r.get("cost_usd") or 0 for r in mr)
        per = f"{cost / k:.3f}" if k else "-"
        lines.append(f"| {short(m)} | " + " | ".join(cells) +
                     f" | {k / max(1, n):.0%} [{lo:.0%}–{hi:.0%}] | {sum(lat) / len(lat) if lat else 0:.1f} "
                     f"| {sum(tc) / max(1, len(tc)):.1f} | {cost:.3f} | {per} |")
    return "\n".join(lines)


def main():
    from g1_agent.tasks import TASKS

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["argon", "rule"])
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--tasks", nargs="*", help="T1 T9 のように一部だけ実行")
    ap.add_argument("--workers", type=int, default=4, help="同時に実行する試行の数")
    ap.add_argument("--max-turns", type=int, default=30, help="1試行で AI に問い合わせる回数の上限")
    args = ap.parse_args()

    from run import load_dotenv

    load_dotenv()
    if sys.platform.startswith("linux"):
        os.environ.setdefault("MUJOCO_GL", "egl")
    tasks = [t for t in TASKS if not args.tasks or t.id in args.tasks]
    out = os.path.join("results", "eval-" + time.strftime("%Y%m%d-%H%M%S"))
    log_dir = os.path.join(out, "logs")
    os.makedirs(log_dir, exist_ok=True)

    jobs = [(m, t.id, trial) for m in args.models for t in tasks
            for trial in range(1 if resolve(m) == "mock" else args.trials)]
    print(f"{len(jobs)} 試行を {args.workers} 並列で実行します → {out}", flush=True)
    rows = []
    # EGL を使うため spawn にする
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as ex:
        futs = [ex.submit(run_task, m, tid, trial, log_dir, args.max_turns) for m, tid, trial in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            rows.append(r)
            print(f"[{i}/{len(jobs)}] {r['model']:6s} {r['task']:4s} #{r['trial']}: {'成功' if r['success'] else '失敗'}"
                  f"  tools={r.get('tool_calls')} latency={r.get('mean_latency_s')}s cost=${r.get('cost_usd', 0)}"
                  + (f"  {r['error']}" if r.get("error") else ""), flush=True)

    order = {(short(m), t.id, k): i for i, (m, t, k) in enumerate((m, t, k) for m in args.models for t in tasks
                                                                    for k in range(args.trials))}
    rows.sort(key=lambda r: order.get((r["model"], r["task"], r["trial"]), 0))
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(os.path.join(out, "runs.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    table = summarize(rows, args.models, tasks)
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(f"コマンド: `python eval.py {' '.join(sys.argv[1:])}`\n\n")
        f.write("\n".join(f"- {t.id}（{t.kind}）: {t.text}" + (f" — {t.note}" if t.note else "") for t in tasks))
        f.write("\n\n" + table + "\n")
    print("\n" + table)
    print(f"\n保存先: {out}/summary.md, runs.csv, logs/")


if __name__ == "__main__":
    main()
