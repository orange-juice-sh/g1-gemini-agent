"""評価結果（runs.csv）からグラフを作る
  python plot_results.py results/eval-A results/eval-B --out results/figures"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval import cost_usd, wilson  # noqa: E402

ORDER = ["argon", "er2", "flash", "qwen", "rule"]
LABEL = {"argon": "Gemini 4 Argon", "er2": "Gemini Robotics-ER 2", "flash": "Gemini 3.8 Flash",
         "qwen": "Qwen3-VL（ローカル）", "rule": "ルールベース"}
KIND_ORDER = ["移動", "探索", "対話", "曖昧", "視覚", "文字", "変化", "長手順", "腕", "デモ"]

BLUE = "#2a78d6"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
TEXT, TEXT2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def setup_font():
    for p in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc"):
        if os.path.exists(p):
            font_manager.fontManager.addfont(p)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=p).get_name()
            break
    plt.rcParams.update({"axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "xtick.color": TEXT2,
                         "ytick.color": TEXT2, "text.color": TEXT, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight"})


def load(dirs):
    """runs.csv を読む
    後のフォルダに同じ (モデル, タスク) があれば置き換える"""
    rows = []
    for d in dirs:
        with open(os.path.join(d, "runs.csv"), encoding="utf-8") as f:
            new = list(csv.DictReader(f))
        redo = {(r["model"], r["task"]) for r in new}
        rows = [r for r in rows if (r["model"], r["task"]) not in redo]
        for r in new:
            r["success"] = r["success"] == "True"
            if r.get("model_id"):
                r["cost_usd"] = cost_usd(r["model_id"], int(r.get("input_tokens") or 0),
                                         int(r.get("output_tokens") or 0), int(r.get("cache_read_tokens") or 0))
            rows.append(r)
    return rows


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def models_in(rows):
    present = {r["model"] for r in rows}
    return [m for m in ORDER if m in present] + sorted(present - set(ORDER))


def plot_success(rows, out):
    models = models_in(rows)[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 0.6 * len(models) + 1.2))
    for i, m in enumerate(models):
        mr = [r for r in rows if r["model"] == m]
        k, n = sum(r["success"] for r in mr), len(mr)
        lo, hi = wilson(k, n)
        p = k / max(1, n)
        ax.barh(i, p * 100, height=0.55, color=BLUE)
        ax.errorbar(p * 100, i, xerr=[[(p - lo) * 100], [(hi - p) * 100]], fmt="none", ecolor=TEXT2,
                    elinewidth=1.2, capsize=4)
        ax.text(hi * 100 + 2, i, f"{p:.0%}（{k}/{n}）", va="center", fontsize=9, color=TEXT)
    ax.set_yticks(range(len(models)), [LABEL.get(m, m) for m in models])
    ax.set_xlim(0, 115)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("全タスクの成功率（線は 95% 信頼区間）", fontsize=11, loc="left")
    fig.savefig(os.path.join(out, "success_rate.png"))
    plt.close(fig)


def plot_by_kind(rows, out):
    models = models_in(rows)
    kinds = [k for k in KIND_ORDER if any(r["kind"] == k for r in rows)]
    fig, ax = plt.subplots(figsize=(1.0 * len(kinds) + 2.6, 0.55 * len(models) + 1.3))
    for i, m in enumerate(models):
        for j, kd in enumerate(kinds):
            tr = [r for r in rows if r["model"] == m and r["kind"] == kd]
            if not tr:
                continue
            p = sum(r["success"] for r in tr) / len(tr)
            color = SEQ[min(len(SEQ) - 1, int(round(p * (len(SEQ) - 1))))]
            ax.add_patch(plt.Rectangle((j + 0.03, i + 0.04), 0.94, 0.92, color=color, linewidth=0))
            ax.text(j + 0.5, i + 0.5, f"{p:.0%}", ha="center", va="center", fontsize=9,
                    color="white" if p >= 0.55 else TEXT)
    ax.set_xlim(0, len(kinds))
    ax.set_ylim(len(models), 0)
    ax.set_xticks([j + 0.5 for j in range(len(kinds))], kinds)
    ax.set_yticks([i + 0.5 for i in range(len(models))], [LABEL.get(m, m) for m in models])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("タスクの種類ごとの成功率", fontsize=11, loc="left")
    fig.savefig(os.path.join(out, "success_by_kind.png"))
    plt.close(fig)


def plot_cost_latency(rows, out):
    models = [m for m in models_in(rows) if m not in ("rule",)][::-1]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 0.6 * len(models) + 1.3), sharey=True)
    for i, m in enumerate(models):
        mr = [r for r in rows if r["model"] == m]
        k = sum(r["success"] for r in mr)
        cost = sum(num(r.get("cost_usd")) or 0 for r in mr)
        per = cost / k if k else None
        lat = [num(r.get("mean_latency_s")) for r in mr if num(r.get("mean_latency_s"))]
        if per == 0:
            a1.text(0, i, "  API 費用なし（自前の GPU で実行）", va="center", fontsize=9, color=TEXT2)
        elif per is not None:
            a1.barh(i, per * 150, height=0.55, color=BLUE)
            a1.text(per * 150, i, f"  {per * 150:.1f}円（${per:.3f}）", va="center", fontsize=9)
        else:
            a1.text(0, i, "  成功なし", va="center", fontsize=9, color=TEXT2)
        if lat:
            v = sum(lat) / len(lat)
            a2.barh(i, v, height=0.55, color=BLUE)
            a2.text(v, i, f"  {v:.1f}秒", va="center", fontsize=9)
    a1.set_yticks(range(len(models)), [LABEL.get(m, m) for m in models])
    a1.set_title("成功1回あたりの費用（1ドル=150円で換算）", fontsize=11, loc="left")
    a2.set_title("AI の平均応答時間（1回の問い合わせ）", fontsize=11, loc="left")
    for a in (a1, a2):
        a.xaxis.grid(True, color=GRID, linewidth=0.8)
        a.set_axisbelow(True)
        a.margins(x=0.35)
    fig.savefig(os.path.join(out, "cost_latency.png"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", help="eval.py の出力フォルダ（results/eval-...）")
    ap.add_argument("--out", default=None, help="グラフの保存先（既定: 最初のフォルダ）")
    args = ap.parse_args()
    out = args.out or args.dirs[0]
    os.makedirs(out, exist_ok=True)
    setup_font()
    rows = load(args.dirs)
    plot_success(rows, out)
    plot_by_kind(rows, out)
    plot_cost_latency(rows, out)
    print(f"グラフを保存しました: {out}/success_rate.png, success_by_kind.png, cost_latency.png")


if __name__ == "__main__":
    main()
