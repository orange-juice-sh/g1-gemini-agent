"""自然言語の指示で、MuJoCo 上の G1 を AI に動かしてもらう

  python run.py "受付に行って"                   # 1つの指示（既定は Gemini 4 Argon）
  python run.py                                  # 対話モード
  python run.py --live --scenario video          # 提出動画の4つの指示（ブラウザに表示）
  python run.py --scenario demo --record videos/demo.mp4 --speed 2
  python run.py --mock "受付に行って"            # ルールベース（API なし）
  python run.py --local "受付に行って"           # ローカルの Qwen3-VL"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

SCENARIOS = {
    "demo": dict(scene="scene_demo", steps=[
        ("来客対応", "受付に行って、誰かいるか確認してください。人がいたら手を振りながら「いらっしゃいませ。ご用件を伺います。」と伝えてください。"),
        ("案内板を読んで道案内", "打ち合わせに来たお客さんに場所を聞かれました。入口の近くにある案内板を読んで場所を確かめたら、"
                         "その場でその場所を指差しながら「打ち合わせの場所はあちらです」と伝えてください。"),
        ("忘れ物を届ける", "打ち合わせの場所に赤い忘れ物の箱があるはずです。探して受付まで運び、受付カウンターの上に置いてください。"),
        ("エレベーターを呼ぶ", "受付の人によると、忘れ物の持ち主は2階にいるそうです。エレベーターを呼んでください。"),
        ("見回りの終了", "見回りは終わりです。充電ステーションに戻ってください。"),
    ]),
    "video": dict(scene="scene_shelf", steps=[
        ("赤い忘れ物を届ける", "休憩スペースにある赤色の四角い物体を受付に持って行ってください。"),
        ("青い忘れ物を届ける", "休憩スペースにある青色の丸い物体を受付に持って行ってください。"),
        ("受付への報告", "受付の人に手を振り、忘れ物を届けたことを伝えてください。"),
        ("エレベーターと充電", "エレベーターのスイッチを押して、再度充電スペースに戻ってください。"),
    ]),
}


def load_dotenv(path=".env"):
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def eval_card(dirs):
    """動画の最後に出すモデル比較のカードの文"""
    import csv

    from eval import wilson
    from plot_results import LABEL, ORDER

    from plot_results import load

    rows = [dict(r, success=str(r["success"])) for r in load(dirs)]
    lines = []
    for m in [m for m in ORDER if any(r["model"] == m for r in rows)]:
        mr = [r for r in rows if r["model"] == m]
        k, n = sum(r["success"] == "True" for r in mr), len(mr)
        lo, hi = wilson(k, n)
        lines.append(f"{LABEL.get(m, m)}：{k / n:.0%}（{k}/{n}、95%信頼区間 {lo:.0%}〜{hi:.0%}）")
    ntask = len({r["task"] for r in rows})
    return (f"定量評価：{ntask} 種類のタスクの成功率", lines,
            "タスク: 移動・探索・対話・曖昧な指示・画像の読み取り・状況の変化・長い手順・腕を使う動作")


def summary_lines(logs, speaker):
    """動画の最後のまとめカードの行"""
    tools = sum(l["usage"]["tool_calls"] for l in logs)
    retries = sum(l["usage"]["safety_retries"] for l in logs)
    looks = sum(1 for l in logs for s in l["steps"] if s["tool"] in ("look", "find_object"))
    sim_t = sum(l["sim_time_s"] for l in logs)
    gestures = sum(1 for l in logs for s in l["steps"] if s["tool"] in ("gesture", "press_button"))
    return [f"{len(logs)} つの指示を {speaker} が {tools} 回のツール呼び出しで実行",
            f"安全層による停止 → AI が回避して再計画：{retries} 回",
            f"カメラ画像による確認：{looks} 回　／　腕の動作（身ぶり・ボタン）：{gestures} 回",
            f"シミュレーション時間 {sim_t:.0f} 秒"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("instruction", nargs="?", help="ロボットへの指示（省略すると対話モード）")
    ap.add_argument("--scenario", choices=SCENARIOS, help="用意した指示の列を順に実行する")
    ap.add_argument("--scene", help="使うロビー（g1_agent/tasks.py の関数名。例: scene_shelf）。--live では既定で scene_shelf")
    ap.add_argument("--task", help="評価タスク（例: T16）のシーンと指示で実行する（g1_agent/tasks.py）")
    ap.add_argument("--model", default=None,
                    help="頭脳にするモデル。gemini（既定: Gemini 4 Argon）、Claude のモデル名（例: claude-sonnet-5-5）、"
                         "local（Qwen3-VL）、mock")
    ap.add_argument("--mock", action="store_true", help="Claude API の代わりにルールベースの代役を使う")
    ap.add_argument("--local", action="store_true", help="ローカルの VLM（既定: Qwen3-VL-30B-A3B）を使う")
    ap.add_argument("--record", metavar="MP4", help="字幕・カメラ小窓・ミニマップ付きの動画を保存する")
    ap.add_argument("--speed", type=float, default=1.0, help="録画の再生速度（2 なら 2倍速）")
    ap.add_argument("--live", action="store_true",
                    help="ターミナルで対話しながら、同じ動きをブラウザに実時間で映す（http://localhost:8765。SSH のポート転送で開く）")
    ap.add_argument("--port", type=int, default=8765, help="--live の映像を配信するポート（コンテナ内）")
    ap.add_argument("--replay", metavar="LOG",
                    help="実行ログ（logs/run-*.json）のツール呼び出しを再生して録画し直す（API を使わない。--scenario と一緒に使う）")
    ap.add_argument("--tts", action="store_true",
                    help="録画で、ロボットの発話を音声にする（VOICEVOX。docker compose --profile tts up -d で起動しておく）")
    ap.add_argument("--eval", nargs="*", metavar="DIR",
                    help="動画の最後に評価結果のカードを出す（eval.py の出力フォルダ results/eval-...）")
    ap.add_argument("--viewer", action="store_true", help="MuJoCo のビューアで実時間表示する")
    ap.add_argument("--rl-gym", default=None, help="unitree_rl_gym のパス（既定: ~/unitree_rl_gym）")
    ap.add_argument("--log-dir", default="logs")
    ap.add_argument("--torch", action="store_true", help="歩行ポリシーを PyTorch で実行する（既定は NumPy）")
    args = ap.parse_args()

    load_dotenv()
    if (args.record or args.live) and "MUJOCO_GL" not in os.environ and sys.platform.startswith("linux") and not args.viewer:
        os.environ["MUJOCO_GL"] = "egl"

    from g1_agent import sim as simmod
    from g1_agent.agent import G1Agent, make_client, speaker_name, summarize_usage
    from g1_agent.skills import Robot

    from g1_agent import tasks as taskmod

    task = taskmod.TASKS_BY_ID[args.task] if args.task else None
    scene = task.scene() if task else None
    if args.scenario and SCENARIOS[args.scenario]["scene"]:
        scene = getattr(taskmod, SCENARIOS[args.scenario]["scene"])()
    scene_name = args.scene or ("scene_shelf" if args.live and not task and not args.scenario else None)
    if scene_name:
        scene = getattr(taskmod, scene_name)()
    sim = simmod.G1Sim(args.rl_gym or simmod.DEFAULT_RL_GYM, use_torch=args.torch, scene=scene)
    robot = Robot(sim)
    sim.step((0, 0, 0), 1.0)

    default_model = os.environ.get("G1_MODEL", "gemini")
    spec = "mock" if args.mock else "local" if args.local else f"replay:{args.replay}" if args.replay \
        else (args.model or default_model)
    for prefix, env in (("claude", "ANTHROPIC_API_KEY"), ("gemini", "GEMINI_API_KEY")):
        key = os.environ.get(env, "")
        if spec.startswith(prefix) and (not key or key.endswith("...")):
            sys.exit(f"{env} が設定されていません。.env に書くか、--mock で試してください。")
    client, model = make_client(spec)

    recorder = viewer = None
    if args.live:
        import webui
        from g1_agent import tts as ttsmod

        hub = webui.Hub()
        recorder = webui.make_live_view(hub)(sim, robot)
        recorder.tts = ttsmod.available()
        recorder.speed = hub.playback = args.speed
        recorder.hide_speed_badge = True
        if args.record:
            os.makedirs(os.path.dirname(args.record) or ".", exist_ok=True)
            recorder.start_recording(args.record)
        webui.serve_in_background(hub, args.port, "viewer.html")
        hub.update(ready=True)
        print(f"映像を配信しています: 手元の PC のブラウザで http://localhost:{os.environ.get('WEBUI_PORT', args.port)} "
              f"を開いてください（ssh -L {os.environ.get('WEBUI_PORT', args.port)}:localhost:"
              f"{os.environ.get('WEBUI_PORT', args.port)} で接続しておく）。"
              + ("音声あり（ブラウザで一度クリック）。" if recorder.tts else "音声なし（VOICEVOX が起動していません）。"),
              flush=True)
    elif args.record:
        from g1_agent.recorder import Recorder

        os.makedirs(os.path.dirname(args.record) or ".", exist_ok=True)
        recorder = Recorder(sim, robot, args.record, speed=args.speed)
        if args.tts:
            from g1_agent import tts

            if not tts.available():
                sys.exit("音声合成（VOICEVOX）に接続できません。docker compose --profile tts up -d で起動してください")
            recorder.tts = True
        recorder.title_card([f"{speaker_name(model)} × Unitree G1", "自然言語で動くロビー案内・見回りヒューマノイド",
                             f"MuJoCo シミュレーション / AI: {model.split('/')[-1]}"
                             + (f" / {args.speed:g}倍速で再生" if args.speed != 1 else "")
                             + ("（発話の場面は等倍）" if args.tts else "")]
                            + (["音声: VOICEVOX:No.7"] if args.tts else []),
                            seconds=4.0)
        recorder.architecture_card(speaker_name(model))
    if args.viewer:
        from g1_agent.recorder import LiveViewer

        viewer = LiveViewer(sim)

    agent = G1Agent(robot, client=client, model=model, recorder=recorder, viewer=viewer)
    agent.pretty = args.live
    robot.auto_detour = args.live

    if args.scenario:
        instructions = SCENARIOS[args.scenario]["steps"]
    elif task:
        instructions = [(f"{task.id}（{task.kind}）", task.text)]
    elif args.instruction:
        instructions = [("", args.instruction)]
    else:
        instructions = None

    os.makedirs(args.log_dir, exist_ok=True)
    logs = []

    def run_one(text):
        print(f"\n=== 指示: {text}")
        try:
            log = agent.run(text)
        except Exception as e:
            print(f"\n=== エラー（{type(e).__name__}）: {e}")
            return
        logs.append(log.to_dict() | {"usage": summarize_usage(log)})
        print(f"=== 報告: {log.final_text}")
        print(f"    {summarize_usage(log)}  シミュレーション時間 {log.sim_time_s}s")

    try:
        if instructions:
            for i, (title, text) in enumerate(instructions, 1):
                if recorder:
                    recorder.chapter_card(i, len(instructions), title, text)
                run_one(text)
            if recorder:
                recorder.end_card("このデモのまとめ", summary_lines(logs, speaker_name(model)),
                                  footer="AI（高レベル判断）＋ 学習済み歩行ポリシー（低レベル制御）＋ 独立した安全層"
                                         + ("　／　音声: VOICEVOX:No.7" if args.tts else ""))
                if args.eval:
                    title, lines, foot = eval_card(args.eval)
                    recorder.end_card(title, lines, seconds=8.0, footer=foot)
        elif args.live:
            # 入力待ちの間もシミュレーションを進める
            import queue
            import threading

            q, ready = queue.Queue(), threading.Event()

            def read_input():
                while True:
                    ready.wait()
                    ready.clear()
                    try:
                        text = input("\n指示を入力してください（空行で終了）> ").strip()
                        q.put(text.encode("utf-8", "surrogateescape").decode("utf-8", "replace"))
                    except EOFError:
                        q.put("")

            threading.Thread(target=read_input, daemon=True).start()
            ready.set()
            while True:
                try:
                    text = q.get(timeout=0.1)
                except queue.Empty:
                    robot.idle(0.1)
                    continue
                if not text:
                    break
                run_one(text)
                ready.set()
        else:
            print("指示を入力してください（空行で終了）")
            while True:
                text = input("> ").strip()
                if not text:
                    break
                run_one(text)
    finally:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = os.path.join(args.log_dir, f"run-{stamp}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(logs, f, ensure_ascii=False, indent=2, default=str)
        print(f"ログ: {path}")
        if recorder and args.live:
            saved = recorder.stop_recording()
            recorder.close()
            if saved:
                print(f"動画: {saved}")
        elif recorder:
            secs = recorder.close()
            print(f"動画: {args.record}（{secs:.0f}秒）")
        robot.close()
        if args.replay:
            print(f"再生の確認: 元のログとの位置のずれの最大値 {client.max_dev:.3f} m")


if __name__ == "__main__":
    main()
