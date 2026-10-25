"""ブラウザから指示を出して G1 を動かす Web UI
  python webui.py   # http://localhost:8765
描画（EGL）は作ったスレッドでしか使えないので、シミュレーションと AI は worker スレッドで動かす"""
from __future__ import annotations

import argparse
import io
import json
import os
import queue
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image

MODELS = [("qwen", "Qwen3-VL（ローカル・無料）"), ("argon", "Gemini 4 Argon"), ("er2", "Gemini Robotics-ER 2"),
          ("flash", "Gemini 3.8 Flash"), ("rule", "ルールベース")]
SCENES = [("scene_demo", "ロビー（案内板あり）"), ("default", "標準のロビー"),
          ("scene_people", "来客あり（オレンジと紫の服の人）"), ("scene_count", "休憩スペースの床に小さな箱")]
EXAMPLES = ["受付に行って、誰かいるか確認してください。人がいたら手を振りながら挨拶してください。",
            "入口の近くにある案内板を読んで、打ち合わせの場所を教えてください。",
            "休憩スペースの棚の上にある赤い忘れ物を、受付カウンターまで運んでください。",
            "エレベーターを呼んでください。",
            "会議室の方向を指差しながら「会議室はあちらです」と伝えてください。",
            "少し疲れたので、座って休める場所に行ってください。",
            "見回りは終わりです。充電ステーションに戻ってください。"]


class Hub:
    def __init__(self):
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.jpeg, self.frame_id = None, 0
        self.commands = queue.Queue()
        self.state = dict(busy=False, model="qwen", scene="scene_demo", history=[], caption="", recording=None,
                          tts=False, error="", ready=False, speech_id=0)
        self.speech = {}
        self.playback = 1.0

    def publish(self, frame):
        buf = io.BytesIO()
        Image.fromarray(frame).save(buf, format="JPEG", quality=82)
        with self.cond:
            self.jpeg, self.frame_id = buf.getvalue(), self.frame_id + 1
            self.cond.notify_all()

    def update(self, **kw):
        with self.lock:
            self.state.update(kw)

    def snapshot(self):
        with self.lock:
            return json.loads(json.dumps(self.state, ensure_ascii=False, default=str))


def make_live_view(hub):
    from g1_agent.recorder import Recorder

    class LiveView(Recorder):
        """録画クラスを流用して、画像をブラウザへ配信する"""

        def __init__(self, sim, robot):
            super().__init__(sim, robot, None, size=(1280, 720), fps=15, speed=1.0)
            self._sync()

        def _sync(self):
            self._wall0, self._sim0 = time.time(), self.sim.time

        def _write(self, frame):
            super()._write(frame)
            hub.publish(frame)

        def _on_step(self, sim):
            if sim.time + 1e-9 >= self.next_t:
                self.next_t += self.speed / self.fps
                self._write(self._compose())
                ahead = (sim.time - self._sim0) / hub.playback - (time.time() - self._wall0)
                if ahead > 0:
                    time.sleep(ahead)
                elif ahead < -0.5:
                    self._sync()

        def hold(self, seconds, thinking=True):
            self.thinking = thinking
            frame = self._compose()
            self._write(frame)
            if self.writer is not None:
                for _ in range(max(0, int(seconds * self.fps) - 1)):
                    self.writer.append_data(frame)
            self.thinking = False
            self._sync()

        def speak_begin(self, text):
            if not self.tts:
                return None
            from g1_agent import tts

            wav, dur = tts.synthesize(text)
            with hub.lock:
                hub.state["speech_id"] += 1
                hub.speech[hub.state["speech_id"]] = wav
                for k in [k for k in hub.speech if k < hub.state["speech_id"] - 5]:
                    del hub.speech[k]
            if self.writer is not None:
                import tempfile

                path = os.path.join(tempfile.gettempdir(), f"g1_webui_say_{len(self.speech_clips)}.wav")
                open(path, "wb").write(wav)
                self.speech_clips.append((self.frames / self.fps, path))
            return dur

        def speak_end(self):
            self._sync()

        def start_recording(self, path):
            import imageio.v2 as imageio

            from g1_agent.recorder import _pick_encoder

            self.codec, params = _pick_encoder()
            self.path, self.frames, self.speech_clips = path, 0, []
            self.writer = imageio.get_writer(path, fps=self.fps, codec=self.codec, quality=None, macro_block_size=8,
                                             ffmpeg_params=params + ["-movflags", "+faststart"])

        def stop_recording(self):
            if self.writer is None:
                return None
            self.writer.close()
            if self.speech_clips:
                self._mux_audio()
            self.writer = None
            return self.path

    return LiveView


def worker(hub, args):
    os.environ.setdefault("MUJOCO_GL", "egl")
    from eval import cost_usd, resolve
    from g1_agent import tasks as taskmod
    from g1_agent import tts
    from g1_agent.agent import G1Agent, make_client, speaker_name
    from g1_agent.sim import DEFAULT_RL_GYM, G1Sim
    from g1_agent.skills import Robot

    LiveView = make_live_view(hub)

    class WebAgent(G1Agent):
        def _caption(self, text):
            super()._caption(text)
            hub.update(caption=text)

        def _execute(self, name, args_):
            res, image = super()._execute(name, args_)
            with hub.lock:
                entry = hub.state["history"][-1]
                entry["steps"].append({"label": self._describe_call(name, args_).lstrip("▶ "),
                                       "ok": res.get("success", True), "msg": res.get("message", "")[:120]})
            return res, image

    tts_on = tts.available()
    hub.update(tts=tts_on)
    env = dict(sim=None, robot=None, view=None, agent=None, model=None)

    def setup(scene_name):
        if env["robot"]:
            if env["view"]:
                env["view"].stop_recording()
                env["view"].close()
            env["robot"].close()
        import g1_agent.world as world

        scene = world.default_scene() if scene_name == "default" else getattr(taskmod, scene_name)()
        taskmod.add_shelf_with_red_box(scene)
        sim = G1Sim(DEFAULT_RL_GYM, scene=scene)
        robot = Robot(sim)
        robot.auto_detour = True
        sim.step((0, 0, 0), 0.5)
        view = LiveView(sim, robot)
        view.tts = tts_on
        env.update(sim=sim, robot=robot, view=view, agent=None, model=None)
        hub.update(scene=scene_name, history=[], caption="", recording=None, ready=True)

    def agent_for(model):
        if env["agent"] is None or env["model"] != model:
            client, name = make_client(resolve(model))
            env["agent"] = WebAgent(env["robot"], client=client, model=name, recorder=env["view"])
            env["model"] = model
            env["view"].speaker = speaker_name(name)
        return env["agent"]

    setup(args.scene)
    while True:
        try:
            cmd = hub.commands.get(timeout=0.1)
        except queue.Empty:
            env["robot"].idle(0.1)
            continue
        try:
            if cmd["type"] == "reset":
                setup(cmd.get("scene", "scene_demo"))
            elif cmd["type"] == "record":
                if cmd["on"]:
                    os.makedirs("videos", exist_ok=True)
                    path = time.strftime("videos/webui-%Y%m%d-%H%M%S.mp4")
                    env["view"].start_recording(path)
                    hub.update(recording=path)
                else:
                    path = env["view"].stop_recording()
                    hub.update(recording=None, error=f"録画を保存しました: {path}" if path else "")
            elif cmd["type"] == "instruct":
                hub.update(busy=True, model=cmd["model"], error="")
                with hub.lock:
                    hub.state["history"].append({"instruction": cmd["text"], "model": cmd["model"], "steps": [],
                                                 "report": "", "cost_usd": None, "latency_s": None})
                agent = agent_for(cmd["model"])
                env["view"].chapter = ""
                log = agent.run(cmd["text"])
                calls = log.api_calls
                cost = sum(cost_usd(log.model, c["input_tokens"], c["output_tokens"], c["cache_read"]) for c in calls)
                lat = sum(c["latency_s"] for c in calls) / max(1, len(calls))
                with hub.lock:
                    e = hub.state["history"][-1]
                    e.update(report=log.final_text or "（報告なし）", cost_usd=round(cost, 4), latency_s=round(lat, 1),
                             fallen=env["sim"].state.fallen)
                os.makedirs("logs", exist_ok=True)
                with open(time.strftime("logs/webui-%Y%m%d-%H%M%S.json"), "w", encoding="utf-8") as f:
                    json.dump([log.to_dict()], f, ensure_ascii=False, indent=1, default=str)
        except Exception as e:
            traceback.print_exc()
            hub.update(error=f"{type(e).__name__}: {str(e)[:300]}")
        finally:
            if cmd["type"] == "instruct":
                hub.update(busy=False)


def serve_in_background(hub, port, page_file="webui.html"):
    """HTTP サーバーを別スレッドで動かす（run.py --live 用）"""
    server = ThreadingHTTPServer(("0.0.0.0", port), make_handler(hub, page_file))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def make_handler(hub, page_file="webui.html"):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):
            pass

        def _send(self, code, body, ctype="application/json; charset=utf-8"):
            data = body if isinstance(body, bytes) else body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                page = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), page_file), encoding="utf-8").read()
                page = page.replace("__MODELS__", json.dumps(MODELS, ensure_ascii=False)) \
                    .replace("__SCENES__", json.dumps(SCENES, ensure_ascii=False)) \
                    .replace("__EXAMPLES__", json.dumps(EXAMPLES, ensure_ascii=False))
                return self._send(200, page, "text/html; charset=utf-8")
            if self.path == "/api/state":
                return self._send(200, json.dumps(hub.snapshot(), ensure_ascii=False))
            if self.path.startswith("/api/speech/"):
                sid = int(self.path.rsplit("/", 1)[1].split(".")[0])
                with hub.lock:
                    wav = hub.speech.get(sid)
                return self._send(200, wav, "audio/wav") if wav else self._send(404, "{}")
            if self.path.startswith("/stream.mjpg"):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                last = -1
                try:
                    while True:
                        with hub.cond:
                            hub.cond.wait_for(lambda: hub.frame_id != last, timeout=5)
                            jpeg, last = hub.jpeg, hub.frame_id
                        if jpeg is None:
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    return
            return self._send(404, "{}")

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            st = hub.snapshot()
            if self.path == "/api/instruct":
                text = (body.get("text") or "").strip()
                if not text:
                    return self._send(400, json.dumps({"error": "指示が空です"}, ensure_ascii=False))
                if st["busy"]:
                    return self._send(409, json.dumps({"error": "前の指示を実行中です"}, ensure_ascii=False))
                hub.commands.put({"type": "instruct", "text": text[:500], "model": body.get("model", "argon")})
                return self._send(200, "{}")
            if self.path == "/api/reset":
                if st["busy"]:
                    return self._send(409, json.dumps({"error": "実行中は初期化できません"}, ensure_ascii=False))
                hub.commands.put({"type": "reset", "scene": body.get("scene", "scene_demo")})
                return self._send(200, "{}")
            if self.path == "/api/record":
                hub.commands.put({"type": "record", "on": bool(body.get("on"))})
                return self._send(200, "{}")
            if self.path == "/api/speed":
                hub.playback = max(0.5, min(4.0, float(body.get("speed", 1.0))))
                return self._send(200, "{}")
            return self._send(404, "{}")

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--scene", default="scene_demo", choices=[s for s, _ in SCENES])
    args = ap.parse_args()

    from run import load_dotenv

    load_dotenv()
    hub = Hub()
    threading.Thread(target=worker, args=(hub, args), daemon=True).start()
    server = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(hub))
    server.daemon_threads = True
    print(f"Web UI を起動しました（コンテナ内のポート {args.port}）。手元の PC で "
          f"ssh -L {args.port}:localhost:{os.environ.get('WEBUI_PORT', args.port)} <ユーザー名>@<サーバー> を実行し、"
          f"ブラウザで http://localhost:{args.port} を開いてください。Ctrl+C で終了します。", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("終了します")
        sys.exit(0)


if __name__ == "__main__":
    main()
