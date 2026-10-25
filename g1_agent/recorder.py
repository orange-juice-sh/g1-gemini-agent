"""録画（字幕・小窓・ミニマップ・カード）と実時間ビューア"""
from __future__ import annotations

import math
import os
import shutil
import subprocess
import time

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import world

FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
    "/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc",
    "C:/Windows/Fonts/meiryo.ttc",
]
BOLD_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Bold.ttc",
]

MINIMAP_NAMES = {"充電ステーション": "充電", "エレベーター前": "EV", "休憩スペース": "休憩"}

NAVY = (14, 22, 40)
ACCENT = (56, 140, 235)
WHITE = (255, 255, 255)


def _font(size, bold=False):
    path = os.environ.get("G1_FONT")
    cands = ([path] if path else []) + (BOLD_CANDIDATES if bold else []) + FONT_CANDIDATES
    for p in cands:
        if p and os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


NO_LINE_START = set("。、，．）」』】〕！？ー・…")


def _wrap_text(draw, text, font, width):
    lines, cur = [], ""
    for ch in text:
        if ch == "\n":
            lines.append(cur)
            cur = ""
        elif draw.textlength(cur + ch, font=font) > width:
            if ch in NO_LINE_START and cur:
                lines.append(cur + ch)
                cur = ""
            else:
                lines.append(cur)
                cur = ch
        else:
            cur += ch
    if cur:
        lines.append(cur)
    return lines


def _pick_encoder():
    """NVENC が使えれば使い、だめなら libx264（G1_VIDEO_CODEC で固定可）"""
    forced = os.environ.get("G1_VIDEO_CODEC")
    system_ffmpeg = shutil.which("ffmpeg")
    if forced:
        codec = forced
    elif system_ffmpeg:
        probe = subprocess.run([system_ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                                "-i", "color=size=256x256:rate=30", "-t", "0.2", "-c:v", "h264_nvenc",
                                "-f", "null", "-"], capture_output=True)
        codec = "h264_nvenc" if probe.returncode == 0 else "libx264"
    else:
        codec = "libx264"
    # imageio 同梱の ffmpeg は NVENC 非対応
    if codec.endswith("nvenc") and system_ffmpeg:
        os.environ["IMAGEIO_FFMPEG_EXE"] = system_ffmpeg
    params = ["-preset", "p5", "-rc", "vbr", "-cq", "21", "-b:v", "0"] if codec.endswith("nvenc") \
        else ["-preset", "medium", "-crf", "20"]
    return codec, params


class Recorder:
    """シミュレーション時間に合わせてフレームを書き出す（speed=2 で2倍速）"""

    def __init__(self, sim, robot, path, size=(1920, 1080), fps=30, speed=1.0):
        import imageio.v2 as imageio

        self.sim, self.robot, self.path = sim, robot, path
        self.W, self.H = size
        self.s = self.H / 720
        self.fps, self.speed = fps, speed
        self.writer = None
        if path:
            self.codec, params = _pick_encoder()
            self.writer = imageio.get_writer(path, fps=fps, codec=self.codec, quality=None, macro_block_size=8,
                                             ffmpeg_params=params + ["-movflags", "+faststart"])
            print(f"   録画: {self.W}x{self.H} {fps}fps, エンコーダ {self.codec}", flush=True)
        self.main = mujoco.Renderer(sim.model, self.H, self.W)
        self.inset = mujoco.Renderer(sim.model, self.S(240), self.S(320))
        self.cam = mujoco.MjvCamera()
        self.cam.distance, self.cam.elevation = 4.3, -27
        self.front = mujoco.MjvCamera()
        self.front.distance, self.front.elevation = 2.9, -10
        self.over = mujoco.MjvCamera()
        r = world.ROOM
        self.over.lookat[:] = [(r["x_min"] + r["x_max"]) / 2 + 0.3, (r["y_min"] + r["y_max"]) / 2, 0.0]
        self.over.distance, self.over.elevation, self.over.azimuth = 15.5, -48, 22
        self._az = None
        self._look = None
        self.shot = "follow"
        self.next_t = sim.time
        self.instruction = ""
        self.chapter = ""
        self.caption = ""
        self.thinking = False
        self.speaker = "AI"
        self.found = []
        self.last_frame = None
        self.frames = 0
        self.tts = False
        self.speech_clips = []
        self.realtime = False
        self.f_title = _font(self.S(44), bold=True)
        self.f_big = _font(self.S(30), bold=True)
        self.f_mid = _font(self.S(24))
        self.f_small = _font(self.S(17))
        sim.frame_callbacks.append(self._on_step)

    def S(self, v):
        return int(round(v * self.s))

    def _on_step(self, sim):
        if sim.time + 1e-9 >= self.next_t:
            self.next_t += self.speed / self.fps
            self._write(self._compose())

    def _write(self, frame):
        self.last_frame = frame
        if self.writer is not None:
            self.writer.append_data(frame)
        self.frames += 1

    def hold(self, seconds, thinking=True):
        """シミュレーション時間が進まない区間（AI の応答待ち）を短く表示する"""
        self.thinking = thinking
        frame = self._compose()
        for _ in range(max(1, int(seconds * self.fps))):
            self._write(frame)
        self.thinking = False

    def _card(self, lines, seconds, kicker=None, footer=None):
        """俯瞰の映像を暗くして文字を出すカード
        lines は (文字, title|big|mid|small)"""
        base = self._render(self.over)
        img = Image.fromarray(base).convert("RGBA")
        img = Image.alpha_composite(img, Image.new("RGBA", img.size, NAVY + (215,))).convert("RGB")
        d = ImageDraw.Draw(img, "RGBA")
        fonts = {"title": self.f_title, "big": self.f_big, "mid": self.f_mid, "small": self.f_small}
        gaps = {"title": 78, "big": 56, "mid": 42, "small": 32}
        wrapped = []
        for text, kind in lines:
            for line in _wrap_text(d, text, fonts[kind], self.W - self.S(240)) or [""]:
                wrapped.append((line, kind))
        total = sum(self.S(gaps[k]) for _, k in wrapped)
        y = (self.H - total) / 2
        if kicker:
            w = d.textlength(kicker, font=self.f_mid) + self.S(36)
            d.rounded_rectangle([(self.W - w) / 2, y - self.S(70), (self.W + w) / 2, y - self.S(30)],
                                radius=self.S(20), fill=ACCENT + (255,))
            d.text(((self.W - w) / 2 + self.S(18), y - self.S(67)), kicker, font=self.f_mid, fill=WHITE)
        for text, kind in wrapped:
            f = fonts[kind]
            w = d.textlength(text, font=f)
            color = WHITE if kind in ("title", "big") else (215, 225, 240)
            d.text(((self.W - w) / 2, y), text, font=f, fill=color)
            y += self.S(gaps[kind])
        if footer:
            w = d.textlength(footer, font=self.f_small)
            d.text(((self.W - w) / 2, self.H - self.S(48)), footer, font=self.f_small, fill=(170, 180, 200))
        frame = np.asarray(img)
        for _ in range(int(seconds * self.fps)):
            self._write(frame)

    def title_card(self, lines, seconds=3.5):
        kinds = ["title"] + ["mid"] * (len(lines) - 1)
        self._card(list(zip(lines, kinds)), seconds)

    def architecture_card(self, speaker, seconds=5.0):
        base = self._render(self.over)
        img = Image.fromarray(base).convert("RGBA")
        img = Image.alpha_composite(img, Image.new("RGBA", img.size, NAVY + (225,))).convert("RGB")
        d = ImageDraw.Draw(img, "RGBA")
        S, W, H = self.S, self.W, self.H
        title = "しくみ：AI が考え、学習済みの歩行ポリシーが体を動かす"
        d.text(((W - d.textlength(title, font=self.f_big)) / 2, S(40)), title, font=self.f_big, fill=WHITE)
        layers = [
            (f"{speaker}（AI・頭脳）", "指示を手順に分け、ツールを呼び、結果と頭部カメラの画像を見て再計画", "約10秒に1回"),
            ("スキル層", "移動・見る・探す・指差す・ボタンを押す・持ち上げて運ぶ（腕は逆運動学）", "10 Hz"),
            ("安全層（AI から独立）", "仮想 LiDAR で進行方向の障害物を見張り、近すぎれば止める", "10 Hz"),
            ("歩行ポリシー（強化学習・Unitree 公式）", "速度の指令に従って脚を動かす", "50 Hz"),
            ("MuJoCo（物理シミュレーション）", "Unitree G1・ロビー・家具・人", "500 Hz"),
        ]
        bw, bh, gap = S(820), S(72), S(26)
        x0 = (W - bw) / 2
        y = S(120)
        colors = [ACCENT, (90, 110, 150), (200, 70, 70), (70, 140, 110), (110, 110, 120)]
        for i, (name, desc, rate) in enumerate(layers):
            d.rounded_rectangle([x0, y, x0 + bw, y + bh], radius=S(12), fill=colors[i] + (235,))
            d.text((x0 + S(18), y + S(6)), name, font=self.f_mid, fill=WHITE)
            d.text((x0 + S(18), y + S(38)), desc, font=self.f_small, fill=(235, 240, 248))
            d.text((x0 + bw - S(18) - d.textlength(rate, font=self.f_small), y + S(8)), rate, font=self.f_small,
                   fill=(235, 240, 248))
            if i < len(layers) - 1:
                cx = W / 2
                d.polygon([(cx - S(10), y + bh + S(4)), (cx + S(10), y + bh + S(4)), (cx, y + bh + gap - S(4))],
                          fill=(200, 210, 225))
            y += bh + gap
        frame = np.asarray(img)
        for _ in range(int(seconds * self.fps)):
            self._write(frame)

    def chapter_card(self, index, total, title, instruction, seconds=3.0):
        self.chapter = f"{index}/{total} {title}" if title else ""
        self._card([(title or "指示", "big"), ("「" + instruction + "」", "mid")], seconds,
                   kicker=f"シーン {index} / {total}")

    def end_card(self, title, lines, seconds=6.0, footer=None):
        self._card([(title, "big")] + [(l, "mid") for l in lines], seconds, footer=footer)

    def speak_begin(self, text: str) -> float | None:
        """発話の音声を作り、終わるまで等倍で録画する
        戻り値は音声の長さ[s]（なしなら None）"""
        if not self.tts:
            return None
        import tempfile

        from . import tts

        wav, dur = tts.synthesize(text)
        path = os.path.join(tempfile.gettempdir(), f"g1_say_{os.getpid()}_{len(self.speech_clips)}.wav")
        open(path, "wb").write(wav)
        self.speech_clips.append((self.frames / self.fps, path))
        self._saved_speed, self.speed, self.realtime = self.speed, 1.0, True
        return dur

    def speak_end(self):
        if self.realtime:
            self.speed, self.realtime = self._saved_speed, False

    def _mux_audio(self):
        """発話の音声を動画に重ねる（映像は再エンコードしない）"""
        ffmpeg = shutil.which("ffmpeg") or os.environ.get("IMAGEIO_FFMPEG_EXE")
        tmp = self.path + ".audio.mp4"
        cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", self.path]
        filters = []
        for i, (t, wav) in enumerate(self.speech_clips):
            cmd += ["-i", wav]
            ms = int(t * 1000)
            filters.append(f"[{i + 1}:a]adelay={ms}|{ms}[a{i}]")
        mix = "".join(f"[a{i}]" for i in range(len(self.speech_clips)))
        # -shortest だけだと最後の発話で映像が切れるので無音で埋める
        filters.append(f"{mix}amix=inputs={len(self.speech_clips)}:normalize=0,apad[aout]")
        cmd += ["-filter_complex", ";".join(filters), "-map", "0:v", "-map", "[aout]", "-c:v", "copy",
                "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", tmp]
        subprocess.run(cmd, check=True)
        os.replace(self.path, self.path.replace(".mp4", "") + "_noaudio.mp4")
        os.replace(tmp, self.path)

    def close(self):
        try:
            self.sim.frame_callbacks.remove(self._on_step)
        except ValueError:
            pass
        if self.writer is not None:
            self.writer.close()
        if self.speech_clips and self.writer is not None:
            self._mux_audio()
        # 終了時の EGL エラーを避けるため明示的に解放する
        for r in (self.main, self.inset):
            r.close()
        return self.frames / self.fps

    def _robot_visible(self, look, az_deg, el_deg, dist):
        m, d = self.sim.model, self.sim.data
        if not hasattr(self, "_robot_bodies"):
            root = m.body("pelvis").id
            self._robot_bodies = {i for i in range(m.nbody) if m.body_rootid[i] == root}
        az, el = math.radians(az_deg), math.radians(el_deg)
        cam = np.asarray(look) - dist * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        x, y, _ = self.sim.pose()
        gid = np.zeros(1, dtype=np.int32)
        groups = np.array([1, 1, 1, 1, 0, 0], dtype=np.uint8)
        for z in (0.6, 0.9, 1.15):
            vec = np.array([x, y, z]) - cam
            mujoco.mj_ray(m, d, cam, vec / np.linalg.norm(vec), groups, 1, -1, gid)
            if gid[0] >= 0 and m.geom_bodyid[gid[0]] not in self._robot_bodies:
                return False
        return True

    def _update_follow(self):
        x, y, yaw = self.sim.pose()
        if self._az is None:
            self._az = math.degrees(yaw) + 35
        look = np.array([x, y, 0.7])
        target, dist = math.degrees(yaw) + 35, 4.3
        if not self._robot_visible(look, self._az, self.cam.elevation, self.cam.distance):
            for dd in (4.3, 3.2, 2.4):
                cand = [off for off in (35, 75, -5, 115, -45, 160, -100, 200) if
                        self._robot_visible(look, math.degrees(yaw) + off, self.cam.elevation, dd)]
                if cand:
                    target, dist = math.degrees(yaw) + cand[0], dd
                    self._follow_override = (target - math.degrees(yaw), dd)
                    break
        elif getattr(self, "_follow_override", None):
            off, dd = self._follow_override
            if self._robot_visible(look, math.degrees(yaw) + off, self.cam.elevation, dd):
                target, dist = math.degrees(yaw) + off, dd
            else:
                self._follow_override = None
        diff = (target - self._az + 180) % 360 - 180
        self._az += 0.08 * diff
        self.cam.distance += 0.08 * (dist - self.cam.distance)
        self.cam.azimuth = self._az
        if self._look is None:
            self._look = np.array([x, y, 0.7])
        self._look += 0.15 * (np.array([x, y, 0.7]) - self._look)
        self.cam.lookat[:] = self._look

    def _clear_azimuth(self, x, y, yaw):
        """正面カメラの方位角
        人や家具にさえぎられるときは見える角度に回り込む"""
        m, d = self.sim.model, self.sim.data
        if not hasattr(self, "_robot_bodies"):
            root = m.body("pelvis").id
            self._robot_bodies = {i for i in range(m.nbody) if m.body_rootid[i] == root}
        targets = [np.array([x, y, z]) for z in (0.6, 0.9, 1.15)]
        gid = np.zeros(1, dtype=np.int32)
        # 見た目だけの部品には光線が当たらないので、group 3 の当たり判定で調べる
        groups = np.array([1, 1, 1, 1, 0, 0], dtype=np.uint8)
        el = math.radians(self.front.elevation)
        for off in (155, 115, 195, 75, 235, 90, 270):
            az = math.radians(math.degrees(yaw) + off)
            fwd = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
            cam = np.array([x, y, 0.95]) - self.front.distance * fwd
            clear = True
            for t in targets:
                vec = t - cam
                mujoco.mj_ray(m, d, cam, vec / np.linalg.norm(vec), groups, 1, -1, gid)
                if gid[0] >= 0 and m.geom_bodyid[gid[0]] not in self._robot_bodies:
                    clear = False
                    break
            if clear:
                return math.degrees(yaw) + off
        return math.degrees(yaw) + 155

    def _render(self, camera, renderer=None):
        r = renderer or self.main
        r.update_scene(self.sim.data, camera=camera)
        return r.render()

    def _compose(self):
        self._update_follow()
        if self.shot == "front":
            x, y, yaw = self.sim.pose()
            self.front.lookat[:] = [x, y, 0.95]
            self.front.azimuth = self._clear_azimuth(x, y, yaw)
        if self.shot == "head":
            main_cam, inset_cam, inset_label = "head_cam", self.cam, "三人称視点"
        elif self.shot == "front":
            main_cam, inset_cam, inset_label = self.front, "head_cam", "頭部カメラ"
        elif self.shot == "overview":
            main_cam, inset_cam, inset_label = self.over, "head_cam", "頭部カメラ"
        else:
            main_cam, inset_cam, inset_label = self.cam, "head_cam", "頭部カメラ"
        img = Image.fromarray(self._render(main_cam))
        d = ImageDraw.Draw(img, "RGBA")
        W, H, S = self.W, self.H, self.S

        top_h = 0
        if self.instruction:
            lines = _wrap_text(d, "指示: " + self.instruction, self.f_mid, W - S(420))[:2]
            top_h = S(16) + S(34) * len(lines)
            d.rectangle([0, 0, W, top_h], fill=NAVY + (200,))
            for i, line in enumerate(lines):
                d.text((S(20), S(8) + S(34) * i), line, font=self.f_mid, fill=WHITE)

        iw, ih = S(320), S(240)
        ix, iy = W - iw - S(16), top_h + S(16)
        img.paste(Image.fromarray(self._render(inset_cam, self.inset)), (ix, iy))
        d.rectangle([ix - 2, iy - 2, ix + iw + 1, iy + ih + 1], outline=(255, 255, 255, 230), width=max(2, S(2)))
        lw = d.textlength(inset_label, font=self.f_small) + S(14)
        d.rectangle([ix, iy, ix + lw, iy + S(24)], fill=(0, 0, 0, 160))
        d.text((ix + S(7), iy + S(1)), inset_label, font=self.f_small, fill=WHITE)
        if self.shot == "head":
            d.rounded_rectangle([S(20), H - S(170), S(20) + d.textlength("頭部カメラの映像", font=self.f_mid) + S(28),
                                 H - S(130)], radius=S(8), fill=(0, 0, 0, 170))
            d.text((S(34), H - S(167)), "頭部カメラの映像", font=self.f_mid, fill=WHITE)

        self._minimap(d, (W - S(336), H - S(330), W - S(16), H - S(110)))

        if self.caption:
            lines = _wrap_text(d, self.caption, self.f_mid, W - S(400))[-3:]
            top = H - S(20) - S(36) * len(lines)
            d.rectangle([0, top - S(12), W - S(352), H], fill=(0, 0, 0, 175))
            for i, line in enumerate(lines):
                d.text((S(22), top + S(36) * i), line, font=self.f_mid, fill=WHITE)

        if self.robot.speech:
            text = "G1「" + self.robot.speech[0] + "」"
            lines = _wrap_text(d, text, self.f_big, S(700))[:4]
            bw = max(d.textlength(l, font=self.f_big) for l in lines) + S(44)
            bx, by = (W - S(352) - bw) / 2, top_h + S(70)
            d.rounded_rectangle([bx, by, bx + bw, by + S(24) + S(44) * len(lines)], radius=S(20),
                                fill=(255, 255, 255, 240), outline=ACCENT + (255,), width=S(4))
            for i, l in enumerate(lines):
                d.text((bx + S(22), by + S(10) + S(44) * i), l, font=self.f_big, fill=(20, 30, 50))

        badges = []
        if self.chapter:
            badges.append((self.chapter, ACCENT + (235,)))
        if not getattr(self, "hide_speed_badge", False):
            if self.realtime:
                badges.append(("等倍（音声）", (60, 60, 60, 210)))
            elif self.speed != 1.0:
                badges.append((f"{self.speed:g}倍速", (60, 60, 60, 210)))
        if self.thinking:
            badges.append((f"{self.speaker} 思考中…", (120, 70, 200, 235)))
        if self.sim.state.safety_active:
            badges.append(("安全停止（障害物）", (210, 40, 40, 240)))
        if self.sim.state.fallen:
            badges.append(("転倒検知", (230, 120, 0, 240)))
        bx, top = S(20), top_h + S(12)
        for text, color in badges:
            w = d.textlength(text, font=self.f_small) + S(20)
            d.rounded_rectangle([bx, top, bx + w, top + S(30)], radius=S(8), fill=color)
            d.text((bx + S(10), top + S(4)), text, font=self.f_small, fill=WHITE)
            bx += w + S(8)
        return np.asarray(img)

    def _minimap(self, d, box):
        x0, y0, x1, y1 = box
        S = self.S
        r = world.ROOM
        d.rounded_rectangle(box, radius=S(8), fill=(250, 250, 250, 235), outline=(80, 80, 80, 255), width=2)
        s = min((x1 - x0 - S(20)) / (r["x_max"] - r["x_min"]), (y1 - y0 - S(20)) / (r["y_max"] - r["y_min"]))

        def P(x, y):
            return (x0 + S(10) + (x - r["x_min"]) * s, y1 - S(10) - (y - r["y_min"]) * s)

        for p in self.sim.scene.props:
            pos = self.sim.prop_pos(p.name)
            if not p.obstacle or pos[2] < -1:
                continue
            cx, cy = P(pos[0], pos[1])
            if p.shape == "box":
                hx, hy = p.size[0] * s, p.size[1] * s
                d.rectangle([cx - hx, cy - hy, cx + hx, cy + hy], fill=(150, 150, 150, 220))
            else:
                rr = p.size[0] * s
                d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=(150, 150, 150, 220))
        placed = []
        for loc in world.visible_locations(self.sim.scene):
            cx, cy = P(loc.x, loc.y)
            c = tuple(int(v * 255) for v in loc.color[:3])
            d.ellipse([cx - S(8), cy - S(8), cx + S(8), cy + S(8)], fill=c + (230,), outline=(90, 90, 90, 255))
            label = MINIMAP_NAMES.get(loc.name, loc.name)
            tw, th = d.textlength(label, font=self.f_small), S(20)
            cands = [(cx + S(11), cy - th / 2), (cx - S(11) - tw, cy - th / 2),
                     (cx - tw / 2, cy + S(10)), (cx - tw / 2, cy - S(10) - th)]
            for lx, ly in cands:
                rect = (lx, ly, lx + tw, ly + th)
                inside = rect[0] >= x0 + 2 and rect[2] <= x1 - 2 and rect[1] >= y0 + S(22) and rect[3] <= y1 - 2
                overlap = any(not (rect[2] < q[0] or rect[0] > q[2] or rect[3] < q[1] or rect[1] > q[3]) for q in placed)
                if inside and not overlap:
                    break
            else:
                lx = min(max(cands[0][0], x0 + 4), x1 - 4 - tw)
                ly = min(max(cands[0][1], y0 + S(22)), y1 - 4 - th)
                rect = (lx, ly, lx + tw, ly + th)
            placed.append(rect)
            d.text((lx, ly - S(2)), label, font=self.f_small, fill=(30, 30, 30))
        colors = {"red": (220, 30, 30), "blue": (30, 70, 230), "green": (30, 170, 60), "point": (150, 60, 200)}
        for fx, fy, color in self.found:
            cx, cy = P(fx, fy)
            c = colors.get(color, (0, 0, 0))
            d.line([cx - S(7), cy - S(7), cx + S(7), cy + S(7)], fill=c, width=S(3))
            d.line([cx - S(7), cy + S(7), cx + S(7), cy - S(7)], fill=c, width=S(3))
        pts = [P(x, y) for x, y in self.sim.trajectory[-400:]]
        if len(pts) > 1:
            d.line(pts, fill=ACCENT + (255,), width=S(3))
        x, y, yaw = self.sim.pose()
        cx, cy = P(x, y)
        tip = (cx + S(14) * math.cos(yaw), cy - S(14) * math.sin(yaw))
        left = (cx + S(8) * math.cos(yaw + 2.4), cy - S(8) * math.sin(yaw + 2.4))
        right = (cx + S(8) * math.cos(yaw - 2.4), cy - S(8) * math.sin(yaw - 2.4))
        d.polygon([tip, left, right], fill=(240, 120, 0), outline=(0, 0, 0))
        d.text((x0 + S(8), y0 + S(3)), "ミニマップ", font=self.f_small, fill=(60, 60, 60))


class LiveViewer:
    """MuJoCo の標準ビューアで実時間表示する"""

    def __init__(self, sim, realtime=True):
        import mujoco.viewer

        self.sim = sim
        self.realtime = realtime
        self.viewer = mujoco.viewer.launch_passive(sim.model, sim.data)
        self.viewer.cam.distance, self.viewer.cam.elevation = 5.0, -25
        self._wall0, self._sim0 = time.time(), sim.time
        sim.frame_callbacks.append(self._on_step)

    def resync(self):
        self._wall0, self._sim0 = time.time(), self.sim.time

    def _on_step(self, sim):
        if sim.counter % 8:
            return
        self.viewer.cam.lookat[:] = [sim.data.qpos[0], sim.data.qpos[1], 0.7]
        self.viewer.sync()
        if self.realtime:
            ahead = (sim.time - self._sim0) - (time.time() - self._wall0)
            if ahead > 0:
                time.sleep(ahead)

    @property
    def running(self):
        return self.viewer.is_running()
