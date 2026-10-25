"""AI が呼ぶスキル（移動・視覚・腕・発話）
速度指令や腕の目標に変換して下位の制御に渡す
戻り値は成功・失敗の理由を含む dict（再計画のため）"""
from __future__ import annotations

import io
import math
import os

import mujoco
import numpy as np
from PIL import Image

from . import world
from .sim import G1Sim

CTRL_DT = 0.1
BLOCKED_TIMEOUT = 1.5

COLOR_RULES = {
    "red": lambda r, g, b: (r > 120) & (r > 2.0 * g) & (r > 2.0 * b),
    "blue": lambda r, g, b: (b > 120) & (b > 1.8 * r) & (b > 1.3 * g),
    "green": lambda r, g, b: (g > 110) & (g > 1.6 * r) & (g > 1.4 * b),
    "yellow": lambda r, g, b: (r > 100) & (g > 0.65 * r) & (b < 0.3 * r),
}
COLOR_JA = {"red": "赤い", "blue": "青い", "green": "緑の", "yellow": "黄色い"}


def _largest_blob(mask: np.ndarray) -> np.ndarray:
    small = mask[::4, ::4]
    labels = np.zeros(small.shape, dtype=np.int32)
    best, best_n, cur = 0, 0, 0
    for sy, sx in zip(*np.nonzero(small)):
        if labels[sy, sx]:
            continue
        cur += 1
        stack, n = [(sy, sx)], 0
        labels[sy, sx] = cur
        while stack:
            y, x = stack.pop()
            n += 1
            for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if 0 <= ny < small.shape[0] and 0 <= nx < small.shape[1] and small[ny, nx] and not labels[ny, nx]:
                    labels[ny, nx] = cur
                    stack.append((ny, nx))
        if n > best_n:
            best, best_n = cur, n
    keep = np.kron(labels == best, np.ones((4, 4), dtype=bool))[:mask.shape[0], :mask.shape[1]]
    return mask & keep


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Robot:
    def __init__(self, sim: G1Sim, cam_size=(480, 640)):
        self.sim = sim
        self.cam_h, self.cam_w = cam_size
        self._rgb = None
        self._depth = None
        self.last_view = None
        self.speech = None
        self.log = []
        # 安全停止したら自動で横に避ける（ライブモードと Web UI のみで、評価では使わない）
        self.auto_detour = False
        self.detours = 0

    def _renderers(self):
        if self._rgb is None:
            self._rgb = mujoco.Renderer(self.sim.model, self.cam_h, self.cam_w)
            self._depth = mujoco.Renderer(self.sim.model, self.cam_h, self.cam_w)
            self._depth.enable_depth_rendering()
        return self._rgb, self._depth

    def close(self):
        for r in (self._rgb, self._depth):
            if r is not None:
                r.close()
        self._rgb = self._depth = None

    def _result(self, ok: bool, message: str, **extra):
        x, y, yaw = self.sim.pose()
        res = {"success": ok, "message": message,
               "position": {"x": round(x, 2), "y": round(y, 2)}, "heading_deg": round(math.degrees(yaw)),
               **extra}
        if self.sim.state.fallen:
            res["fallen"] = True
            res["message"] += "（転倒を検知。以降の移動は危険です）"
        return res

    def idle(self, seconds: float):
        self.sim.step((0, 0, 0), seconds)

    def nearest_location(self):
        x, y, _ = self.sim.pose()
        loc = min(world.visible_locations(self.sim.scene), key=lambda l: math.hypot(l.x - x, l.y - y))
        return loc.name, math.hypot(loc.x - x, loc.y - y)

    def get_status(self):
        name, dist = self.nearest_location()
        where = f"{name}にいる" if dist < 0.8 else f"{name}から {dist:.1f}m の位置"
        return self._result(True, where, safety_stops_total=self.sim.state.safety_stops,
                            sim_time_s=round(self.sim.time, 1))

    def scan_surroundings(self):
        scan = self.sim.lidar()
        names = ["前", "左前", "左", "左後ろ", "後ろ", "右後ろ", "右", "右前"]
        sectors = {}
        for i, n in enumerate(names):
            center = math.radians(45 * i)
            diff = np.abs(_wrap(self.sim.lidar_angles - center))
            d = float(scan[diff <= math.radians(22.5)].min())
            sectors[n] = round(d, 2) if d < self.sim.lidar_range else f"{self.sim.lidar_range}m以上"
        return self._result(True, "各方向の最も近い障害物までの距離 [m]（高さ25cmの仮想LiDAR）", obstacles=sectors)

    def _drive(self, cmd_fn, done_fn, timeout):
        self.rest_arms()
        t0, blocked_since = self.sim.time, None
        while True:
            if done_fn():
                return "done"
            if self.sim.state.fallen:
                return "fallen"
            if self.sim.time - t0 > timeout:
                return "timeout"
            self.sim.step(cmd_fn(), CTRL_DT)
            if self.sim.state.safety_active:
                blocked_since = blocked_since if blocked_since is not None else self.sim.time
                if self.sim.time - blocked_since > BLOCKED_TIMEOUT:
                    return "blocked"
            else:
                blocked_since = None

    def _finish(self, status, goal_desc, done_msg=None):
        self.idle(0.4)
        if status == "done":
            return self._result(True, done_msg or f"{goal_desc}に到着しました")
        if status == "blocked":
            obstacles = self.scan_surroundings()["obstacles"]
            return self._result(False, f"{goal_desc}へ向かう途中、進行方向の障害物で安全停止しました。"
                                       "左右どちらかに避けてから再度向かってください。", obstacles=obstacles)
        if status == "fallen":
            return self._result(False, "転倒しました。")
        return self._result(False, f"{goal_desc}へ時間内に到着できませんでした。")

    def walk_to(self, location: str | None = None, x: float | None = None, y: float | None = None,
                stop_radius: float = 0.35, timeout: float = 45.0):
        if location:
            loc = world.location_by_name(location)
            if loc is None:
                names = "、".join(l.name for l in world.visible_locations(self.sim.scene))
                return self._result(False, f"「{location}」という場所は地図にありません。候補: {names}")
            gx, gy, desc = loc.x, loc.y, loc.name
        elif x is not None and y is not None:
            gx, gy, desc = float(x), float(y), f"座標({x:.1f}, {y:.1f})"
        else:
            return self._result(False, "location か x,y のどちらかを指定してください")

        def cmd():
            px, py, yaw = self.sim.pose()
            dist = math.hypot(gx - px, gy - py)
            err = _wrap(math.atan2(gy - py, gx - px) - yaw)
            wz = float(np.clip(1.6 * err, -0.8, 0.8))
            if abs(err) > 0.6:
                return (0.0, 0.0, wz)
            vx = float(np.clip(0.7 * dist, 0.15, 0.6)) * math.cos(err)
            return (vx, 0.0, wz)

        def done():
            px, py, _ = self.sim.pose()
            return math.hypot(gx - px, gy - py) < stop_radius

        def near_goal_stop():
            """目的地の手前の障害物（棚・カウンターなど）で止まったか"""
            px, py, _ = self.sim.pose()
            return math.hypot(gx - px, gy - py) <= 1.2

        status = self._drive(cmd, done, timeout)
        if status == "blocked" and near_goal_stop():
            self.idle(0.3)
            px, py, _ = self.sim.pose()
            return self._result(True, f"{desc}のすぐ手前に家具などがあるため、安全のため約{math.hypot(gx - px, gy - py):.1f}m 手前で"
                                      "止まりました（目的地に着いたものとして扱います）。棚や台の上の物は pick_up、"
                                      "台に置くときは put_down、ボタンは press_button が、自分でさらに近づいて行います")
        side, note = None, ""
        for step in (1.0, 1.3, 1.6):
            if status != "blocked" or not self.auto_detour:
                break
            obs = self.scan_surroundings()["obstacles"]
            dist = lambda k: obs[k] if isinstance(obs[k], (int, float)) else self.sim.lidar_range
            if side is None:
                side = "left" if dist("左前") + dist("左") >= dist("右前") + dist("右") else "right"
            moved = self.walk(side, step)
            if not moved["success"]:
                break
            self.detours += 1
            note = f"（途中の障害物を{'左' if side == 'left' else '右'}に自動で避けました）"
            status = self._drive(cmd, done, timeout)
        res = self._finish(status, desc)
        if note and res["success"]:
            res["message"] += note
        return res

    def walk(self, direction: str, distance_m: float):
        """向きを保ったまま前後左右に歩く"""
        dirs = {"forward": (1, 0), "backward": (-1, 0), "left": (0, 1), "right": (0, -1),
                "前": (1, 0), "後ろ": (-1, 0), "左": (0, 1), "右": (0, -1)}
        if direction not in dirs:
            return self._result(False, f"direction は {list(dirs)[:4]} のいずれかです")
        distance_m = float(np.clip(distance_m, 0.0, 5.0))
        x0, y0, yaw0 = self.sim.pose()
        ux, uy = dirs[direction]
        speed = 0.45 if ux else 0.3

        def cmd():
            px, py, yaw = self.sim.pose()
            # 進行方向と直角のずれを打ち消す
            dx, dy = px - x0, py - y0
            fwd = math.cos(yaw0) * dx + math.sin(yaw0) * dy
            side = -math.sin(yaw0) * dx + math.cos(yaw0) * dy
            vx = ux * speed if ux else float(np.clip(-2.0 * fwd, -0.3, 0.3))
            vy = uy * speed if uy else float(np.clip(-2.0 * side, -0.3, 0.3))
            return (vx, vy, float(np.clip(1.5 * _wrap(yaw0 - yaw), -0.5, 0.5)))

        def done():
            px, py, _ = self.sim.pose()
            return math.hypot(px - x0, py - y0) >= distance_m - 0.05

        label = {"forward": "前", "backward": "後ろ", "left": "左", "right": "右"}.get(direction, direction)
        desc = f"{label}へ{distance_m:.1f}m"
        return self._finish(self._drive(cmd, done, 6 + distance_m / speed * 2), desc, f"{desc}移動しました")

    def turn(self, angle_deg: float):
        """その場で旋回する（正で左）"""
        _, _, yaw0 = self.sim.pose()
        target = _wrap(yaw0 + math.radians(float(angle_deg)))

        def cmd():
            _, _, yaw = self.sim.pose()
            return (0.0, 0.0, float(np.clip(2.0 * _wrap(target - yaw), -0.8, 0.8)))

        def done():
            _, _, yaw = self.sim.pose()
            return abs(_wrap(target - yaw)) < math.radians(4)

        st = self._drive(cmd, done, 4 + abs(angle_deg) / 30)
        self.idle(0.3)
        return self._result(st == "done", f"{angle_deg:+.0f}度旋回しました" if st == "done" else "旋回できませんでした")

    def face(self, x: float, y: float):
        px, py, yaw = self.sim.pose()
        return self.turn(math.degrees(_wrap(math.atan2(y - py, x - px) - yaw)))

    def _capture(self):
        rgb_r, depth_r = self._renderers()
        rgb_r.update_scene(self.sim.data, camera="head_cam")
        rgb = rgb_r.render().copy()
        depth_r.update_scene(self.sim.data, camera="head_cam")
        depth = depth_r.render().copy()
        self.last_view = rgb
        cid = self.sim.model.camera("head_cam").id
        self._last_capture = (depth, self.sim.data.cam_xpos[cid].copy(), self.sim.data.cam_xmat[cid].reshape(3, 3).copy())
        return rgb, depth

    def look(self):
        rgb, _ = self._capture()
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="JPEG", quality=85)
        return self._result(True, f"頭部カメラの画像です（視野角は横約{self._hfov_deg():.0f}度）"), buf.getvalue()

    def _hfov_deg(self):
        fovy = math.radians(self.sim.model.cam_fovy[self.sim.model.camera("head_cam").id])
        return math.degrees(2 * math.atan(math.tan(fovy / 2) * self.cam_w / self.cam_h))

    def find_object(self, color: str):
        """色で物を検出し、深度から位置を推定する"""
        if color not in COLOR_RULES:
            return self._result(False, f"color は {list(COLOR_RULES)} のいずれかです"), None
        rgb, depth = self._capture()
        r, g, b = (rgb[..., i].astype(np.int32) for i in range(3))
        mask = COLOR_RULES[color](r, g, b)
        if mask.sum() < 30:
            return self._result(False, f"{COLOR_JA[color]}物体は今の視野には見つかりません。"
                                       "turn で向きを変えるか、別の場所へ移動して探してください。", found=False), None
        mask = _largest_blob(mask)
        # 塊の画素を深度から3次元に戻し、その中央値を位置とする
        ys, xs = np.nonzero(mask)
        pts = self._pixels_to_world(xs, ys, depth[ys, xs])
        ox, oy = float(np.median(pts[:, 0])), float(np.median(pts[:, 1]))
        px, py, yaw = self.sim.pose()
        horiz = math.hypot(ox - px, oy - py)
        bearing = _wrap(math.atan2(oy - py, ox - px) - yaw)
        info = {"found": True, "bearing_deg": round(math.degrees(bearing), 1),
                "distance_m": round(horiz, 2), "estimated_position": {"x": round(ox, 2), "y": round(oy, 2)}}
        side = "左" if bearing > 0.05 else "右" if bearing < -0.05 else "正面"
        return self._result(True, f"{COLOR_JA[color]}物体を{side}方向 約{horiz:.1f}m に検出しました", **info), (ox, oy)

    def _pixels_to_world(self, us, vs, zs, cam=None):
        if cam is None:
            cid = self.sim.model.camera("head_cam").id
            cam = (self.sim.data.cam_xpos[cid], self.sim.data.cam_xmat[cid].reshape(3, 3))
        fovy = math.radians(self.sim.model.cam_fovy[self.sim.model.camera("head_cam").id])
        f = (self.cam_h / 2) / math.tan(fovy / 2)
        us, vs, zs = (np.atleast_1d(np.asarray(a, dtype=float)) for a in (us, vs, zs))
        p_cam = np.stack([(us - self.cam_w / 2) / f, -(vs - self.cam_h / 2) / f, -np.ones_like(us)], 1) * zs[:, None]
        return cam[0] + p_cam @ cam[1].T  # カメラ座標系（-z が視線）→ 世界座標

    def locate_point(self, x: float, y: float, stop_distance: float = 0.9):
        """最後に look した画像の点（0〜1000 の座標）の床の上の位置を求める"""
        if getattr(self, "_last_capture", None) is None:
            return self._result(False, "先に look で画像を取得してください")
        depth, cam_pos, cam_mat = self._last_capture
        u = float(np.clip(x, 0, 1000)) / 1000 * (self.cam_w - 1)
        v = float(np.clip(y, 0, 1000)) / 1000 * (self.cam_h - 1)
        ui, vi = int(round(u)), int(round(v))
        patch = depth[max(0, vi - 3):vi + 4, max(0, ui - 3):ui + 4]
        z = float(np.median(patch))  # 深度は光軸方向の距離
        if not np.isfinite(z) or z > 15:
            return self._result(False, "その点までの距離を測れませんでした（遠すぎるか、空や壁の上の方を指しています）")
        p = self._pixels_to_world(u, v, z, (cam_pos, cam_mat))[0]
        r = world.ROOM
        if not (r["x_min"] < p[0] < r["x_max"] and r["y_min"] < p[1] < r["y_max"]):
            return self._result(False, "指定した点は部屋の外（壁の向こう）です。部屋の中の人や物の足元を指してください")
        px, py, _ = self.sim.pose()
        d = math.hypot(p[0] - px, p[1] - py)
        k = max(0.0, d - stop_distance) / max(d, 1e-6)
        stand = (float(px + (p[0] - px) * k), float(py + (p[1] - py) * k))
        return self._result(True, f"指定した点は約{d:.1f}m 先、高さ{p[2]:.1f}m にあります。"
                                  f"近づくには walk_to で approach_position へ移動してください",
                            point_position={"x": round(float(p[0]), 2), "y": round(float(p[1]), 2),
                                            "height_m": round(float(p[2]), 2)},
                            distance_m=round(d, 2),
                            approach_position={"x": round(stand[0], 2), "y": round(stand[1], 2)})

    def approach_object(self, color: str, stop_distance: float = 0.9):
        res, pos = self.find_object(color)
        if pos is None:
            return res
        px, py, _ = self.sim.pose()
        ox, oy = pos
        d = math.hypot(ox - px, oy - py)
        if d > stop_distance:
            gx = ox - (ox - px) / d * stop_distance
            gy = oy - (oy - py) / d * stop_distance
            moved = self.walk_to(x=gx, y=gy, stop_radius=0.2)
            if not moved["success"]:
                return moved
        self.face(ox, oy)
        return self._result(True, f"{COLOR_JA[color]}物体の前（約{stop_distance:.1f}m手前）に到着し、物体の方を向きました",
                            object_position=res["estimated_position"])

    def rest_arms(self, duration: float = 0.6):
        if self.sim.held is not None:
            return
        if self.sim.arms and np.abs(self.sim.arm_target).max() > 0.05:
            self.sim.move_arms({n: 0.0 for n in self.sim.upper_names}, duration)
            self.idle(duration + 0.1)

    def _target_xy(self, location=None, x=None, y=None):
        if location:
            loc = world.location_by_name(location)
            if loc is None:
                return None, f"「{location}」という場所は地図にありません"
            return (loc.x, loc.y), loc.name
        if x is not None and y is not None:
            return (float(x), float(y)), f"座標({float(x):.1f}, {float(y):.1f})"
        return None, "location か x,y を指定してください"

    def gesture(self, kind: str, location: str | None = None, x: float | None = None, y: float | None = None):
        """ジェスチャー（wave / point / welcome / rest）
        次に移動するまで姿勢を保つ"""
        if not self.sim.arms:
            return self._result(False, "このロボットのモデルには動かせる腕がありません")
        if kind == "rest":
            self.rest_arms()
            return self._result(True, "腕を下ろしました")
        if kind == "wave":
            up = {"right_shoulder_pitch_joint": -2.5, "right_shoulder_roll_joint": -0.45,
                  "right_shoulder_yaw_joint": 0.0, "right_elbow_joint": 0.25}
            self.sim.move_arms(up, 0.7)
            self.idle(0.8)
            for roll in (-0.15, -0.75, -0.15, -0.75, -0.45):
                self.sim.move_arms({"right_shoulder_roll_joint": roll}, 0.32)
                self.idle(0.34)
            return self._result(True, "手を振りました")
        if kind == "welcome":
            px, py, yaw = self.sim.pose()
            c, s_ = math.cos(yaw), math.sin(yaw)
            sol = {}
            for side, sign in (("left", 1), ("right", -1)):
                fx, fy, z = 0.30, 0.42 * sign, 1.0
                sol.update(self.sim.solve_arm_ik(side, (px + c * fx - s_ * fy, py + s_ * fx + c * fy, z))[0])
            self.sim.move_arms(sol, 0.8)
            self.idle(1.0)
            return self._result(True, "両手を広げて迎える姿勢をとりました")
        if kind == "point":
            target, desc = self._target_xy(location, x, y)
            if target is None:
                return self._result(False, f"指差す先が分かりません: {desc}")
            px, py, yaw = self.sim.pose()
            bearing = _wrap(math.atan2(target[1] - py, target[0] - px) - yaw)
            if abs(bearing) > math.radians(110):
                return self._result(False, f"{desc}は体の後ろ側（{math.degrees(bearing):+.0f}度）にあり指差せません。"
                                           "turn で体を向けてから指差してください")
            side = "left" if bearing > math.radians(10) else "right"
            shoulder = self.sim.data.xpos[self.sim.model.body(f"{side}_shoulder_pitch_link").id].copy()
            d = np.array([target[0], target[1], shoulder[2] - 0.05]) - shoulder
            hand_goal = shoulder + 0.6 * d / np.linalg.norm(d)
            sol, miss = self.sim.solve_arm_ik(side, hand_goal)
            self.rest_arms(0.4)
            self.sim.move_arms(sol, 0.8)
            self.idle(1.0)
            tip = self.sim.site_pos(f"{side}_hand_tip")
            shoulder = self.sim.data.xpos[self.sim.model.body(f"{side}_shoulder_pitch_link").id]
            arm_dir = math.atan2(tip[1] - shoulder[1], tip[0] - shoulder[0])
            want = math.atan2(target[1] - shoulder[1], target[0] - shoulder[0])
            err = abs(math.degrees(_wrap(arm_dir - want)))
            self.pointing = {"target": target, "error_deg": err}
            hand = "左手" if side == "left" else "右手"
            return self._result(True, f"{hand}で{desc}の方向を指差しています（向きのずれ {err:.0f}度）",
                                pointing_error_deg=round(err, 1))
        return self._result(False, f"kind は wave / point / welcome / rest のいずれかです（{kind} は未対応）")

    def _body_frame(self):
        x, y, yaw = self.sim.pose()
        fwd = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        left = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
        return np.array([x, y, 0.0]), fwd, left

    def _hands_to(self, center, half_width, duration=0.8, servo=0.0, tol=0.03):
        """両手の指先を center の左右へ動かし、servo 秒間 IK で追従する
        戻り値: 大きい方のずれ[m]"""
        def solve():
            _, _, left = self._body_frame()
            sol = {}
            for side, sgn in (("left", 1), ("right", -1)):
                sol.update(self.sim.solve_arm_ik(side, center + sgn * half_width * left, iters=80)[0])
            return sol, left

        sol, left = solve()
        self.sim.move_arms(sol, duration)
        self.idle(duration + 0.1)
        t_end = self.sim.time + servo
        while True:
            _, _, left = self._body_frame()
            err = max(np.linalg.norm(self.sim.site_pos(f"{sd}_hand_tip") - (center + sg * half_width * left))
                      for sd, sg in (("left", 1), ("right", -1)))
            if err < tol or self.sim.time >= t_end:
                return float(err)
            sol, _ = solve()
            self.sim.move_arms(sol, 0.1)
            self.idle(0.1)

    def _carry_pose(self):
        base, fwd, _ = self._body_frame()
        self._hands_to(base + fwd * 0.33 + np.array([0, 0, 0.95]), self._held_half + 0.012, duration=0.9)

    def _approach_close(self, target_xy, dist, timeout=10.0):
        """安全層の停止距離を縮め、target_xy の dist 手前までゆっくり近づく"""
        stop, slow = self.sim.safety.stop_distance, self.sim.safety.slow_distance
        self.sim.safety.stop_distance, self.sim.safety.slow_distance = 0.12, 0.35
        try:
            self.face(*target_xy)
            px, py, _ = self.sim.pose()
            d = math.hypot(target_xy[0] - px, target_xy[1] - py)
            if d > dist:
                k = (d - dist) / d
                self.walk_to(x=px + (target_xy[0] - px) * k, y=py + (target_xy[1] - py) * k, stop_radius=0.05,
                             timeout=timeout)
            self.face(*target_xy)
        finally:
            self.sim.safety.stop_distance, self.sim.safety.slow_distance = stop, slow

    def pick_up(self, color: str):
        """指定した色の物の前まで行き、両手ではさんで持ち上げる（低い物はしゃがんで取る）"""
        if not self.sim.arms:
            return self._result(False, "このロボットのモデルには動かせる腕がありません")
        if self.sim.held is not None:
            return self._result(False, f"すでに{self.sim.held[0]}を持っています。先に put_down で置いてください")
        from . import squat_policy

        low_limit = 0.40 if squat_policy.available() else 0.62
        cands = [p for p in self.sim.scene.objects if color in p.tags]
        prop, low = None, None
        for i in range(6):
            res, pos = self.find_object(color)
            if pos is not None and cands:
                near = min(cands, key=lambda p: math.hypot(*(self.sim.prop_pos(p.name)[:2] - np.array(pos))))
                if math.hypot(*(self.sim.prop_pos(near.name)[:2] - np.array(pos))) <= 0.6:
                    if self.sim.prop_pos(near.name)[2] >= low_limit:
                        prop = near
                        break
                    low = near
            if i < 5:
                self.turn(60)
        if prop is None:
            if low is not None:
                return self._result(False, f"見つかった{low.name}は低い位置（高さ{self.sim.prop_pos(low.name)[2]:.2f}m）にあり、"
                                           "しゃがんでも手が届きません。床の物は拾えません")
            return self._result(False, f"持ち運べる{COLOR_JA[color]}物は、この場所から見回しても見つかりませんでした。"
                                       "探し物がありそうな場所（棚やテーブルの近く）へ移動してから試してください")
        obj = self.sim.prop_pos(prop.name)
        if prop.shape == "prism":
            half, half_z = 0.5 * float(prop.size[0]), float(prop.size[2])
        elif prop.shape == "sphere":
            half = half_z = float(prop.size[0])
        else:
            half, half_z = float(prop.size[0]), float(prop.size[-1])
        px, py, _ = self.sim.pose()
        d = math.hypot(obj[0] - px, obj[1] - py)
        squat = obj[2] < 0.62
        if squat:
            # 斜めから近づくと、しゃがんだ膝が棚に当たる
            nx, ny = self._shelf_normal(obj)
            moved = self.walk_to(x=obj[0] + 0.65 * nx, y=obj[1] + 0.65 * ny, stop_radius=0.12)
            if not moved["success"]:
                return moved
            px, py, _ = self.sim.pose()
            d = math.hypot(obj[0] - px, obj[1] - py)
        if d > 1.0:
            k = (d - 0.95) / d
            moved = self.walk_to(x=px + (obj[0] - px) * k, y=py + (obj[1] - py) * k, stop_radius=0.2)
            if not moved["success"]:
                return moved
        self._approach_close(obj[:2], 0.28 if squat else 0.33)
        if squat:
            # まれに大きく回るので、ずれが大きければやり直す
            n = self._shelf_normal(obj)
            want_yaw = math.atan2(-n[1], -n[0])
            for attempt in range(2):
                self._align_for_squat(obj)
                self.sim.set_mode("squat")
                self.sim.squat_target = float(np.clip(obj[2] + 0.06, 0.50, 0.78))
                t_end = self.sim.time + 5.0
                while self.sim.time < t_end and abs(self.sim.data.qpos[2] - self.sim.squat_target) > 0.03:
                    self.idle(0.2)
                self.idle(0.5)
                if self.sim.state.fallen:
                    break
                yaw_err = abs(_wrap(self.sim.pose()[2] - want_yaw))
                err = 1.0
                if yaw_err < math.radians(12):
                    self._held_half, self._held_half_z = half, half_z
                    err = self._hands_to(obj, half + 0.012, duration=1.0, servo=3.0, tol=0.03)
                if os.environ.get("G1_DEBUG_PRESS"):
                    print(f"     pick[{attempt + 1}]: 向きのずれ {math.degrees(yaw_err):.0f}° 手のずれ {err * 100:.0f}cm "
                          f"pose={[round(v, 2) for v in self.sim.pose()]} 腰={self.sim.data.qpos[2]:.2f}")
                if err <= 0.05 or self.sim.state.fallen:
                    break
                if attempt == 0:
                    self.sim.move_arms({n_: 0.0 for n_ in self.sim.upper_names}, 0.6)
                    self.idle(0.7)
                    self._stand_up()
                    self.walk("backward", 0.25)
        else:
            self._held_half, self._held_half_z = half, half_z
            err = self._hands_to(obj, half + 0.012, duration=1.0, servo=3.0, tol=0.03)
        if os.environ.get("G1_DEBUG_PRESS"):
            print(f"     pick: err={err:.3f} pose={[round(v, 2) for v in self.sim.pose()]} obj={np.round(obj, 2)} "
                  f"腰={self.sim.data.qpos[2]:.2f}")
        if err > 0.05:
            self.sim.move_arms({n_: 0.0 for n_ in self.sim.upper_names}, 0.6)
            self.idle(0.7)
            self._stand_up()
            self.walk("backward", 0.4)
            return self._result(False, f"{prop.name}に両手が届きませんでした（ずれ {err * 100:.0f}cm）")
        self.sim.grasp(prop.name)
        if squat:
            self._hands_to(obj + np.array([0, 0, 0.08]), half + 0.012, duration=0.6)
            self._stand_up()
        self._carry_pose()
        self.walk("backward", 0.4)
        return self._result(True, f"{prop.name}を両手で持ち上げました。届け先の台の前で put_down で置けます",
                            holding=prop.name)

    @staticmethod
    def _shelf_normal(obj):
        """棚の正面の向き（物にいちばん近い壁から離れる向き）"""
        r = world.ROOM
        walls = [(obj[0] - r["x_min"], (1.0, 0.0)), (r["x_max"] - obj[0], (-1.0, 0.0)),
                 (obj[1] - r["y_min"], (0.0, 1.0)), (r["y_max"] - obj[1], (0.0, -1.0))]
        return np.array(min(walls)[1])

    def _align_for_squat(self, obj, dist=0.28):
        """物の正面・dist 手前に、小さな速度指令で位置と向きを合わせる（閉ループ）"""
        n = self._shelf_normal(obj)
        # 物の方向ではなく棚に垂直な向きに合わせる（近いと足踏みのずれで方向が大きく変わる）
        goal_xy = np.asarray(obj[:2]) + dist * n
        goal_yaw = _wrap(math.atan2(-n[1], -n[0]))

        def errors():
            x, y, yaw = self.sim.pose()
            c, s_ = math.cos(yaw), math.sin(yaw)
            ex, ey = goal_xy[0] - x, goal_xy[1] - y
            return c * ex + s_ * ey, -s_ * ex + c * ey, _wrap(goal_yaw - yaw)

        def push(e, k, lim, floor, tol):
            if abs(e) <= tol:
                return 0.0
            return float(np.clip(k * e, -lim, lim)) if abs(k * e) > floor else math.copysign(floor, e)

        def cmd():
            f, l, w = errors()
            # 後ろ向きの小さな速度ではほとんど動かない
            return (push(f, 1.5, 0.2, 0.08 if f > 0 else 0.14, 0.02), push(l, 1.5, 0.15, 0.08, 0.02),
                    push(w, 1.5, 0.4, 0.15, math.radians(3)))

        def done():
            f, l, w = errors()
            return abs(f) < 0.03 and abs(l) < 0.05 and abs(w) < math.radians(5)

        stop, slow = self.sim.safety.stop_distance, self.sim.safety.slow_distance
        self.sim.safety.stop_distance, self.sim.safety.slow_distance = 0.12, 0.35
        try:
            # 待つと前に流れて近づきすぎる
            self._drive(cmd, done, 8.0)
        finally:
            self.sim.safety.stop_distance, self.sim.safety.slow_distance = stop, slow
        # 足踏み中なので、両足が床についた瞬間に切り替える
        t_end = self.sim.time + 0.8
        while self.sim.time < t_end and not self.sim.feet_on_ground():
            self.sim.step((0, 0, 0), 0.02)
        if os.environ.get("G1_DEBUG_PRESS"):
            f, l, w = errors()
            print(f"     align: 前{f:+.3f} 左{l:+.3f} 向き{math.degrees(w):+.1f}° 角速度{self.sim.data.qvel[5]:+.2f}")

    def _stand_up(self):
        if self.sim.mode != "squat":
            return
        self.sim.squat_target = 0.78
        t_end = self.sim.time + 4.0
        while self.sim.time < t_end and self.sim.data.qpos[2] < 0.74:
            self.idle(0.2)
        self.idle(0.4)
        self.sim.set_mode("walk")
        self.idle(0.4)

    def put_down(self):
        """持っている物を目の前の台の上に置く"""
        if self.sim.held is None:
            return self._result(False, "何も持っていません")
        x0, y0, yaw = self.sim.pose()
        gid = np.zeros(1, dtype=np.int32)
        groups = np.array([0, 0, 1, 1, 0, 0], dtype=np.uint8)
        best = None
        for dyaw in np.radians(np.arange(-60, 61, 10)):
            a = yaw + dyaw
            for ahead in np.arange(0.35, 1.3, 0.05):
                p = np.array([x0 + ahead * math.cos(a), y0 + ahead * math.sin(a), 1.6])
                dd = mujoco.mj_ray(self.sim.model, self.sim.data, p, np.array([0, 0, -1.0]), groups, 1, -1, gid)
                if dd > 0 and 0.6 < 1.6 - dd < 1.3:
                    if best is None or ahead < best[0]:
                        best = (ahead, a, 1.6 - dd)
                    break
        if best is None:
            return self._result(False, "近くに物を置ける台（高さ0.6〜1.3m）がありません。台の正面に移動してから put_down してください")
        hit_at, a, top = best
        base = np.array([x0, y0, 0.0])
        fwd = np.array([math.cos(a), math.sin(a), 0.0])
        spot = base + fwd * (hit_at + 0.15)
        self._approach_close(spot[:2], 0.33)
        _, _, left = self._body_frame()
        for shift in (0.0, 0.22, -0.22, 0.44, -0.44):
            cand = spot + left * shift
            if all(np.linalg.norm(self.sim.prop_pos(o.name)[:2] - cand[:2]) > 0.16
                   for o in self.sim.scene.objects if o.name != self.sim.held[0]):
                spot = cand
                break
        place = np.array([spot[0], spot[1], top + self._held_half_z + 0.005])
        self._hands_to(place + np.array([0, 0, 0.02]), self._held_half + 0.012, duration=1.0, servo=1.5, tol=0.03)
        name = self.sim.release(place)
        self.sim.move_arms({n: 0.0 for n in self.sim.upper_names}, 0.7)
        self.idle(0.8)
        self.walk("backward", 0.4)
        return self._result(True, f"{name}を台（高さ{top:.2f}m）の上に置きました")

    def _hand_touches(self, geom_id: int) -> bool:
        m, d = self.sim.model, self.sim.data
        hand = {m.body(n).id for n in ("right_wrist_roll_link", "right_wrist_pitch_link", "right_wrist_yaw_link")}
        for c in d.contact[:d.ncon]:
            pair = (c.geom1, c.geom2)
            if geom_id in pair and any(m.geom_bodyid[g] in hand for g in pair):
                return True
        return False

    def press_button(self, name: str = "エレベーターのボタン"):
        """壁のボタンの前まで行き、右手の指先で押す"""
        if not self.sim.arms:
            return self._result(False, "このロボットのモデルには動かせる腕がありません")
        if name not in self.sim._prop_index:
            buttons = [p.name for p in self.sim.scene.all_props() if "button" in p.tags]
            return self._result(False, f"「{name}」というボタンはありません。候補: {'、'.join(buttons)}")
        b = self.sim.prop_pos(name)
        r = world.ROOM
        walls = {(0, 1): r["y_max"] - b[1], (0, -1): b[1] - r["y_min"], (1, 0): r["x_max"] - b[0], (-1, 0): b[0] - r["x_min"]}
        n = np.array(min(walls, key=walls.get), dtype=float)
        approach = b[:2] - n * 0.95
        res = self.walk_to(x=float(approach[0]), y=float(approach[1]), stop_radius=0.2)
        if not res["success"]:
            return res
        self.face(float(b[0]), float(b[1]))
        stop, slow = self.sim.safety.stop_distance, self.sim.safety.slow_distance
        self.sim.safety.stop_distance, self.sim.safety.slow_distance = 0.32, 0.5
        try:
            close = b[:2] - n * 0.40
            self.walk_to(x=float(close[0]), y=float(close[1]), stop_radius=0.06, timeout=12)
            self.face(float(b[0]), float(b[1]))
            # 足踏みで体が揺れるので、0.1 秒ごとに IK を解き直す
            pressed, miss = False, None
            face = np.array([b[0], b[1], b[2]]) - np.append(n, 0) * 0.012
            goal = face + np.append(n, 0) * 0.015
            lateral = np.array([-n[1], n[0], 0.0])
            sol, _ = self.sim.solve_arm_ik("right", goal)
            self.sim.move_arms(sol, 0.9)
            self.idle(0.9)
            t_end = self.sim.time + 3.0
            while self.sim.time < t_end:
                sol, _ = self.sim.solve_arm_ik("right", goal, iters=60)
                self.sim.move_arms(sol, 0.1)
                self.idle(0.1)
                rel = self.sim.site_pos("right_hand_tip") - face
                depth, side, up = float(rel @ np.append(n, 0)), float(rel @ lateral), float(rel[2])
                miss = math.sqrt(side * side + up * up + min(0.0, depth) ** 2)
                if os.environ.get("G1_DEBUG_PRESS"):
                    print(f"     t={self.sim.time:.1f} depth={depth:+.3f} side={side:+.3f} up={up:+.3f}")
                if abs(side) < 0.04 and abs(up) < 0.04 and -0.005 < depth < 0.03:
                    pressed = True
                    break
            if pressed:
                gid = self.sim.model.geom(f"prop{self.sim._prop_index[name]}").id
                self.sim.model.geom_rgba[gid] = (1.0, 0.65, 0.1, 1)
                self.sim.pressed_buttons.add(name)
                self.sim.state.events.append((self.sim.time, f"{name}が押された"))
                self.idle(0.6)
            self.rest_arms()
            self.walk("backward", 0.4)
        finally:
            self.sim.safety.stop_distance, self.sim.safety.slow_distance = stop, slow
        if not pressed:
            return self._result(False, f"{name}に指先が届きませんでした（ずれ {miss * 100:.0f}cm）")
        opened = self.sim.on_button_pressed(name)
        return self._result(True, f"{name}を押しました。" + (opened or ""))

    def say(self, message: str, duration: float | None = None):
        """話す（吹き出しと、あれば音声）
        発話中はその場で待つ"""
        self.speech = (message, self.sim.time)
        self.idle(duration if duration is not None else min(6.0, 1.5 + len(message) * 0.08))
        self.speech = None
        return self._result(True, "発話しました")
