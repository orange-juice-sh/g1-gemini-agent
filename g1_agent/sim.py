"""MuJoCo 上の Unitree G1
歩行ポリシー（50Hz）+ PD（500Hz）、AI から独立した安全層（仮想 LiDAR）、転倒検知、頭部カメラ"""
from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass, field

import mujoco
import numpy as np
import yaml
import xml.etree.ElementTree as ET

from . import world
from .policy import load_policy

DEFAULT_RL_GYM = os.environ.get("UNITREE_RL_GYM", os.path.expanduser("~/unitree_rl_gym"))

OBSTACLE_GROUP = 2  # 仮想 LiDAR が当たる group
PROXY_GROUP = 3  # 見えない当たり判定（LiDAR は当たる）
ASSETS = os.environ.get("G1_ASSETS", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets"))

ARMS_DEFAULT = os.environ.get("G1_ARMS", "1") != "0"
ARM_MODEL = "g1_29dof_lock_waist.xml"
LEG_JOINTS = [f"{s}_{j}_joint" for s in ("left", "right")
              for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
UPPER_JOINTS = ["waist_yaw_joint"] + [f"{s}_{j}_joint" for s in ("left", "right") for j in
                                      ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                                       "wrist_roll", "wrist_pitch", "wrist_yaw")]


def _upper_gains(name):
    """上半身の PD ゲイン (kp, kd)
    kd は陰的な減衰として積分する"""
    if "waist" in name:
        return 200.0, 5.0
    if "shoulder" in name:
        return 60.0, 2.0
    if "elbow" in name:
        return 40.0, 2.0
    return 20.0, 1.0


@dataclass
class SafetyConfig:
    stop_distance: float = 0.7  # [m] これより近いと並進停止
    slow_distance: float = 1.1  # [m] これより近いと減速
    sector_deg: float = 35.0  # [deg] 監視する進行方向の範囲


@dataclass
class SimState:
    safety_stops: int = 0
    safety_active: bool = False
    fallen: bool = False
    events: list = field(default_factory=list)


def _yaw_quat(yaw):
    return [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]


def _quat_to_yaw(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _gravity_orientation(q):
    w, x, y, z = q
    return np.array([2 * (-z * x + w * y), -2 * (z * y + w * x), 1 - 2 * (w * w + z * z)])


def _sign_texture(text: str, size) -> str:
    """案内板の文字の PNG を作る（MuJoCo は box の面に左右反転して貼るので反転しておく）"""
    from PIL import Image, ImageDraw, ImageFont

    w = 1024
    h = max(64, int(w * size[2] / size[1]))
    img = Image.new("RGB", (w, h), (250, 250, 246))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, w - 1, h - 1], outline=(40, 60, 90), width=max(8, w // 80))
    font = None
    for fs in range(140, 20, -6):
        for p in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", "/usr/share/fonts/noto-cjk/NotoSansCJK-Bold.ttc"):
            if os.path.exists(p):
                font = ImageFont.truetype(p, fs)
                break
        if font is None:
            font = ImageFont.load_default()
            break
        lines = text.split("\n")
        if max(d.textlength(l, font=font) for l in lines) < w * 0.9 and len(lines) * fs * 1.3 < h * 0.85:
            break
    lines = text.split("\n")
    lh = font.size * 1.3 if hasattr(font, "size") else 16
    y = (h - lh * len(lines)) / 2
    for line in lines:
        d.text(((w - d.textlength(line, font=font)) / 2, y), line, font=font, fill=(25, 35, 60))
        y += lh
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    out_dir = os.path.join(tempfile.gettempdir(), f"g1_signs_{os.getuid() if hasattr(os, 'getuid') else 0}")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"sign_{abs(hash((text, tuple(size)))) % 10**10}.png")
    # 並列実行で書きかけを読まないよう置き換えで書く
    tmp = f"{path}.{os.getpid()}.tmp"
    img.save(tmp, format="PNG")
    os.replace(tmp, path)
    return path


def _prism_mesh(size) -> str:
    """三角柱の OBJ を作る（中心が原点）"""
    w, d, h = size
    v = [(-w, -d, -h), (w, -d, -h), (0, -d, h), (-w, d, -h), (w, d, -h), (0, d, h)]
    faces = [(1, 3, 2), (4, 5, 6), (1, 2, 5), (1, 5, 4), (2, 3, 6), (2, 6, 5), (3, 1, 4), (3, 4, 6)]
    out_dir = os.path.join(tempfile.gettempdir(), f"g1_meshes_{os.getuid() if hasattr(os, 'getuid') else 0}")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"prism_{w:.3f}_{d:.3f}_{h:.3f}.obj")
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        f.write("".join(f"v {a} {b} {c}\n" for a, b, c in v) + "".join(f"f {a} {b} {c}\n" for a, b, c in faces))
    os.replace(tmp, path)
    return path


def _human_geoms(wb, name, x, y, yaw, shirt, scale=1.0):
    """マネキン風の人の見た目（足元を中心に scale 倍）"""
    n_before = len(list(wb))
    _human_parts(wb, name, x, y, yaw, shirt)
    if scale == 1.0:
        return

    def sp(vals):
        v = [float(t) for t in vals.split()]
        out = []
        for i in range(0, len(v), 3):
            out += [x + (v[i] - x) * scale, y + (v[i + 1] - y) * scale, v[i + 2] * scale]
        return " ".join(f"{t:.4f}" for t in out)

    for el in list(wb)[n_before:]:
        for key in ("pos", "fromto"):
            if el.get(key):
                el.set(key, sp(el.get(key)))
        if el.get("size"):
            el.set("size", " ".join(f"{float(t) * scale:.4f}" for t in el.get("size").split()))


def _human_parts(wb, name, x, y, yaw, shirt):
    c, s = math.cos(yaw), math.sin(yaw)

    def P(fx, fy, z):
        return f"{x + c * fx - s * fy:.4f} {y + s * fx + c * fy:.4f} {z:.4f}"

    skin, pants, hair, shoes = "0.93 0.78 0.64 1", "0.16 0.17 0.22 1", "0.12 0.09 0.07 1", "0.08 0.08 0.08 1"
    shirt = " ".join(f"{v:.3g}" for v in shirt[:3]) + " 1"
    g = dict(contype="0", conaffinity="0", group="0")
    yaw_e = f"0 0 {yaw:.4f}"
    for i, fy in enumerate((0.09, -0.09)):
        ET.SubElement(wb, "geom", name=f"{name}_leg{i}", type="capsule", fromto=f"{P(0, fy, 0.1)} {P(0, fy, 0.82)}",
                      size="0.065", rgba=pants, **g)
        ET.SubElement(wb, "geom", name=f"{name}_shoe{i}", type="box", pos=P(0.04, fy, 0.04), size="0.11 0.05 0.04",
                      euler=yaw_e, rgba=shoes, **g)
        ET.SubElement(wb, "geom", name=f"{name}_arm{i}", type="capsule",
                      fromto=f"{P(0, fy * 2.6, 1.36)} {P(0.04, fy * 2.75, 0.92)}", size="0.05", rgba=shirt, **g)
        ET.SubElement(wb, "geom", name=f"{name}_hand{i}", type="sphere", pos=P(0.05, fy * 2.78, 0.86), size="0.045",
                      rgba=skin, **g)
    ET.SubElement(wb, "geom", name=f"{name}_hips", type="ellipsoid", pos=P(0, 0, 0.86), size="0.13 0.19 0.1",
                  euler=yaw_e, rgba=pants, **g)
    ET.SubElement(wb, "geom", name=f"{name}_torso", type="ellipsoid", pos=P(0, 0, 1.15), size="0.13 0.21 0.27",
                  euler=yaw_e, rgba=shirt, **g)
    ET.SubElement(wb, "geom", name=f"{name}_neck", type="capsule", fromto=f"{P(0, 0, 1.38)} {P(0, 0, 1.47)}",
                  size="0.05", rgba=skin, **g)
    ET.SubElement(wb, "geom", name=f"{name}_head", type="ellipsoid", pos=P(0.01, 0, 1.57), size="0.095 0.085 0.115",
                  euler=yaw_e, rgba=skin, **g)
    ET.SubElement(wb, "geom", name=f"{name}_hair", type="ellipsoid", pos=P(-0.02, 0, 1.63), size="0.1 0.092 0.085",
                  euler=yaw_e, rgba=hair, **g)


def _mesh_geoms(asset, wb, key, model, x, y, yaw, scale):
    """家具のモデルを見た目として置く
    素材がなければ False"""
    import json

    d = os.path.join(ASSETS, "models", model)
    meta_path = os.path.join(d, "meta.json")
    if not os.path.exists(meta_path):
        return False
    meta = json.load(open(meta_path, encoding="utf-8"))
    for k, part in enumerate(meta["parts"]):
        mname = f"{key}_m{k}"
        sc = scale if isinstance(scale, (tuple, list)) else (scale, scale, scale)
        ET.SubElement(asset, "mesh", name=mname, file=os.path.join(d, part["obj"]), scale=" ".join(f"{v:g}" for v in sc),
                      inertia="shell")
        attrs = dict(type="mesh", mesh=mname, pos=f"{x:.4f} {y:.4f} 0", euler=f"0 0 {yaw:.4f}",
                     contype="0", conaffinity="0", group="0")
        if part.get("png"):
            ET.SubElement(asset, "texture", name=f"{mname}_tex", type="2d", file=os.path.join(d, part["png"]))
            ET.SubElement(asset, "material", name=f"{mname}_mat", texture=f"{mname}_tex", specular="0.2")
            attrs["material"] = f"{mname}_mat"
        else:
            attrs["rgba"] = " ".join(f"{v:.3g}" for v in part["rgba"])
        ET.SubElement(wb, "geom", name=f"{key}_v{k}", **attrs)
    return True


def build_model(rl_gym_dir: str, scene: world.Scene | None = None, arms: bool = ARMS_DEFAULT) -> mujoco.MjModel:
    """G1 のモデルにロビーとカメラを加えた MuJoCo モデルを作る"""
    scene = scene or world.default_scene()
    desc = os.path.join(rl_gym_dir, "resources/robots/g1_description")
    root = ET.parse(os.path.join(desc, ARM_MODEL if arms else "g1_12dof.xml")).getroot()
    root.find("compiler").set("meshdir", os.path.join(desc, "meshes") + os.sep)
    if arms:
        for tag in ("visual",):
            for el in root.findall(tag):
                root.remove(el)
        for parent in root.iter():
            for el in list(parent):
                if el.tag == "texture" and el.get("type") == "skybox" or el.get("name") == "groundplane" \
                        or el.tag == "light" or (el.tag == "geom" and el.get("type") == "plane"):
                    parent.remove(el)
        # 12dof の XML と同じ既定値（ないと軽い関節の PD が発振する）
        default = root.find("default")
        if default is None:
            default = ET.Element("default")
            root.insert(1, default)
        ET.SubElement(default, "joint", damping="0.001", armature="0.01", frictionloss="0.1")
        for side in ("left", "right"):
            hand = next(b for b in root.iter("body") if b.get("name") == f"{side}_wrist_yaw_link")
            ET.SubElement(hand, "site", name=f"{side}_hand_tip", pos="0.13 0 0", size="0.01", rgba="1 0 0 0")
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", "0.002")

    vis = ET.SubElement(root, "visual")
    ET.SubElement(vis, "global", offwidth="1920", offheight="1080", azimuth="-140", elevation="-25")
    ET.SubElement(vis, "headlight", ambient="0.32 0.32 0.32", diffuse="0.25 0.25 0.25", specular="0.05 0.05 0.05")
    ET.SubElement(vis, "quality", shadowsize="8192", offsamples="8")
    ET.SubElement(vis, "map", haze="0.1", shadowclip="12")

    # 色検出に反応しないよう、床と壁は無彩色に近づける
    asset = root.find("asset")
    tex_dir = os.path.join(ASSETS, "textures")
    have_tex = os.path.exists(os.path.join(tex_dir, "floor.png"))
    if have_tex:
        for key, rep_ in (("floor", "0.45 0.45"), ("wall", "0.6 0.6"), ("carpet", "0.8 0.8"),
                          ("wood_light", "1 1"), ("wood_dark", "1 1")):
            ET.SubElement(asset, "texture", name=f"{key}_tex", type="2d", file=os.path.join(tex_dir, f"{key}.png"))
        ET.SubElement(asset, "material", name="floor_mat", texture="floor_tex", texrepeat="0.45 0.45", texuniform="true",
                      reflectance="0.18", specular="0.4", shininess="0.6")
        ET.SubElement(asset, "material", name="wall_mat", texture="wall_tex", texrepeat="0.6 0.6", texuniform="true",
                      rgba="0.97 0.96 0.94 1", specular="0.1")
        ET.SubElement(asset, "material", name="carpet_mat", texture="carpet_tex", texrepeat="0.8 0.8", texuniform="true")
        ET.SubElement(asset, "material", name="wood_light_mat", texture="wood_light_tex", texrepeat="1 1",
                      texuniform="true", specular="0.3")
        ET.SubElement(asset, "material", name="wood_dark_mat", texture="wood_dark_tex", texrepeat="1 1",
                      texuniform="true", specular="0.3")
    else:
        ET.SubElement(asset, "texture", name="floor_tex", type="2d", builtin="checker", width="512", height="512",
                      rgb1="0.78 0.76 0.72", rgb2="0.72 0.70 0.66", mark="edge", markrgb="0.60 0.58 0.55")
        ET.SubElement(asset, "material", name="floor_mat", texture="floor_tex", texrepeat="1.6 1.6", texuniform="true",
                      reflectance="0.12", specular="0.3", shininess="0.4")
        ET.SubElement(asset, "material", name="wall_mat", rgba="0.90 0.89 0.86 1", specular="0.1")
    ET.SubElement(asset, "material", name="trim_mat", rgba="0.36 0.34 0.32 1")
    ET.SubElement(asset, "texture", name="sky", type="skybox", builtin="gradient", width="256", height="256",
                  rgb1="0.93 0.95 0.98", rgb2="0.70 0.76 0.84")

    def f(v):
        return " ".join(f"{x:.4g}" for x in v)

    wb = root.find("worldbody")
    ET.SubElement(wb, "light", name="sun", pos="4 1 8", dir="-0.25 -0.15 -1", directional="true",
                  diffuse="0.50 0.49 0.46", specular="0.15 0.15 0.15", castshadow="true")
    ET.SubElement(wb, "light", pos="-1 -4 4", dir="0.4 0.6 -1", directional="true", diffuse="0.18 0.19 0.21",
                  specular="0 0 0", castshadow="false")
    for i, (lx, ly) in enumerate([(1.0, 0.0), (6.0, 2.5), (6.0, -2.5), (9.0, 0.0)]):
        ET.SubElement(wb, "light", name=f"ceiling{i}", pos=f((lx, ly, 3.2)), dir="0 0 -1", directional="false",
                      diffuse="0.16 0.15 0.13", specular="0 0 0", attenuation="1 0.1 0.02", castshadow="false")
    rm = world.ROOM
    ET.SubElement(wb, "geom", name="floor", type="plane", material="floor_mat",
                  pos=f"{(rm['x_min'] + rm['x_max']) / 2:.3f} {(rm['y_min'] + rm['y_max']) / 2:.3f} 0",
                  size=f"{(rm['x_max'] - rm['x_min']) / 2 + 0.05:.3f} {(rm['y_max'] - rm['y_min']) / 2 + 0.05:.3f} 0.05")
    ET.SubElement(wb, "geom", name="ground", type="plane", size="0 0 0.05", pos="0 0 -0.002", rgba="0.42 0.43 0.45 1",
                  contype="0", conaffinity="0")

    r = world.ROOM
    cx, cy = (r["x_min"] + r["x_max"]) / 2, (r["y_min"] + r["y_max"]) / 2
    hx, hy, h = (r["x_max"] - r["x_min"]) / 2, (r["y_max"] - r["y_min"]) / 2, r["wall_h"] / 2
    H = world.ROOM.get("tall_wall_h", 2.6) / 2
    walls = [((cx, r["y_max"], H), (hx, 0.05, H)), ((cx, r["y_min"], h), (hx, 0.05, h)),
             ((r["x_max"], cy, H), (0.05, hy, H)), ((r["x_min"], cy, h), (0.05, hy, h))]
    for i, (pos, size) in enumerate(walls):
        h = pos[2]
        ET.SubElement(wb, "geom", name=f"wall{i}", type="box", pos=f(pos), size=f(size),
                      material="wall_mat", group=str(OBSTACLE_GROUP))
        top = (pos[0], pos[1], 2 * h + 0.015)
        ET.SubElement(wb, "geom", name=f"wall_trim{i}", type="box", pos=f(top),
                      size=f((size[0] + 0.01, size[1] + 0.01, 0.015)), material="trim_mat",
                      contype="0", conaffinity="0", group="0")

    for i, loc in enumerate(world.LOCATIONS):
        if loc.name in scene.hidden_zones or loc.name in scene.hidden_locations:
            continue
        ET.SubElement(wb, "geom", name=f"zone{i}", type="cylinder", pos=f((loc.x, loc.y, 0.002)),
                      size="0.55 0.002", rgba=f(loc.color), contype="0", conaffinity="0")

    for i, p in enumerate(scene.all_props()):
        attrs = dict(name=f"prop{i}", type=p.shape, pos=f(p.pos), size=f(p.size), rgba=f(p.rgba),
                     euler=f"0 0 {math.radians(p.yaw_deg):.4f}", group=str(OBSTACLE_GROUP if p.obstacle else 0))
        if p.shape == "prism":
            ET.SubElement(asset, "mesh", name=f"prism{i}", file=_prism_mesh(p.size))
            attrs.update(type="mesh", mesh=f"prism{i}")
            del attrs["size"]
        if not p.collide:
            attrs.update(contype="0", conaffinity="0")
        if p.material and have_tex:
            attrs["material"] = f"{p.material}_mat"
            attrs["rgba"] = "1 1 1 1"
        # 見た目を差し替えた物は、元の形を見えない当たり判定にする
        yaw_v = math.radians(p.yaw_deg + p.visual_yaw_deg)
        if p.visual.startswith("mesh:") and _mesh_geoms(asset, wb, f"prop{i}", p.visual[5:], p.pos[0], p.pos[1], yaw_v,
                                                        p.visual_scale):
            attrs["group"] = str(PROXY_GROUP if p.obstacle else 4)
        elif p.visual == "hidden":
            attrs["group"] = str(PROXY_GROUP if p.obstacle else 4)
        elif p.visual == "human":
            scale = p.visual_scale if isinstance(p.visual_scale, (int, float)) else 1.0
            _human_geoms(wb, f"prop{i}", p.pos[0], p.pos[1], yaw_v, p.rgba, scale)
            attrs["group"] = str(PROXY_GROUP if p.obstacle else 4)
        if p.text:
            ET.SubElement(asset, "texture", name=f"sign_tex{i}", type="2d", file=_sign_texture(p.text, p.size))
            ET.SubElement(asset, "material", name=f"sign_mat{i}", texture=f"sign_tex{i}")
            attrs["material"] = f"sign_mat{i}"
            attrs["rgba"] = "1 1 1 1"
        ET.SubElement(wb, "geom", **attrs)

    # カメラの視線は -z
    pelvis = next(b for b in wb.iter("body") if b.get("name") == "pelvis")
    t = math.radians(18)
    ET.SubElement(pelvis, "camera", name="head_cam", pos="0.1 0 0.48", fovy="70",
                  xyaxes=f"0 -1 0 {math.sin(t):.4f} 0 {math.cos(t):.4f}")
    # 肩の付け根の当たり判定は実物より大きく、両手を寄せると胴体に当たるので外す
    body_names = {b.get("name") for b in wb.iter("body")}
    if "torso_link" in body_names:
        contact = root.find("contact")
        if contact is None:
            contact = ET.SubElement(root, "contact")
        for side in ("left", "right"):
            for part in ("pitch", "roll", "yaw"):
                if f"{side}_shoulder_{part}_link" in body_names:
                    ET.SubElement(contact, "exclude", body1="torso_link", body2=f"{side}_shoulder_{part}_link")
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


class G1Sim:
    def __init__(self, rl_gym_dir: str = DEFAULT_RL_GYM, use_torch: bool = False,
                 safety: SafetyConfig | None = None, scene: world.Scene | None = None, arms: bool | None = None):
        cfg_path = os.path.join(rl_gym_dir, "deploy/deploy_mujoco/configs/g1.yaml")
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        self.cfg = cfg
        self.kps = np.array(cfg["kps"], dtype=np.float32)
        self.kds = np.array(cfg["kds"], dtype=np.float32)
        self.default_angles = np.array(cfg["default_angles"], dtype=np.float32)
        self.cmd_scale = np.array(cfg["cmd_scale"], dtype=np.float32)
        self.decimation = cfg["control_decimation"]
        self.n_act = cfg["num_actions"]

        self.scene = scene or world.default_scene()
        self.arms = ARMS_DEFAULT if arms is None else arms
        self.model = build_model(rl_gym_dir, self.scene, self.arms)
        m = self.model
        # free joint を足しても番号がずれないよう名前から求める
        self.leg_q = np.array([m.joint(n).qposadr[0] for n in LEG_JOINTS])
        self.leg_v = np.array([m.joint(n).dofadr[0] for n in LEG_JOINTS])
        self.leg_u = np.array([m.actuator(n).id for n in LEG_JOINTS])
        upper = UPPER_JOINTS if self.arms else []
        self.upper_names = upper
        self.up_q = np.array([m.joint(n).qposadr[0] for n in upper], dtype=int)
        self.up_v = np.array([m.joint(n).dofadr[0] for n in upper], dtype=int)
        self.up_u = np.array([m.actuator(n).id for n in upper], dtype=int)
        self.up_kp = np.array([_upper_gains(n)[0] for n in upper])
        for n, dof in zip(upper, self.up_v):
            m.dof_damping[dof] = _upper_gains(n)[1]
        self.arm_target = np.zeros(len(upper))
        self._arm_motion = None
        self._prop_index = {p.name: i for i, p in enumerate(self.scene.all_props())}
        self.data = mujoco.MjData(self.model)
        self.policy = load_policy(os.path.join(rl_gym_dir, "deploy/pre_train/g1/motion.pt"), use_torch)
        self.safety = safety or SafetyConfig()
        self.state = SimState()
        self.frame_callbacks = []
        self.max_cmd = np.array([0.6, 0.4, 0.8], dtype=np.float32)

        self.lidar_angles = np.radians(np.arange(-180, 180, 5))
        self.lidar_height = 0.25
        self.lidar_range = 6.0
        self._geomgroup = np.zeros(6, dtype=np.uint8)
        self._geomgroup[OBSTACLE_GROUP] = 1
        self._geomgroup[PROXY_GROUP] = 1
        self._geomid = np.zeros(1, dtype=np.int32)
        self.reset()

    def reset(self, pose=None):
        pose = pose or self.scene.start_pose
        mujoco.mj_resetData(self.model, self.data)
        x, y, yaw = pose
        self.data.qpos[:3] = [x, y, 0.79]
        self.data.qpos[3:7] = _yaw_quat(yaw)
        self.data.qpos[self.leg_q] = self.default_angles
        mujoco.mj_forward(self.model, self.data)
        self.policy.reset()
        self.action = np.zeros(self.n_act, dtype=np.float32)
        self.target_q = self.default_angles.copy()
        self.cmd = np.zeros(3, dtype=np.float32)
        self.counter = 0
        self.state = SimState()
        self.trajectory = [(x, y)]
        self._pending_events = list(self.scene.events)
        self.arm_target = np.zeros(len(self.upper_names))
        self._arm_motion = None
        self.pressed_buttons = set()
        self.mode = "walk"
        self.squat_height = self.squat_target = 0.78
        self._prop_motions = []
        self.held = None

    def grasp(self, name: str):
        """物を両手の間に固定して持つ（つかむ動作は簡略化）
        持っている間は LiDAR と衝突の対象から外す"""
        gid = self.model.geom(f"prop{self._prop_index[name]}").id
        m = self.model
        self.held = (name, gid, int(m.geom_group[gid]), int(m.geom_contype[gid]), int(m.geom_conaffinity[gid]))
        m.geom_group[gid], m.geom_contype[gid], m.geom_conaffinity[gid] = 0, 0, 0
        self.state.events.append((self.time, f"{name}を持ち上げた"))

    def release(self, pos):
        name, gid, group, contype, conaff = self.held
        m = self.model
        m.geom_pos[gid] = pos
        m.geom_group[gid], m.geom_contype[gid], m.geom_conaffinity[gid] = group, contype, conaff
        self.held = None
        mujoco.mj_forward(m, self.data)
        self.state.events.append((self.time, f"{name}を置いた"))
        return name

    def _update_held(self):
        gid = self.held[1]
        a, b = self.site_pos("left_hand_tip"), self.site_pos("right_hand_tip")
        self.model.geom_pos[gid] = (a + b) / 2 - np.array([0, 0, 0.02])
        _, _, yaw = self.pose()
        self.model.geom_quat[gid] = _yaw_quat(yaw)

    def set_mode(self, mode: str):
        """歩行（walk）としゃがむ（squat）のポリシーを切り替える（止まっている間に呼ぶ）"""
        if mode == self.mode:
            return
        if mode == "squat":
            from .squat_policy import SquatPolicy

            if getattr(self, "_squat", None) is None:
                self._squat = SquatPolicy()
                m, meta = self.model, self._squat.meta
                self._sq_obs_q = np.array([m.joint(n).qposadr[0] for n in meta["obs_joints"]])
                self._sq_obs_v = np.array([m.joint(n).dofadr[0] for n in meta["obs_joints"]])
                self._sq_default_obs = np.array([meta["default"].get(n, 0.0) for n in meta["obs_joints"]])
                self._sq_default_act = np.array([meta["default"].get(n, 0.0) for n in meta["policy_joints"]])
                self._sq_waist = self.upper_names.index("waist_yaw_joint")
            self._squat.reset()
            self.squat_height = self.squat_target = self._squat.meta["stand_height"]
        else:
            self.policy.reset()
            self.action = np.zeros(self.n_act, dtype=np.float32)
            self.target_q = self.default_angles.copy()
            self.arm_target[self._sq_waist] = 0.0
        self.mode = mode
        self.state.events.append((self.time, f"{'しゃがむ' if mode == 'squat' else '歩行'}ポリシーに切り替え"))

    def _squat_step(self):
        """しゃがむポリシーを1回動かす（50 Hz）"""
        d, sq, meta = self.data, self._squat, self._squat.meta
        dh = self.squat_target - self.squat_height
        self.squat_height += float(np.clip(dh, -0.008, 0.008))
        obs = np.concatenate([
            d.qvel[3:6] * meta["ang_vel_scale"], _gravity_orientation(d.qpos[3:7]), [self.squat_height],
            d.qpos[self._sq_obs_q] - self._sq_default_obs, d.qvel[self._sq_obs_v] * meta["joint_vel_scale"],
            sq.last_action])
        act = sq(obs)
        target = self._sq_default_act + act * np.asarray(meta["action_scale"])
        self.target_q = target[:12].astype(np.float32)
        self.arm_target[self._sq_waist] = target[12]

    def animate_prop(self, name: str, to_pos, duration: float = 1.5):
        gid = self.model.geom(f"prop{self._prop_index[name]}").id
        self._prop_motions.append((gid, self.time, duration, self.model.geom_pos[gid].copy(), np.asarray(to_pos, float)))

    def on_button_pressed(self, name: str):
        if name == "エレベーターのボタン":
            for door, dx in (("エレベーターの扉（左）", -0.55), ("エレベーターの扉（右）", 0.55)):
                p = self.prop_pos(door)
                self.animate_prop(door, (p[0] + dx, p[1] + 0.06, p[2]), 1.6)
            self.state.events.append((self.time, "エレベーターの扉が開いた"))
            return "エレベーターが到着し、扉が開きます。"
        return None

    def move_arms(self, targets: dict, duration: float = 0.8):
        """上半身の関節を duration 秒かけて目標角へ動かす"""
        if not self.arms:
            return
        goal = self.arm_target.copy()
        for name, v in targets.items():
            if name in self.upper_names:
                goal[self.upper_names.index(name)] = v
        start = self.arm_target.copy()
        self._arm_motion = (self.time, max(duration, 1e-3), start, goal)

    def solve_arm_ik(self, side: str, target, iters: int = 200):
        """指先を target に届かせる関節角を減衰付き最小二乗で求める
        戻り値: (関節角, 届かなかった距離[m])"""
        m = self.model
        names = [f"{side}_{j}_joint" for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                                               "wrist_roll", "wrist_pitch", "wrist_yaw")]
        qa = np.array([m.joint(n).qposadr[0] for n in names])
        va = np.array([m.joint(n).dofadr[0] for n in names])
        lo = np.array([m.jnt_range[m.joint(n).id][0] for n in names])
        hi = np.array([m.jnt_range[m.joint(n).id][1] for n in names])
        d = mujoco.MjData(m)
        d.qpos[:] = self.data.qpos
        sid = m.site(f"{side}_hand_tip").id
        jac = np.zeros((3, m.nv))
        target = np.asarray(target, dtype=float)
        for _ in range(iters):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            err = target - d.site_xpos[sid]
            if np.linalg.norm(err) < 0.005:
                break
            mujoco.mj_jacSite(m, d, jac, None, sid)
            J = jac[:, va]
            dq = J.T @ np.linalg.solve(J @ J.T + 0.01 * np.eye(3), err)
            d.qpos[qa] = np.clip(d.qpos[qa] + np.clip(dq, -0.2, 0.2), lo, hi)
        mujoco.mj_kinematics(m, d)
        return {n: float(d.qpos[a]) for n, a in zip(names, qa)}, float(np.linalg.norm(target - d.site_xpos[sid]))

    def site_pos(self, name: str):
        return self.data.site_xpos[self.model.site(name).id].copy()

    def _update_arm_target(self):
        t0, T, start, goal = self._arm_motion
        u = min(1.0, (self.time - t0) / T)
        u = u * u * u * (10 - 15 * u + 6 * u * u)
        self.arm_target = start + (goal - start) * u
        if u >= 1.0:
            self._arm_motion = None

    def prop_pos(self, name: str):
        return self.model.geom_pos[self.model.geom(f"prop{self._prop_index[name]}").id].copy()

    def _check_events(self):
        x, y, _ = self.pose()
        for ev in list(self._pending_events):
            x0, x1, y0, y1 = ev.trigger_box
            if x0 <= x <= x1 and y0 <= y <= y1:
                gid = self.model.geom(f"prop{self._prop_index[ev.prop]}").id
                self.model.geom_pos[gid] = ev.pos
                mujoco.mj_forward(self.model, self.data)
                self._pending_events.remove(ev)
                self.state.events.append((self.time, f"できごと: {ev.prop} が現れた"))

    @property
    def time(self) -> float:
        return self.data.time

    def pose(self):
        q = self.data.qpos
        return float(q[0]), float(q[1]), _quat_to_yaw(q[3:7])

    def lidar(self) -> np.ndarray:
        x, y, yaw = self.pose()
        origin = np.array([x, y, self.lidar_height])
        out = np.full(len(self.lidar_angles), self.lidar_range)
        for i, a in enumerate(self.lidar_angles):
            vec = np.array([math.cos(yaw + a), math.sin(yaw + a), 0.0])
            d = mujoco.mj_ray(self.model, self.data, origin, vec, self._geomgroup, 1, -1, self._geomid)
            if 0 <= d < self.lidar_range:
                out[i] = d
        return out

    def feet_on_ground(self) -> bool:
        """両足が接地しているか"""
        m, d = self.model, self.data
        if not hasattr(self, "_foot_bodies"):
            self._foot_bodies = (m.body("left_ankle_roll_link").id, m.body("right_ankle_roll_link").id)
        touching = set()
        for c in d.contact[:d.ncon]:
            for g in (c.geom1, c.geom2):
                b = int(m.geom_bodyid[g])
                if b in self._foot_bodies:
                    touching.add(b)
        return len(touching) == 2

    def is_fallen(self) -> bool:
        g = _gravity_orientation(self.data.qpos[3:7])
        return self.data.qpos[2] < 0.5 or g[2] > -0.6

    def _apply_safety(self, cmd: np.ndarray, scan: np.ndarray) -> np.ndarray:
        """進行方向に障害物があれば並進を止める"""
        v = cmd[:2]
        speed = float(np.linalg.norm(v))
        if speed < 1e-3:
            self.state.safety_active = False
            return cmd
        heading = math.atan2(v[1], v[0])
        diff = np.abs((self.lidar_angles - heading + np.pi) % (2 * np.pi) - np.pi)
        sector = scan[diff <= math.radians(self.safety.sector_deg)]
        nearest = float(sector.min()) if len(sector) else self.lidar_range
        out = cmd.copy()
        if nearest < self.safety.stop_distance:
            out[:2] = 0.0
            if not self.state.safety_active:
                self.state.safety_stops += 1
                self.state.events.append((self.time, f"安全停止: 進行方向 {nearest:.2f}m に障害物"))
            self.state.safety_active = True
        else:
            self.state.safety_active = False
            if nearest < self.safety.slow_distance:
                ratio = (nearest - self.safety.stop_distance) / (self.safety.slow_distance - self.safety.stop_distance)
                out[:2] *= max(0.3, ratio)
        return out

    def step(self, cmd, duration: float):
        """速度指令 (vx, vy, wz) を duration 秒与えてシミュレーションを進める"""
        cmd = np.clip(np.asarray(cmd, dtype=np.float32), -self.max_cmd, self.max_cmd)
        n = int(round(duration / self.model.opt.timestep))
        m, d = self.model, self.data
        for _ in range(n):
            if self.counter % self.decimation == 0:
                scan = self.lidar() if (self.counter // self.decimation) % 5 == 0 or not hasattr(self, "_scan") else self._scan
                self._scan = scan
                self.cmd = self._apply_safety(cmd, scan)
                if self.mode == "squat":
                    self._squat_step()
                else:
                    self._policy_step()
                if self.is_fallen() and not self.state.fallen:
                    self.state.fallen = True
                    self.state.events.append((self.time, "転倒を検知"))
            d.ctrl[self.leg_u] = (self.target_q - d.qpos[self.leg_q]) * self.kps - d.qvel[self.leg_v] * self.kds
            if self.arms:
                if self._arm_motion is not None:
                    self._update_arm_target()
                d.ctrl[self.up_u] = (self.arm_target - d.qpos[self.up_q]) * self.up_kp
            mujoco.mj_step(m, d)
            self.counter += 1
            if self.counter % 50 == 0:
                self.trajectory.append((float(d.qpos[0]), float(d.qpos[1])))
                if self._pending_events:
                    self._check_events()
            if self.held is not None and self.counter % 5 == 0:
                self._update_held()
            if self._prop_motions and self.counter % 5 == 0:
                for mo in list(self._prop_motions):
                    gid, t0, T, a, b = mo
                    u = min(1.0, (self.time - t0) / T)
                    self.model.geom_pos[gid] = a + (b - a) * (3 * u * u - 2 * u * u * u)
                    if u >= 1.0:
                        self._prop_motions.remove(mo)
            for cb in self.frame_callbacks:
                cb(self)

    def _policy_step(self):
        d = self.data
        obs = np.zeros(self.cfg["num_obs"], dtype=np.float32)
        phase = (self.counter * self.model.opt.timestep) % 0.8 / 0.8
        n = self.n_act
        obs[:3] = d.qvel[3:6] * self.cfg["ang_vel_scale"]
        obs[3:6] = _gravity_orientation(d.qpos[3:7])
        obs[6:9] = self.cmd * self.cmd_scale
        obs[9:9 + n] = (d.qpos[self.leg_q] - self.default_angles) * self.cfg["dof_pos_scale"]
        obs[9 + n:9 + 2 * n] = d.qvel[self.leg_v] * self.cfg["dof_vel_scale"]
        obs[9 + 2 * n:9 + 3 * n] = self.action
        obs[9 + 3 * n:9 + 3 * n + 2] = [math.sin(2 * math.pi * phase), math.cos(2 * math.pi * phase)]
        self.action = self.policy(obs)
        self.target_q = self.action * self.cfg["action_scale"] + self.default_angles
