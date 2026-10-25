"""ロビーの環境（場所・障害物・探し物）
AI には場所の座標だけを渡し、障害物と探し物の位置は渡さない（LiDAR とカメラで知覚させる）"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Location:
    name: str
    x: float
    y: float
    description: str
    color: tuple = (0.9, 0.9, 0.9, 0.35)


@dataclass
class Prop:
    """静的な物体
    size は MuJoCo の半径・半長（prism は底辺・奥行き・高さの半分）"""

    name: str
    shape: str
    pos: tuple
    size: tuple
    rgba: tuple
    obstacle: bool = True
    yaw_deg: float = 0.0
    tags: list = field(default_factory=list)
    text: str = ""
    # "mesh:<名前>" / "human" / "hidden"（assets/ がなければ単純な形で描く）
    visual: str = ""
    visual_yaw_deg: float = 0.0  # モデルの正面は -y 向き
    visual_scale: float | tuple = 1.0
    material: str = ""
    collide: bool = True


ROOM = dict(x_min=-2.5, x_max=10.5, y_min=-5.5, y_max=5.5, wall_h=0.9)

LOCATIONS = [
    Location("入口", 0.0, 0.0, "ロビーの入口。ロボットの待機位置。", (0.85, 0.85, 0.85, 0.35)),
    Location("受付", 6.0, 3.0, "受付カウンターの前。来客対応をする場所。", (0.95, 0.75, 0.3, 0.35)),
    Location("休憩スペース", 5.5, -3.0, "ソファとローテーブルがあるくつろぎスペース。", (0.4, 0.8, 0.5, 0.35)),
    Location("会議室", 9.2, 0.0, "会議室の扉の前。", (0.45, 0.6, 0.95, 0.35)),
    Location("充電ステーション", -1.2, 3.8, "ロボットの充電場所。", (0.95, 0.9, 0.3, 0.45)),
    Location("エレベーター前", 3.9, 4.4, "エレベーターの乗り場。扉の右に呼び出しボタンがある。", (0.75, 0.75, 0.8, 0.35)),
]

PROPS = [
    Prop("柱", "cylinder", (3.0, 1.5, 1.3), (0.3, 1.3), (0.75, 0.72, 0.68, 1), material="wall"),
    Prop("観葉植物", "cylinder", (3.4, -1.7, 0.45), (0.28, 0.45), (0.2, 0.55, 0.25, 1), visual="mesh:potted_plant_02"),
    Prop("受付カウンター", "box", (7.1, 3.0, 0.55), (0.35, 1.1, 0.55), (0.55, 0.38, 0.25, 1), material="wood_light"),
    # put_down で台の高さを測るため obstacle=True
    Prop("受付カウンターの天板", "box", (7.08, 3.0, 1.115), (0.4, 1.15, 0.015), (0.93, 0.92, 0.9, 1), collide=False),
    Prop("受付の人", "capsule", (7.85, 3.0, 0.67), (0.2, 0.45), (0.25, 0.35, 0.7, 1), tags=["person"],
         visual="human", visual_yaw_deg=180),
    Prop("受付の棚", "box", (8.75, 3.0, 0.34), (0.26, 1.22, 0.34), (0.5, 0.35, 0.2, 1), visual="mesh:modern_wooden_cabinet",
         visual_yaw_deg=-90),
    Prop("受付の看板", "box", (8.78, 3.0, 1.45), (0.02, 0.62, 0.2), (1, 1, 1, 1), obstacle=False, collide=False,
         text="受付  RECEPTION"),
    Prop("休憩スペースのカーペット", "box", (6.5, -3.9, 0.003), (1.8, 1.3, 0.003), (0.3, 0.32, 0.38, 1), obstacle=False,
         collide=False, material="carpet"),
    Prop("ソファ", "box", (6.6, -4.65, 0.35), (0.9, 0.4, 0.35), (0.35, 0.35, 0.45, 1), visual="mesh:sofa_02",
         visual_yaw_deg=180),
    Prop("ローテーブル", "box", (6.6, -3.5, 0.2), (0.6, 0.3, 0.2), (0.6, 0.5, 0.4, 1),
         visual="mesh:modern_coffee_table_01", visual_yaw_deg=90),
    Prop("肘掛け椅子", "box", (8.1, -3.5, 0.45), (0.42, 0.45, 0.45), (0.3, 0.3, 0.3, 1),
         visual="mesh:modern_arm_chair_01", visual_yaw_deg=-90),
    Prop("サイドテーブル", "box", (5.2, -4.75, 0.36), (0.36, 0.29, 0.36), (0.6, 0.45, 0.3, 1), visual="mesh:side_table_01",
         visual_scale=1.3),
    Prop("会議室の扉", "box", (10.4, 0.0, 1.05), (0.05, 0.6, 1.05), (0.45, 0.32, 0.22, 1), material="wood_dark"),
    Prop("会議室の看板", "box", (10.43, 0.0, 2.3), (0.02, 0.5, 0.14), (1, 1, 1, 1), obstacle=False, collide=False,
         text="会議室  MEETING ROOM"),
    Prop("待合の椅子1", "box", (9.95, 1.6, 0.45), (0.25, 0.22, 0.45), (0.3, 0.25, 0.2, 1), visual="mesh:dining_chair_02",
         visual_yaw_deg=-90),
    Prop("待合の椅子2", "box", (9.95, 2.3, 0.45), (0.25, 0.22, 0.45), (0.3, 0.25, 0.2, 1), visual="mesh:dining_chair_02",
         visual_yaw_deg=-90),
    Prop("入口の観葉植物", "cylinder", (-1.9, -1.6, 0.45), (0.3, 0.45), (0.2, 0.55, 0.25, 1), visual="mesh:potted_plant_02"),
    Prop("エレベーターの柱（左）", "box", (2.92, 5.44, 1.12), (0.06, 0.06, 1.12), (0.55, 0.56, 0.58, 1)),
    Prop("エレベーターの柱（右）", "box", (4.28, 5.44, 1.12), (0.06, 0.06, 1.12), (0.55, 0.56, 0.58, 1)),
    Prop("エレベーターの梁", "box", (3.6, 5.44, 2.17), (0.74, 0.06, 0.07), (0.55, 0.56, 0.58, 1), obstacle=False),
    Prop("エレベーターの中", "box", (3.6, 5.85, 1.05), (0.62, 0.35, 1.05), (0.12, 0.12, 0.14, 1), obstacle=False),
    Prop("エレベーターの扉（左）", "box", (3.3, 5.41, 1.05), (0.3, 0.025, 1.05), (0.78, 0.79, 0.8, 1)),
    Prop("エレベーターの扉（右）", "box", (3.9, 5.41, 1.05), (0.3, 0.025, 1.05), (0.74, 0.75, 0.77, 1)),
    Prop("エレベーターの操作盤", "box", (4.62, 5.43, 1.1), (0.07, 0.02, 0.14), (0.2, 0.2, 0.22, 1), obstacle=False),
    Prop("エレベーターのボタン", "box", (4.62, 5.405, 1.1), (0.03, 0.012, 0.03), (0.9, 0.9, 0.86, 1), obstacle=False,
         tags=["button"]),
    Prop("エレベーターの看板", "box", (3.6, 5.43, 2.4), (0.02, 0.6, 0.12), (1, 1, 1, 1), obstacle=False, collide=False,
         yaw_deg=90, text="エレベーター  ELEVATOR"),
]

OBJECTS = [
    Prop("赤い箱", "box", (4.3, -3.9, 0.15), (0.15, 0.15, 0.15), (0.9, 0.1, 0.1, 1), yaw_deg=20, tags=["red", "lost_item"]),
    Prop("青い箱", "box", (8.6, -1.6, 0.15), (0.15, 0.15, 0.15), (0.1, 0.3, 0.95, 1), yaw_deg=-10, tags=["blue"]),
    Prop("緑のボール", "cylinder", (1.2, -3.8, 0.12), (0.12, 0.12), (0.1, 0.8, 0.2, 1), tags=["green"]),
]

START_POSE = (0.0, 0.0, 0.0)


@dataclass
class Event:
    """ロボットが trigger_box に入ったら、物体 prop を pos に動かす"""

    prop: str
    pos: tuple
    trigger_box: tuple


@dataclass
class Scene:
    """シミュレーションに置く物の一式"""

    props: list
    objects: list
    events: list = field(default_factory=list)
    start_pose: tuple = START_POSE
    hidden_zones: list = field(default_factory=list)
    hidden_locations: list = field(default_factory=list)

    def all_props(self):
        return self.props + self.objects


def default_scene() -> Scene:
    import copy

    return Scene(copy.deepcopy(PROPS), copy.deepcopy(OBJECTS))


def location_by_name(name: str) -> Location | None:
    for loc in LOCATIONS:
        if loc.name == name or name in loc.name:
            return loc
    return None


def visible_locations(scene=None) -> list:
    hidden = set(getattr(scene, "hidden_locations", []) or [])
    return [l for l in LOCATIONS if l.name not in hidden]


def map_description(locations=None) -> str:
    locations = LOCATIONS if locations is None else locations
    origin = "入口から奥へ" if any(l.name == "入口" for l in locations) else "部屋の西から東へ"
    lines = [f"座標系: x軸={origin}, y軸=左方向, 単位はメートル, 角度は反時計回りが正。"]
    for loc in locations:
        lines.append(f"- {loc.name}: ({loc.x:.1f}, {loc.y:.1f}) {loc.description}")
    lines.append(
        f"部屋の範囲: x={ROOM['x_min']}〜{ROOM['x_max']}, y={ROOM['y_min']}〜{ROOM['y_max']}。"
        "柱や家具などの障害物、探し物の位置は地図に載っていないので、scan と look で確認すること。"
    )
    return "\n".join(lines)
