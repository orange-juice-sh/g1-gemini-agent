"""評価タスク（指示・シーン・成功条件）
成功はシミュレーションの最終状態とログから判定する"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable

from . import world
from .world import Event, Prop, Scene


def person(name, x, y, shirt, yaw=180):
    return [Prop(name, "capsule", (x, y, 0.67), (0.2, 0.45), shirt, tags=["person"], visual="human", visual_yaw_deg=yaw)]


def scene_people():
    s = world.default_scene()
    s.props += person("オレンジの服の来客", 2.2, -2.6, (0.95, 0.5, 0.1, 1))
    s.props += person("紫の服の来客", 8.0, 2.2, (0.5, 0.25, 0.65, 1))
    return s


def scene_sign():
    s = world.default_scene()
    text = "本日の打ち合わせは\n休憩スペースで行います"
    s.props += [Prop("案内板", "box", (2.3, -0.7, 1.15), (0.015, 0.55, 0.32), (1, 1, 1, 1), obstacle=False, text=text),
                Prop("案内板（裏）", "box", (2.332, -0.7, 1.15), (0.015, 0.55, 0.32), (1, 1, 1, 1), obstacle=False,
                     yaw_deg=180, text=text),
                Prop("案内板の支柱", "cylinder", (2.33, -0.7, 0.45), (0.04, 0.45), (0.3, 0.3, 0.32, 1))]
    return s


def scene_count():
    s = world.default_scene()
    s.objects += [Prop("黄色い箱", "box", (5.75, -3.95, 0.12), (0.12, 0.12, 0.12), (0.95, 0.85, 0.15, 1), yaw_deg=35),
                  Prop("白い箱", "box", (6.4, -2.5, 0.12), (0.12, 0.12, 0.12), (0.92, 0.92, 0.9, 1), yaw_deg=-15)]
    return s


def scene_partition():
    s = world.default_scene()
    s.props.append(Prop("仕切り", "box", (5.0, 0.0, -5.0), (0.06, 2.2, 0.6), (0.55, 0.6, 0.68, 1)))
    s.events.append(Event("仕切り", (5.0, 0.0, 0.6), trigger_box=(1.5, 4.0, -3.0, 3.0)))
    return s


def put_red_box_on_table(s):
    """赤い忘れ物の箱を休憩スペースのサイドテーブルの上に置く"""
    red = next(o for o in s.objects if o.name == "赤い箱")
    red.pos, red.size, red.yaw_deg = (5.2, -4.55, 0.716 + 0.07), (0.07, 0.07, 0.07), 0.0
    return s


SHELF_UPPER_TOP = 0.865
SHELF_LOWER_TOP = 0.435


def add_shelf_with_red_box(s):
    """休憩スペースに2段の棚を置く（上の段に赤い箱、下の段に青いボールと黄色い置物）"""
    cx, back, depth, width = 3.9, -5.45, 0.40, 1.30
    cy = back + depth / 2
    wood = dict(obstacle=False, collide=False, material="wood_light", rgba=(0.62, 0.45, 0.3, 1))
    s.props += [
        # 当たり判定は棚の下部だけにし、前面を 0.15m 奥にして、しゃがんだ膝が当たらないようにする
        Prop("棚", "box", (cx, back + (depth - 0.15) / 2, 0.20), (width / 2, (depth - 0.15) / 2, 0.20), (0.6, 0.45, 0.3, 1),
             visual="hidden"),
        Prop("棚の側板（左）", "box", (cx - width / 2 + 0.015, cy, 0.44), (0.015, depth / 2, 0.44), **wood),
        Prop("棚の側板（右）", "box", (cx + width / 2 - 0.015, cy, 0.44), (0.015, depth / 2, 0.44), **wood),
        Prop("棚の背板", "box", (cx, back + 0.01, 0.44), (width / 2, 0.01, 0.44), **wood),
        Prop("棚の天板", "box", (cx, cy, SHELF_UPPER_TOP - 0.015), (width / 2, depth / 2, 0.015), **wood),
        Prop("棚の中段", "box", (cx, cy, SHELF_LOWER_TOP - 0.015), (width / 2 - 0.03, depth / 2 - 0.01, 0.015), **wood),
        Prop("棚の底板", "box", (cx, cy, 0.03), (width / 2, depth / 2, 0.03), **wood),
    ]
    front = back + depth
    red = next(o for o in s.objects if o.name == "赤い箱")
    red.pos, red.size, red.yaw_deg = (cx, front - 0.08, SHELF_UPPER_TOP + 0.07), (0.07, 0.07, 0.07), 0.0
    s.objects = [o for o in s.objects if o.name != "青い箱"]
    s.objects += [
        Prop("青いボール", "sphere", (cx - 0.35, front - 0.09, SHELF_LOWER_TOP + 0.075), (0.075,), (0.12, 0.3, 0.9, 1),
             tags=["blue", "lost_item"]),
        Prop("黄色い三角の置物", "prism", (cx + 0.35, front - 0.09, SHELF_LOWER_TOP + 0.08), (0.09, 0.06, 0.08),
             (0.95, 0.82, 0.08, 1), tags=["yellow", "lost_item"]),
    ]
    return s


def scene_shelf():
    """撮影・ライブモード用のロビー（棚、低い受付カウンター、充電器）"""
    s = add_shelf_with_red_box(world.default_scene())
    carpet = next(p for p in s.props if p.name == "休憩スペースのカーペット")
    x0, x1, y0, y1 = 3.0, 8.3, -5.45, -2.6
    carpet.pos = ((x0 + x1) / 2, (y0 + y1) / 2, carpet.pos[2])
    carpet.size = ((x1 - x0) / 2, (y1 - y0) / 2, carpet.size[2])
    s.objects = [o for o in s.objects if o.name != "緑のボール"]
    # 床の目印は描かない（AI が青い円を「青い丸い物体」と取り違えるため）
    s.hidden_zones = [l.name for l in world.LOCATIONS]
    top = 0.85
    counter = next(p for p in s.props if p.name == "受付カウンター")
    counter.pos, counter.size = (counter.pos[0], counter.pos[1], (top - 0.03) / 2), (counter.size[0], counter.size[1], (top - 0.03) / 2)
    board = next(p for p in s.props if p.name == "受付カウンターの天板")
    board.pos = (board.pos[0], board.pos[1], top - 0.015)
    person = next(p for p in s.props if p.name == "受付の人")
    k = 0.78
    person.pos, person.size, person.visual_scale = (person.pos[0], person.pos[1], person.pos[2] * k), \
        tuple(v * k for v in person.size), k
    s.hidden_locations = ["入口"]
    # 探し物と取り違えないよう、鮮やかな色は使わない
    cs = world.location_by_name("充電ステーション")
    wall_y = world.ROOM["y_max"]
    s.props += [
        Prop("充電ステーションの床パネル", "box", (cs.x, cs.y, 0.006), (0.5, 0.42, 0.006), (0.22, 0.24, 0.27, 1),
             obstacle=False, collide=False),
        Prop("充電器", "box", (cs.x, wall_y - 0.2, 0.55), (0.28, 0.17, 0.55), (0.88, 0.89, 0.9, 1)),
        Prop("充電器の画面", "box", (cs.x, wall_y - 0.375, 0.82), (0.18, 0.008, 0.11), (0.08, 0.1, 0.13, 1),
             obstacle=False, collide=False),
        Prop("充電器の端子", "box", (cs.x, wall_y - 0.39, 0.45), (0.1, 0.02, 0.05), (0.35, 0.36, 0.38, 1),
             obstacle=False, collide=False),
        Prop("充電ステーションの看板", "box", (cs.x, wall_y - 0.07, 1.5), (0.02, 0.5, 0.13), (1, 1, 1, 1),
             obstacle=False, collide=False, yaw_deg=90, text="充電ステーション  CHARGING"),
    ]
    return s


def scene_lost_item():
    return put_red_box_on_table(world.default_scene())


def scene_demo():
    """demo シナリオ用（案内板あり）"""
    s = scene_lost_item()
    s.props += [p for p in scene_sign().props if p.name.startswith("案内板")]
    return s


def scene_moved_box():
    s = world.default_scene()
    red = next(o for o in s.objects if o.name == "赤い箱")
    red.pos = (1.2, 3.2, 0.15)
    return s


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def near_loc(name, r=0.8):
    loc = world.location_by_name(name)
    return lambda sim, log: _dist(sim.pose(), (loc.x, loc.y)) < r


def near_prop(name, r=1.3):
    return lambda sim, log: _dist(sim.pose(), sim.prop_pos(name)) < r


def said(pattern=None, near=None, r=1.5):
    """say を使ったか（pattern で内容、near で場所を確かめる）"""
    def check(sim, log):
        for s in log.steps:
            if s["tool"] != "say":
                continue
            if pattern and not re.search(pattern, s["input"].get("message", "")):
                continue
            if near is not None:
                p = s["result"]["position"]
                if _dist((p["x"], p["y"]), near) > r:
                    continue
            return True
        return False
    return check


def said_near_loc(name, pattern=None, r=1.5):
    loc = world.location_by_name(name)
    return said(pattern, (loc.x, loc.y), r)


def said_near_prop(name, pattern=None, r=1.6):
    def check(sim, log):
        p = sim.prop_pos(name)
        return said(pattern, (p[0], p[1]), r)(sim, log)
    return check


def pointed_then_said(name, pattern=None, max_err_deg=25):
    """場所を正しく指差し、移動せずに話したか"""
    loc = world.location_by_name(name)

    def check(sim, log):
        pointing = False
        for s in log.steps:
            t, inp, res = s["tool"], s["input"], s["result"]
            if t == "gesture" and inp.get("kind") == "point" and res.get("success"):
                tgt = (loc.x, loc.y) if inp.get("location") == loc.name else (inp.get("x"), inp.get("y"))
                ok_target = tgt[0] is not None and _dist(tgt, (loc.x, loc.y)) < 1.5
                pointing = ok_target and res.get("pointing_error_deg", 99) <= max_err_deg
            elif t in ("walk_to", "walk", "turn", "approach_object", "press_button") or \
                    (t == "gesture" and inp.get("kind") != "point"):
                pointing = False
            elif t == "say" and pointing and (not pattern or re.search(pattern, inp.get("message", ""))):
                return True
        return False
    return check


def gestured(kind):
    return lambda sim, log: any(s["tool"] == "gesture" and s["input"].get("kind") == kind and s["result"].get("success")
                                for s in log.steps)


def placed_on_counter(name="赤い箱"):
    def check(sim, log):
        p = sim.prop_pos(name)
        return sim.held is None and 6.7 < p[0] < 7.5 and 1.9 < p[1] < 4.1 and p[2] > 0.8
    return check


def button_pressed(name="エレベーターのボタン"):
    return lambda sim, log: name in sim.pressed_buttons


def reported(pattern):
    return lambda sim, log: bool(re.search(pattern, log.final_text or ""))


def all_of(*fs):
    return lambda sim, log: all(f(sim, log) for f in fs)


@dataclass
class Task:
    id: str
    kind: str
    text: str
    check: Callable
    scene: Callable = world.default_scene
    note: str = ""
    tags: list = field(default_factory=list)
    assist: bool = False


RED = world.OBJECTS[0].pos
BLUE = world.OBJECTS[1].pos
UKETSUKE = world.location_by_name("受付")

TASKS = [
    Task("T1", "移動", "受付まで行ってください。", near_loc("受付")),
    Task("T2", "移動", "会議室の前に行ってください。", near_loc("会議室")),
    Task("T3", "探索", "休憩スペースで赤い箱を探して、その前まで行ってください。", near_prop("赤い箱")),
    Task("T4", "探索", "青い箱がどこかにあるので、探してその前まで行ってください。", near_prop("青い箱")),
    Task("T5", "対話", "受付にいる人に「こんにちは」と挨拶してください。",
         all_of(near_loc("受付", 1.2), said_near_loc("受付"))),
    Task("T6", "対話", "お客さんを会議室まで案内します。会議室まで先導し、着いたら到着したことを伝えてください。",
         all_of(near_loc("会議室"), said_near_loc("会議室"))),
    Task("T7", "曖昧", "少し疲れたので、座って休める場所に行ってください。", near_loc("休憩スペース")),
    Task("T8", "曖昧", "バッテリーが少なくなってきました。", near_loc("充電ステーション")),
    Task("T9", "視覚", "オレンジ色の服を着たお客さんのところへ行き、「ご案内しますので少々お待ちください」と伝えてください。",
         all_of(near_prop("オレンジの服の来客", 1.6), said_near_prop("オレンジの服の来客")), scene_people,
         "色の道具（赤・青・緑）では探せない人を、画像で見分けて近づく（locate_point）"),
    Task("T10", "文字", "入口の近くにある案内板を読んで、本日の打ち合わせの場所へ行ってください。",
         near_loc("休憩スペース"), scene_sign, "日本語の案内板の文字を読み取り、行き先を決める"),
    Task("T11", "視覚", "休憩スペースの床に小さな箱がいくつあるか数えて、個数を報告してください。",
         reported(r"(3|３|三)\s*(個|つ)"), scene_count, "複数の物体を見て数える（ソファやテーブルは数えない）"),
    Task("T12", "変化", "会議室の前に行ってください。", near_loc("会議室"), scene_partition,
         "移動中に幅4.4mの仕切りが現れる。1m 横に避けるだけでは通れず、回り込む再計画が必要"),
    Task("T13", "変化", "休憩スペースに赤い忘れ物の箱があるはずです。探してその前まで行ってください。",
         near_prop("赤い箱"), scene_moved_box, "箱が聞いていた場所になく、探す範囲を広げる必要がある"),
    Task("T14", "長手順",
         "受付の人に「こんにちは」と挨拶してから、会議室の前に行って「会議室に到着しました」と伝え、最後に充電ステーションに戻ってください。",
         all_of(said_near_loc("受付", "こんにちは"), said_near_loc("会議室", "到着"), near_loc("充電ステーション")),
         note="3つの場所と2回の発話を順番どおりに実行する"),
    Task("T15", "探索", "緑のボールが床のどこかに落ちています。探してその前まで行ってください。",
         near_prop("緑のボール", 1.2), note="同じ緑色の観葉植物と見分ける必要がある"),
]

TASKS += [
    Task("T16", "腕", "会議室の場所を聞かれました。会議室の方向を指差しながら「会議室はあちらです」と伝えてください。",
         pointed_then_said("会議室", "あちら"), note="指差し（腕の IK）と発話を組み合わせて道案内する"),
    Task("T17", "腕", "入口から入ってきたお客さんに、手を振りながら「ようこそ」と挨拶してください。",
         all_of(gestured("wave"), said("ようこそ")), note="手を振る身ぶりと挨拶"),
    Task("T18", "腕", "お客さんが2階に行きたいそうです。エレベーターを呼んでください。",
         button_pressed(), note="人の高さに合わせた設備（呼び出しボタン）を指先で押す"),
    Task("T19", "腕",
         "受付に行き、受付の前から会議室の方向を指差して「会議室はあちらです」と伝えてください。",
         all_of(near_loc("受付", 1.2), pointed_then_said("会議室", "あちら")),
         note="移動後、体の向きを考えて指差す（会議室が体の後ろ側になることがある）"),
    Task("T20", "腕", "休憩スペースのテーブルの上に赤い忘れ物の箱があります。受付まで運んで、受付カウンターの上に置いてください。",
         placed_on_counter(), scene_lost_item, "探して両手で持ち上げ、運んで台の上に置く（つかむ動作は簡略化）"),
]

def placed_on_counter_obj(name):
    return placed_on_counter(name)


DEMO_TASKS = [
    Task("D1", "デモ", "休憩スペースにある赤い色の四角い物体を受付に持って行ってください。",
         placed_on_counter_obj("赤い箱"), scene_shelf, "色と形の言葉から、棚の上の3つのうち赤い箱を選んで運ぶ", assist=True),
    Task("D2", "デモ", "休憩スペースにある青色の丸い物体を受付に持って行ってください。",
         placed_on_counter_obj("青いボール"), scene_shelf, "色と形の言葉から、棚の上の3つのうち青いボールを選んで運ぶ", assist=True),
    Task("D3", "デモ", "充電スペースに戻ってください。", near_loc("充電ステーション"), scene_shelf,
         "地図の名前（充電ステーション）と違う言い方で場所を指示する", assist=True),
    Task("D4", "デモ", "エレベーターのスイッチを押して、再度充電スペースに戻ってください。",
         all_of(button_pressed(), near_loc("充電ステーション")), scene_shelf, "ボタンを押してから戻る2段階の指示", assist=True),
]
TASKS += DEMO_TASKS

TASKS_BY_ID = {t.id: t for t in TASKS}
