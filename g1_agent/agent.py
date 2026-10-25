"""AI が tool use で G1 を動かすエージェント（観測 → 計画 → 実行 → 確認のループ）"""
from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass, field

from . import world
from .skills import Robot

DEFAULT_MODEL = os.environ.get("G1_CLAUDE_MODEL", "claude-sonnet-5-5")
KEEP_IMAGES = 2

def build_system_prompt(locations=None) -> str:
    return f"""あなたはオフィスのロビーで働くヒューマノイドロボット Unitree G1 の頭脳です。
人からの日本語の指示を、用意されたツールを順に呼び出して実行してください。

# ロボットについて
- 二足歩行で移動できます。腕でジェスチャー（手を振る・場所を指差す・両手で迎える）とボタン押しができます。
  テーブルの上などにある小さな箱は、両手ではさんで持ち上げて運べます（腰を曲げられないので床の物は拾えません）。
- 低レベルの歩行は学習済みの歩行ポリシーが担当し、あなたは行き先や行動を決めます。
- 進行方向に障害物があると、安全装置があなたの指示に関係なく停止させます。

# 地図（既知の場所）
{world.map_description(locations)}

# 行動の原則
1. 指示を短い手順に分け、1つずつツールで実行して結果を確認する。結果を見ずに成功したと言わない。
2. walk_to が「安全停止」で失敗したら、結果に含まれる各方向の障害物距離を見て空いている側へ walk で
   0.8〜1.0m ほど横に避け、もう一度 walk_to する。同じ失敗を3回繰り返したら別の経路か中止を検討する。
3. 何があるかを確認するときは look で頭部カメラの画像を見る。赤・青・緑・黄の物体は find_object と
   approach_object が使える。地図にない人や物に近づくときは、look の画像で対象の足元を指して locate_point を呼び、
   返ってきた approach_position へ walk_to する。受付の人のように地図の場所にいる人には、その場所へ walk_to すれば十分。見つからなければ turn で 60度ずつ向きを変えて探す。
   案内板などの文字も look の画像から読み取れる。
4. 人に何かを伝える指示では say を使う。丁寧で短い日本語で話す。挨拶では gesture の wave、
   場所を聞かれたときの案内では gesture の point（その場所を指差す）を say の前に行い、身ぶりと言葉を組み合わせる。
   「〜に行って」「〜まで案内して」のように移動を求められたら、指差しで済ませずに実際に walk_to で移動する。
   指差す場所が体の後ろ側にあるときは、先に turn でその方向へ体を向ける。
5. エレベーターなどのボタンを押すときは press_button を使う（ボタンの前まで自動で移動して押す）。
   物を届けるときは、見つけた物を pick_up で持ち上げ（物の前まで自動で移動する）、届け先の場所へ walk_to し、
   put_down で目の前の台（受付カウンターなど）に置く。
6. 転倒を検知したら、それ以上移動せずに状況を報告する。
7. すべて終わったら、何をして何が分かったかを日本語で報告する（このときはツールを呼ばない）。
   報告は動画の字幕に出るので、1〜2文・60字以内に短くまとめ、座標などの細かい数値は必要なときだけ入れる。
"""


SYSTEM_PROMPT = build_system_prompt()

TOOLS = [
    {"name": "get_status", "description": "現在位置・向き・最寄りの場所・転倒の有無を取得する。",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "scan_surroundings",
     "description": "仮想LiDARで、前・左前・左・左後ろ・後ろ・右後ろ・右・右前の8方向それぞれについて最も近い障害物までの距離[m]を取得する。",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "look", "description": "頭部カメラの画像を取得して見る。人や物、周囲の様子を確認するときに使う。",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "walk_to",
     "description": "地図上の場所の名前、または座標(x, y)まで歩く。障害物で安全停止した場合は失敗として理由が返る。",
     "input_schema": {"type": "object", "properties": {
         "location": {"type": "string", "enum": [l.name for l in world.LOCATIONS], "description": "地図上の場所の名前"},
         "x": {"type": "number", "description": "目標のx座標[m]（location を使わない場合）"},
         "y": {"type": "number", "description": "目標のy座標[m]（location を使わない場合）"}}}},
    {"name": "walk",
     "description": "体の向きを変えずに、前・後ろ・左・右へ指定した距離だけ歩く。障害物を横に避けるときに使う。",
     "input_schema": {"type": "object", "properties": {
         "direction": {"type": "string", "enum": ["forward", "backward", "left", "right"]},
         "distance_m": {"type": "number", "minimum": 0.1, "maximum": 3.0}},
         "required": ["direction", "distance_m"]}},
    {"name": "turn", "description": "その場で旋回する。正の角度で左（反時計回り）、負の角度で右。",
     "input_schema": {"type": "object", "properties": {
         "angle_deg": {"type": "number", "minimum": -180, "maximum": 180}}, "required": ["angle_deg"]}},
    {"name": "find_object",
     "description": "頭部カメラの画像から指定した色の物体を画像処理で検出し、方向と距離、推定位置を返す。",
     "input_schema": {"type": "object", "properties": {
         "color": {"type": "string", "enum": ["red", "blue", "green", "yellow"]}}, "required": ["color"]}},
    {"name": "locate_point",
     "description": "最後に look した画像の中の1点を指定すると、その点の床の上の位置(x, y)と、"
                    "その約0.9m手前の立ち位置(approach_position)を深度画像から求める。人や色以外の物に近づくときに、"
                    "look の画像で対象の位置を指し、この結果の approach_position へ walk_to する。",
     "input_schema": {"type": "object", "properties": {
         "x": {"type": "number", "minimum": 0, "maximum": 1000, "description": "画像の横位置（左端0〜右端1000）"},
         "y": {"type": "number", "minimum": 0, "maximum": 1000,
               "description": "画像の縦位置（上端0〜下端1000）。人や物の足元（床に接する所）を指すと正確"}},
         "required": ["x", "y"]}},
    {"name": "approach_object",
     "description": "指定した色の物体を検出して、その約0.9m手前まで歩き、物体の方を向く。物体が視野にない場合は失敗する。",
     "input_schema": {"type": "object", "properties": {
         "color": {"type": "string", "enum": ["red", "blue", "green", "yellow"]}}, "required": ["color"]}},
    {"name": "gesture",
     "description": "腕でジェスチャーをする。姿勢は次に移動するまで保つので、gesture のあとに say すると身ぶりをしながら話せる。"
                    "wave=手を振る、point=指定した場所や座標の方向を指差す（location か x,y が必要）、"
                    "welcome=両手を広げて迎える、rest=腕を下ろす。",
     "input_schema": {"type": "object", "properties": {
         "kind": {"type": "string", "enum": ["wave", "point", "welcome", "rest"]},
         "location": {"type": "string", "enum": [l.name for l in world.LOCATIONS], "description": "point で指差す場所"},
         "x": {"type": "number", "description": "point で指差す座標x[m]"},
         "y": {"type": "number", "description": "point で指差す座標y[m]"}},
         "required": ["kind"]}},
    {"name": "press_button",
     "description": "壁のボタンの前まで移動し、右手の指先でボタンを押す。押せたかどうかと、押した結果（扉が開くなど）が返る。",
     "input_schema": {"type": "object", "properties": {
         "name": {"type": "string", "enum": ["エレベーターのボタン"]}}, "required": ["name"]}},
    {"name": "pick_up",
     "description": "指定した色の小さな物（忘れ物の箱など）を探し、その前まで移動して両手で持ち上げる。"
                    "物が視野にない場合や、床の上など低い所にある場合は失敗する。持ったまま walk_to などで移動できる。",
     "input_schema": {"type": "object", "properties": {
         "color": {"type": "string", "enum": ["red", "blue", "green", "yellow"]}}, "required": ["color"]}},
    {"name": "put_down",
     "description": "持っている物を、近くの台（受付カウンターやテーブルなど、高さ0.6〜1.3m）の上に置く。"
                    "台の方へ向き直って近づいてから置く。",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "say", "description": "近くの人に向けて日本語で話す。",
     "input_schema": {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]}},
]
TOOLS[-1]["cache_control"] = {"type": "ephemeral"}


def build_tools(locations):
    """場所の選択肢を locations に絞ったツール定義"""
    import copy

    tools = copy.deepcopy(TOOLS)
    names = [l.name for l in locations]
    for t in tools:
        loc = t["input_schema"]["properties"].get("location")
        if loc and "enum" in loc:
            loc["enum"] = names
    return tools

TOOL_LABEL = {"get_status": "状態確認", "scan_surroundings": "周囲をスキャン", "look": "カメラで確認",
              "walk_to": "移動", "walk": "横移動", "turn": "旋回", "find_object": "物体検出",
              "approach_object": "物体に接近", "say": "発話", "locate_point": "位置を特定",
              "gesture": "ジェスチャー", "press_button": "ボタンを押す", "pick_up": "持ち上げる",
              "put_down": "置く"}


@dataclass
class RunLog:
    instruction: str
    model: str
    steps: list = field(default_factory=list)
    api_calls: list = field(default_factory=list)
    final_text: str = ""
    success: bool | None = None
    sim_time_s: float = 0.0
    wall_time_s: float = 0.0

    def to_dict(self):
        return self.__dict__


def make_client(spec: str):
    """モデルの指定から (client, モデル名) を作る
    mock / gemini[:ID] / local[:名前] / それ以外は Claude API のモデル名"""
    if spec == "gemini" or spec.startswith("gemini:") or spec.startswith("gemini-"):
        from .gemini_llm import DEFAULT_GEMINI_MODEL, GeminiClient

        name = spec.split(":", 1)[1] if ":" in spec else (spec if spec.startswith("gemini-") else DEFAULT_GEMINI_MODEL)
        return GeminiClient(), name
    if spec.startswith("replay:"):
        from .replay_llm import ReplayClient

        client = ReplayClient.from_file(spec.split(":", 1)[1])
        return client, client.model
    if spec.startswith("mock"):
        from .mock_llm import MockClient

        return MockClient(), "mock-rule-based"
    if spec == "local" or spec.startswith("local:"):
        from .local_llm import DEFAULT_LOCAL_MODEL, LocalVLMClient

        return LocalVLMClient(), spec.split(":", 1)[1] if ":" in spec else DEFAULT_LOCAL_MODEL
    import anthropic

    return anthropic.Anthropic(), spec


def speaker_name(model: str) -> str:
    if model.startswith("mock"):
        return "ルールベース"
    if model.startswith("claude"):
        return "Claude"
    if model.startswith("gemini"):
        if "argon" in model:
            return "Gemini 4 Argon"
        if "robotics-er-2" in model:
            return "Gemini Robotics-ER 2"
        return "Gemini " + model.split("-", 1)[1].replace("-", " ").title() if "-" in model else "Gemini"
    if "qwen" in model.lower():
        return "Qwen3-VL" if "vl" in model.lower() else "Qwen"
    return model.split("/")[-1]


class G1Agent:
    def __init__(self, robot: Robot, client=None, model: str = DEFAULT_MODEL, recorder=None, viewer=None,
                 max_turns: int = 30, verbose: bool = True):
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client, self.robot, self.model = client, robot, model
        self.recorder, self.viewer = recorder, viewer
        self.max_turns, self.verbose = max_turns, verbose
        self.memory = []
        self.pretty = False
        self.speaker = speaker_name(model)
        scene = getattr(robot.sim, "scene", None)
        if getattr(scene, "hidden_locations", None):
            locs = world.visible_locations(scene)
            self.system_prompt, self.tools = build_system_prompt(locs), build_tools(locs)
        else:
            self.system_prompt, self.tools = SYSTEM_PROMPT, TOOLS
        if recorder:
            recorder.speaker = self.speaker

    def _caption(self, text):
        if self.verbose:
            print(text, flush=True)
        if self.recorder:
            self.recorder.caption = text

    @staticmethod
    def _describe_call(name, args):
        label = TOOL_LABEL.get(name, name)
        if name == "walk_to":
            arg = args.get("location") or f"({args.get('x')}, {args.get('y')})"
        elif name == "walk":
            arg = f"{args.get('direction')} {args.get('distance_m')}m"
        elif name == "turn":
            arg = f"{args.get('angle_deg')}度"
        elif name in ("find_object", "approach_object", "pick_up"):
            arg = {"red": "赤", "blue": "青", "green": "緑", "yellow": "黄"}.get(args.get("color"), args.get("color"))
        elif name == "say":
            arg = "「" + args.get("message", "")[:20] + "…」"
        elif name == "locate_point":
            arg = f"画像の ({args.get('x'):.0f}, {args.get('y'):.0f})"
        elif name == "gesture":
            kinds = {"wave": "手を振る", "point": "指差し", "welcome": "両手で迎える", "rest": "腕を下ろす"}
            arg = kinds.get(args.get("kind"), args.get("kind"))
            if args.get("kind") == "point":
                arg += "：" + (args.get("location") or f"({args.get('x')}, {args.get('y')})")
        elif name == "press_button":
            arg = args.get("name", "")
        else:
            arg = ""
        return f"▶ {label} {name}({arg})"

    SHOT_OF_TOOL = {"walk_to": "follow", "walk": "follow", "turn": "follow", "approach_object": "follow",
                    "say": "front", "gesture": "front", "press_button": "follow", "pick_up": "front", "put_down": "front",
                    "look": "head", "find_object": "head", "locate_point": "head"}

    def _execute(self, name, args):
        r = self.robot
        image = None
        if self.recorder and name in self.SHOT_OF_TOOL:
            self.recorder.shot = self.SHOT_OF_TOOL[name]
        if name == "get_status":
            res = r.get_status()
        elif name == "scan_surroundings":
            res = r.scan_surroundings()
        elif name == "look":
            res, image = r.look()
        elif name == "walk_to":
            res = r.walk_to(location=args.get("location"), x=args.get("x"), y=args.get("y"))
        elif name == "walk":
            res = r.walk(args["direction"], args["distance_m"])
        elif name == "turn":
            res = r.turn(args["angle_deg"])
        elif name == "find_object":
            res, pos = r.find_object(args["color"])
            if pos and self.recorder:
                self.recorder.found.append((pos[0], pos[1], args["color"]))
        elif name == "gesture":
            res = r.gesture(args.get("kind", ""), location=args.get("location"), x=args.get("x"), y=args.get("y"))
        elif name == "press_button":
            res = r.press_button(args.get("name", "エレベーターのボタン"))
        elif name == "pick_up":
            res = r.pick_up(args.get("color", "red"))
        elif name == "put_down":
            res = r.put_down()
        elif name == "locate_point":
            res = r.locate_point(args["x"], args["y"])
            if res.get("point_position") and self.recorder:
                pp = res["point_position"]
                self.recorder.found.append((pp["x"], pp["y"], "point"))
        elif name == "approach_object":
            res = r.approach_object(args["color"])
            if res.get("object_position") and self.recorder:
                op = res["object_position"]
                self.recorder.found.append((op["x"], op["y"], args["color"]))
        elif name == "say":
            dur = None
            if self.recorder:
                self.recorder.caption = ""
                dur = self.recorder.speak_begin(args["message"])
            res = r.say(args["message"], duration=None if dur is None else dur + 0.3)
            if self.recorder:
                self.recorder.speak_end()
        else:
            res = {"success": False, "message": f"未知のツール {name}"}
        if self.recorder and self.recorder.shot == "head":
            self.recorder.hold(1.5, thinking=False)
        return res, image

    @staticmethod
    def _tool_result(tool_id, res, image):
        content = [{"type": "text", "text": json.dumps(res, ensure_ascii=False)}]
        if image is not None:
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                        "data": base64.b64encode(image).decode()}})
        return {"type": "tool_result", "tool_use_id": tool_id, "content": content,
                **({"is_error": True} if res.get("success") is False and res.get("fallen") else {})}

    @staticmethod
    def _prune_images(messages):
        """古いカメラ画像を文字に置き換える"""
        seen = 0
        for msg in reversed(messages):
            if msg["role"] != "user" or not isinstance(msg["content"], list):
                continue
            for block in msg["content"]:
                if block.get("type") != "tool_result":
                    continue
                for i, c in enumerate(block["content"]):
                    if c.get("type") == "image":
                        seen += 1
                        if seen > KEEP_IMAGES:
                            block["content"][i] = {"type": "text", "text": "（古いカメラ画像は省略）"}

    def run(self, instruction: str) -> RunLog:
        log = RunLog(instruction, self.model)
        t_wall, t_sim = time.time(), self.robot.sim.time
        if self.recorder:
            self.recorder.instruction = instruction
            self.recorder.caption = ""
            self.recorder.shot = "overview"
            self.recorder.hold(1.5)
        content = instruction
        if self.memory:
            notes = "\n".join(f"- 指示「{i}」→ 報告「{r}」" for i, r in self.memory[-4:])
            content = f"{instruction}\n\n（参考: これまでの作業の記録）\n{notes}"
        messages = [{"role": "user", "content": content}]

        for turn in range(self.max_turns):
            self._prune_images(messages)
            t0 = time.time()
            if self.verbose:
                print(f"   … {self.speaker} の返事を待っています（{turn + 1} 回目）", flush=True)
            if self.recorder:
                self.recorder.hold(0.6)
            resp = self.client.messages.create(
                model=self.model, max_tokens=1024,
                system=[{"type": "text", "text": self.system_prompt, "cache_control": {"type": "ephemeral"}}],
                tools=self.tools, messages=messages)
            latency = time.time() - t0
            if self.verbose:
                print(f"   … {latency:.1f} 秒で返事が来ました", flush=True)
            if self.viewer:
                self.viewer.resync()
            u = resp.usage
            log.api_calls.append({"latency_s": round(latency, 2), "input_tokens": u.input_tokens,
                                  "output_tokens": u.output_tokens,
                                  "cache_read": getattr(u, "cache_read_input_tokens", 0) or 0,
                                  "cache_write": getattr(u, "cache_creation_input_tokens", 0) or 0})
            messages.append({"role": "assistant", "content": resp.content})
            results = []
            for block in resp.content:
                if block.type == "text" and block.text.strip():
                    self._caption(f"{self.speaker}: " + block.text.strip())
                    if resp.stop_reason == "end_turn":
                        log.final_text = block.text.strip()
                    elif self.recorder:
                        self.recorder.hold(1.2, thinking=False)
                elif block.type == "tool_use":
                    self._caption(self._describe_call(block.name, block.input))
                    res, image = self._execute(block.name, block.input)
                    if self.verbose and self.pretty:
                        mark = "✗" if res.get("success") is False else "→"
                        print(f"     {mark} {res.get('message', '')}", flush=True)
                    elif self.verbose:
                        print("   ->", json.dumps(res, ensure_ascii=False)[:300], flush=True)
                    log.steps.append({"tool": block.name, "input": block.input, "result": res,
                                      "sim_time": round(self.robot.sim.time, 1)})
                    results.append(self._tool_result(block.id, res, image))
            if resp.stop_reason != "tool_use":
                break
            messages.append({"role": "user", "content": results})

        if self.recorder:
            self.recorder.hold(2.5, thinking=False)
        log.sim_time_s = round(self.robot.sim.time - t_sim, 1)
        log.wall_time_s = round(time.time() - t_wall, 1)
        self.memory.append((instruction, log.final_text or "（報告なし）"))
        return log


G1ClaudeAgent = G1Agent


def summarize_usage(log: RunLog) -> dict:
    calls = log.api_calls
    return {"api_calls": len(calls),
            "tool_calls": len(log.steps),
            "mean_latency_s": round(sum(c["latency_s"] for c in calls) / max(1, len(calls)), 2),
            "input_tokens": sum(c["input_tokens"] for c in calls),
            "cache_read_tokens": sum(c["cache_read"] for c in calls),
            "output_tokens": sum(c["output_tokens"] for c in calls),
            "safety_retries": sum(1 for s in log.steps if s["tool"] == "walk_to" and not s["result"]["success"])}
