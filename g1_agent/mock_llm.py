"""API なしで動くルールベースの代役（評価のベースライン）"""
from __future__ import annotations

import json
import re
import uuid
from types import SimpleNamespace

from . import world

COLORS = {"赤": "red", "青": "blue", "緑": "green", "黄": "yellow"}


def _text(t):
    return SimpleNamespace(type="text", text=t)


def _tool(name, **args):
    return SimpleNamespace(type="tool_use", id="toolu_" + uuid.uuid4().hex[:12], name=name, input=args)


def _resp(blocks, stop):
    usage = SimpleNamespace(input_tokens=0, output_tokens=0, cache_read_input_tokens=0, cache_creation_input_tokens=0)
    return SimpleNamespace(content=blocks, stop_reason=stop, usage=usage)


class _Planner:
    def __init__(self, instruction: str):
        self.plan = []
        events = []
        for loc in world.LOCATIONS:
            for m in re.finditer(re.escape(loc.name), instruction):
                events.append((m.start(), "loc", loc.name))
        for ja, en in COLORS.items():
            m = re.search(ja + "[いの色]", instruction)
            if m:
                events.append((m.start(), "color", en))
        events.sort()
        for _, kind, val in events:
            if kind == "loc":
                self.plan.append(("walk_to", {"location": val}))
                if val == "受付" and re.search("確認|様子|誰か|人", instruction):
                    self.plan.append(("look", {}))
            else:
                self.plan.append(("find_object", {"color": val}))
                self.plan.append(("approach_object", {"color": val}))
        m = re.search("「(.+?)」", instruction)
        if m:
            self.plan.append(("say", {"message": m.group(1)}))
        elif re.search("挨拶|あいさつ", instruction):
            self.plan.append(("say", {"message": "こんにちは。ご用件があればお声がけください。"}))
        self.retries = 0
        self.search_turns = 0
        self.done = []

    def next(self, last_call, last_result):
        if last_call is not None:
            name, args = last_call
            ok = last_result.get("success")
            if name == "walk_to" and not ok and "安全停止" in last_result.get("message", "") and self.retries < 3:
                self.retries += 1
                obs = last_result.get("obstacles", {})
                dist = lambda k: obs.get(k, 9) if isinstance(obs.get(k, 9), (int, float)) else 9
                side = "left" if dist("左前") + dist("左") > dist("右前") + dist("右") else "right"
                self.plan[:0] = [("walk", {"direction": side, "distance_m": 0.9}), (name, args)]
            elif name == "find_object" and not ok and self.search_turns < 6:
                self.search_turns += 1
                self.plan[:0] = [("turn", {"angle_deg": 60}), (name, args)]
            elif name == "find_object" and ok:
                self.search_turns = 0
            elif name == "find_object" and not ok:
                self.plan = [p for p in self.plan if p[0] != "approach_object"]
            self.done.append((name, last_result))
        if self.plan:
            return self.plan.pop(0)
        return None

    def report(self):
        fails = [n for n, r in self.done if not r.get("success")]
        found = [r for n, r in self.done if n == "approach_object" and r.get("success")]
        text = f"指示された手順を {len(self.done)} 回の行動で実行しました。"
        if found:
            p = found[-1]["object_position"]
            text += f"探していた物体は ({p['x']}, {p['y']}) 付近にありました。"
        if fails:
            text += f"途中で {len(fails)} 回の失敗（安全停止など）があり、その都度やり直しました。"
        return text


class MockClient:
    """anthropic.Anthropic() の代わり
    messages.create だけを実装"""

    def __init__(self):
        self.messages = self
        self._planner = None
        self._last_call = None

    def create(self, model, max_tokens, system, tools, messages):
        if len(messages) == 1:
            self._planner = _Planner(messages[0]["content"].split("\n\n（参考:")[0])
            self._last_call = None
            last_result = None
        else:
            block = messages[-1]["content"][-1]
            last_result = json.loads(block["content"][0]["text"])
        action = self._planner.next(self._last_call, last_result)
        if action is None:
            return _resp([_text(self._planner.report())], "end_turn")
        self._last_call = action
        return _resp([_tool(action[0], **action[1])], "tool_use")
