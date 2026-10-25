"""実行ログのツール呼び出しを順に返す再生用の頭脳
シミュレーションは決定的なので、API を使わずに同じ動きを描き直せる"""
from __future__ import annotations

import json
import math
import uuid
from types import SimpleNamespace


class ReplayClient:
    def __init__(self, logs):
        self.logs = logs
        self.messages = self
        self.i = -1
        self.k = 0
        self.max_dev = 0.0

    @classmethod
    def from_file(cls, path):
        return cls(json.load(open(path, encoding="utf-8")))

    @property
    def model(self):
        return self.logs[0]["model"]

    def _check(self, messages):
        block = messages[-1]["content"][-1]
        got = json.loads(block["content"][0]["text"]).get("position")
        want = self.logs[self.i]["steps"][self.k - 1]["result"].get("position")
        if got and want:
            self.max_dev = max(self.max_dev, math.hypot(got["x"] - want["x"], got["y"] - want["y"]))

    def create(self, model, max_tokens, system, tools, messages):
        if len(messages) == 1:
            self.i, self.k = self.i + 1, 0
        else:
            self._check(messages)
        log = self.logs[self.i]
        usage = SimpleNamespace(input_tokens=0, output_tokens=0, cache_read_input_tokens=0, cache_creation_input_tokens=0)
        if self.k < len(log["steps"]):
            s = log["steps"][self.k]
            self.k += 1
            block = SimpleNamespace(type="tool_use", id="replay_" + uuid.uuid4().hex[:10], name=s["tool"], input=s["input"])
            return SimpleNamespace(content=[block], stop_reason="tool_use", usage=usage)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=log["final_text"])], stop_reason="end_turn",
                               usage=usage)
