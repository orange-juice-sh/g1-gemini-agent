"""Gemini（Interactions API）を Claude API と同じ形で呼ぶ変換層
会話の状態はサーバー側に保存し（previous_interaction_id）、毎回は新しい入力だけを送る"""
from __future__ import annotations

import json
import os
import uuid
from types import SimpleNamespace

DEFAULT_GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-4-argon")


def _get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def convert_tools(tools):
    return [{"type": "function", "name": t["name"], "description": t.get("description", ""),
             "parameters": t.get("input_schema", {"type": "object", "properties": {}})} for t in tools]


def convert_tool_results(content, names=None):
    """tool_result を function_result に変換する（Gemini は関数名が必須なので names で補う）"""
    names = names or {}
    steps = []
    for b in content:
        if _get(b, "type") != "tool_result":
            continue
        result = []
        for c in _get(b, "content", []):
            if c.get("type") == "text":
                result.append({"type": "text", "text": c["text"]})
            elif c.get("type") == "image":
                src = c["source"]
                result.append({"type": "image", "data": src["data"], "mime_type": src["media_type"]})
        call_id = _get(b, "tool_use_id")
        step = {"type": "function_result", "call_id": call_id, "result": result,
                "name": _get(b, "name") or names.get(call_id, "")}
        if _get(b, "is_error"):
            step["is_error"] = True
        steps.append(step)
    return steps


def convert_response(interaction):
    steps = _get(interaction, "steps", None) or []
    last_input = max((i for i, s in enumerate(steps) if _get(s, "type") in ("user_input", "function_result")),
                     default=-1)
    blocks = []
    for s in steps[last_input + 1:]:
        stype = _get(s, "type")
        if stype == "model_output":
            text = "".join(_get(c, "text", "") or "" for c in (_get(s, "content", None) or [])
                           if _get(c, "type") == "text").strip()
            if text:
                blocks.append(SimpleNamespace(type="text", text=text))
        elif stype == "function_call":
            args = _get(s, "arguments", None) or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            blocks.append(SimpleNamespace(type="tool_use", id=_get(s, "id") or "call_" + uuid.uuid4().hex[:12],
                                          name=_get(s, "name"), input=dict(args)))
    stop = "tool_use" if any(b.type == "tool_use" for b in blocks) else "end_turn"
    u = _get(interaction, "usage", None)
    usage = SimpleNamespace(input_tokens=(_get(u, "total_input_tokens", 0) or 0) if u else 0,
                            output_tokens=((_get(u, "total_output_tokens", 0) or 0)
                                           + (_get(u, "total_thought_tokens", 0) or 0)) if u else 0,
                            cache_read_input_tokens=(_get(u, "total_cached_tokens", 0) or 0) if u else 0,
                            cache_creation_input_tokens=0)
    return SimpleNamespace(content=blocks, stop_reason=stop, usage=usage)


class GeminiClient:
    """anthropic.Anthropic() の代わり
    messages.create だけを実装"""

    def __init__(self, api_key: str | None = None, base_url: str | None = None, timeout_s: float | None = None):
        from google import genai
        from google.genai import types

        timeout_s = timeout_s or float(os.environ.get("GEMINI_TIMEOUT", "120"))
        self.timeout_s = timeout_s
        # 429 は再試行しない（SDK が長い Retry-After を待ち続けるため）
        retry = types.HttpRetryOptions(attempts=3, http_status_codes=[500, 502, 503, 504])
        opts = types.HttpOptions(base_url=base_url, timeout=int(timeout_s * 1000), retry_options=retry)
        self.client = genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY"), http_options=opts)
        self.thinking_level = os.environ.get("GEMINI_THINKING_LEVEL")
        self.messages = self
        self._previous_id = None

    def create(self, model, max_tokens, system, tools, messages):
        system_text = system if isinstance(system, str) else "\n".join(_get(b, "text", "") for b in system)
        last = messages[-1]
        if len(messages) == 1:
            self._previous_id = None
            new_input = last["content"]
        else:
            names = {}
            for msg in messages:
                if msg["role"] == "assistant" and not isinstance(msg["content"], str):
                    for b in msg["content"]:
                        if _get(b, "type") == "tool_use":
                            names[_get(b, "id")] = _get(b, "name")
            new_input = convert_tool_results(last["content"], names)
        # 思考トークンも出力の上限に含まれる
        config = {"max_output_tokens": max(max_tokens, 4096)}
        if self.thinking_level:
            config["thinking_level"] = self.thinking_level
        kwargs = dict(model=model, input=new_input, system_instruction=system_text,
                      tools=convert_tools(tools), generation_config=config)
        if self._previous_id:
            kwargs["previous_interaction_id"] = self._previous_id
        try:
            try:
                interaction = self.client.interactions.create(**kwargs)
            except Exception as e:
                if not ("timeout" in type(e).__name__.lower() or "timed out" in str(e).lower()):
                    raise
                print(f"   … {self.timeout_s:.0f} 秒以内に返事がなかったので、もう一度問い合わせます", flush=True)
                interaction = self.client.interactions.create(**kwargs)
        except Exception as e:
            if "timeout" in type(e).__name__.lower() or "timed out" in str(e).lower():
                raise TimeoutError(f"Gemini（{model}）から {self.timeout_s:.0f} 秒以内に返事がありませんでした。"
                                   "混雑しているか、無料枠のため後回しにされている可能性があります。") from e
            if "429" in str(e) or "too_many_requests" in str(e) or "rate limit" in str(e).lower():
                raise RuntimeError(f"Gemini（{model}）の回数制限に達しました（429）。"
                                   f"無料枠なら1日の上限の可能性があります。詳細: {str(e)[:300]}") from e
            raise
        self._previous_id = _get(interaction, "id")
        return convert_response(interaction)
