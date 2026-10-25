"""OpenAI 互換 API（vLLM の Qwen3-VL など）を Claude API と同じ形で呼ぶ変換層"""
from __future__ import annotations

import json
import os
import re
import uuid
from types import SimpleNamespace

DEFAULT_LOCAL_URL = os.environ.get("LOCAL_LLM_URL", "http://llm:8000/v1")
DEFAULT_LOCAL_MODEL = os.environ.get("LOCAL_LLM_MODEL", "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8")

_THINK = re.compile(r"<think>.*?</think>", re.S)


def _get(block, key, default=None):
    if isinstance(block, dict):
        return block.get(key, default)
    return getattr(block, key, default)


def convert_tools(tools):
    return [{"type": "function",
             "function": {"name": t["name"], "description": t.get("description", ""),
                          "parameters": t.get("input_schema", {"type": "object", "properties": {}})}}
            for t in tools]


def convert_messages(system, messages):
    sys_text = system if isinstance(system, str) else "\n".join(_get(b, "text", "") for b in system)
    out = [{"role": "system", "content": sys_text}]
    for msg in messages:
        role, content = msg["role"], msg["content"]
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue
        if role == "assistant":
            texts, calls = [], []
            for b in content:
                if _get(b, "type") == "text":
                    texts.append(_get(b, "text", ""))
                elif _get(b, "type") == "tool_use":
                    calls.append({"id": _get(b, "id"), "type": "function",
                                  "function": {"name": _get(b, "name"),
                                               "arguments": json.dumps(_get(b, "input", {}), ensure_ascii=False)}})
            m = {"role": "assistant", "content": "\n".join(texts) or None}
            if calls:
                m["tool_calls"] = calls
            out.append(m)
            continue
        # tool メッセージは画像を持てないので、直後の user メッセージで渡す
        images = []
        for b in content:
            if _get(b, "type") == "tool_result":
                texts = []
                for c in _get(b, "content", []):
                    if c.get("type") == "text":
                        texts.append(c["text"])
                    elif c.get("type") == "image":
                        src = c["source"]
                        images.append(f"data:{src['media_type']};base64,{src['data']}")
                out.append({"role": "tool", "tool_call_id": _get(b, "tool_use_id"), "content": "\n".join(texts)})
            elif _get(b, "type") == "text":
                out.append({"role": "user", "content": _get(b, "text")})
        if images:
            parts = [{"type": "text", "text": "直前のツール（look など）で取得した頭部カメラの画像です。"}]
            parts += [{"type": "image_url", "image_url": {"url": u}} for u in images]
            out.append({"role": "user", "content": parts})
    return out


def convert_response(resp):
    msg = resp.choices[0].message
    blocks = []
    text = _THINK.sub("", msg.content or "").strip()
    if text:
        blocks.append(SimpleNamespace(type="text", text=text))
    for call in msg.tool_calls or []:
        try:
            args = json.loads(call.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {}
        blocks.append(SimpleNamespace(type="tool_use", id=call.id or "call_" + uuid.uuid4().hex[:12],
                                      name=call.function.name, input=args))
    stop = "tool_use" if any(b.type == "tool_use" for b in blocks) else "end_turn"
    u = resp.usage
    usage = SimpleNamespace(input_tokens=getattr(u, "prompt_tokens", 0) or 0,
                            output_tokens=getattr(u, "completion_tokens", 0) or 0,
                            cache_read_input_tokens=0, cache_creation_input_tokens=0)
    return SimpleNamespace(content=blocks, stop_reason=stop, usage=usage)


class LocalVLMClient:
    """anthropic.Anthropic() の代わり
    messages.create だけを実装"""

    def __init__(self, base_url: str = DEFAULT_LOCAL_URL, api_key: str = "EMPTY", temperature: float = 0.2):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=180)
        self.temperature = temperature
        self.messages = self

    def create(self, model, max_tokens, system, tools, messages):
        resp = self.client.chat.completions.create(
            model=model, max_tokens=max_tokens, temperature=self.temperature,
            messages=convert_messages(system, messages), tools=convert_tools(tools), tool_choice="auto")
        return convert_response(resp)
