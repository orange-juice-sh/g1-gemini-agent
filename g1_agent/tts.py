"""VOICEVOX ENGINE で発話を音声にする（docker compose --profile tts）
声は No.7（アナウンス）
動画には「VOICEVOX:No.7」のクレジットが必要"""
from __future__ import annotations

import io
import json
import os
import urllib.parse
import urllib.request
import wave

TTS_URL = os.environ.get("TTS_URL", "http://tts:50021")
SPEAKER = int(os.environ.get("TTS_SPEAKER", "30"))
CREDIT = "音声: VOICEVOX:No.7"


def available() -> bool:
    try:
        urllib.request.urlopen(f"{TTS_URL}/version", timeout=3)
        return True
    except Exception:
        return False


def synthesize(text: str, speed: float = 1.1) -> tuple[bytes, float]:
    """WAV のバイト列と長さ[s]を返す"""
    q = urllib.parse.urlencode({"text": text, "speaker": SPEAKER})
    query = json.load(urllib.request.urlopen(urllib.request.Request(f"{TTS_URL}/audio_query?{q}", method="POST"),
                                             timeout=60))
    query["speedScale"] = speed
    query["postPhonemeLength"] = 0.2
    req = urllib.request.Request(f"{TTS_URL}/synthesis?speaker={SPEAKER}", data=json.dumps(query).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    wav = urllib.request.urlopen(req, timeout=120).read()
    with wave.open(io.BytesIO(wav)) as w:
        return wav, w.getnframes() / w.getframerate()
