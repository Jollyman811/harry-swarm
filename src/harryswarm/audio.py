from __future__ import annotations
from io import BytesIO
from openai import OpenAI
from .config import get_settings

def _client() -> OpenAI:
    s = get_settings()
    return OpenAI(api_key=s.api_key, organization=s.org_id, project=s.project_id)

def transcribe_bytes(audio_bytes: bytes, filename: str = "input.wav", model: str = "gpt-4o-transcribe") -> str:
    client = _client()
    buf = BytesIO(audio_bytes)
    buf.name = filename
    res = client.audio.transcriptions.create(model=model, file=buf)
    return getattr(res, "text", str(res))

def synthesize_speech(text: str, voice: str = "alloy", model: str = "gpt-4o-mini-tts", fmt: str = "mp3") -> bytes:
    client = _client()
    out = client.audio.speech.create(model=model, voice=voice, input=text, format=fmt)
    return getattr(out, "content", bytes(out))
