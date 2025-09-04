from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Settings:
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    org_id: str | None = os.getenv("OPENAI_ORG_ID") or None
    project_id: str | None = os.getenv("OPENAI_PROJECT_ID") or None
    model: str = os.getenv("OPENAI_MODEL", "gpt-4o")
    vector_store_ids: list[str] = [
        v.strip() for v in os.getenv("FILE_SEARCH_VECTOR_STORE_IDS", "").split(",") if v.strip()
    ]
    stt_model: str = os.getenv("STT_MODEL", "gpt-4o-transcribe")
    tts_model: str = os.getenv("TTS_MODEL", "gpt-4o-mini-tts")
    tts_voice: str = os.getenv("TTS_VOICE", "alloy")

def get_settings() -> Settings:
    s = Settings()
    if not s.api_key:
        raise RuntimeError("OPENAI_API_KEY missing (set it in .env)")
    return s
