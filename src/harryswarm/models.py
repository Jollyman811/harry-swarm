from __future__ import annotations
from functools import lru_cache
from typing import List, Tuple
from .client import get_openai_client

AGENT_HINTS = ("gpt-4o", "omni", "realtime", "search", "mini", "response", "operator", "gpt-image", "gpt-5")

def _agent_friendly(mid: str) -> bool:
    m = mid.lower()
    return m.startswith("gpt-") and any(h in m for h in AGENT_HINTS)

@lru_cache(maxsize=1)
def list_models() -> List[str]:
    client = get_openai_client()
    try:
        models = client.models.list()
        ids = sorted({m.id for m in getattr(models, "data", [])})
        if ids:
            return ids
    except Exception:
        pass
    # fallback set (manual override in UI still works)
    return [
        "gpt-4o", "gpt-4o-mini", "gpt-4o-realtime-preview", "gpt-image-1",
        "gpt-5", "gpt-5-mini", "gpt-5-realtime",
    ]

def group_models(all_ids: List[str]) -> Tuple[List[str], List[str], List[str]]:
    agentish = sorted([m for m in all_ids if _agent_friendly(m)])
    gpt5 = sorted([m for m in all_ids if m.lower().startswith("gpt-5")])
    return agentish, gpt5, sorted(all_ids)
