from typing import Optional, Tuple, Any, Dict, List
from .client import get_openai_client

def _mk_message(role: str, text: str) -> Dict[str, Any]:
    return {"role": role, "content": [{"type": "input_text", "text": text}]}

def run_agent(
    prompt: str,
    model: str,
    tools: List[Dict[str, Any]],
    tool_resources: Optional[Dict[str, Any]] = None,
    system_prompt: Optional[str] = None,
    temperature: float = 0.2,
) -> Tuple[str, Any]:
    client = get_openai_client()

    blocks: List[Dict[str, Any]] = []
    if system_prompt:
        blocks.append(_mk_message("system", system_prompt))
    blocks.append(_mk_message("user", prompt))

    resp = client.responses.create(
        model=model,
        input=blocks,
        tools=tools,
        tool_resources=tool_resources,
        temperature=temperature,
    )

    # Best-effort extract (SDK commonly exposes .output_text)
    text = getattr(resp, "output_text", None)
    if not text:
        text = _coalesce_text(resp)

    return text, resp

def _coalesce_text(resp: Any) -> str:
    out = []
    try:
        for item in getattr(resp, "output", []) or []:
            for c in getattr(item, "content", []) or []:
                if getattr(c, "type", "") in ("output_text", "text"):
                    out.append(getattr(c, "text", ""))
    except Exception:
        return str(resp)
    return "\n".join([t for t in out if t])
