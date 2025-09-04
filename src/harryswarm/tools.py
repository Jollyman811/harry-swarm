from typing import Tuple, Optional, List, Dict, Any
from .config import get_settings

def build_tools(
    enable_web: bool = True,
    enable_file: bool = True,
    enable_code: bool = True,
    enable_computer: bool = False
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    tools: List[Dict[str, Any]] = []
    if enable_web:
        tools.append({"type": "web_search"})
    if enable_file:
        tools.append({"type": "file_search"})
    if enable_code:
        tools.append({"type": "code_interpreter"})
    if enable_computer:
        tools.append({"type": "computer_use"})

    # tool resources (e.g., vector store ids for File Search)
    s = get_settings()
    tool_resources: Dict[str, Any] = {}
    if enable_file and s.vector_store_ids:
        tool_resources["file_search"] = {"vector_store_ids": s.vector_store_ids}

    return tools, (tool_resources or None)
