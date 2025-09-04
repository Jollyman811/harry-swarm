from openai import OpenAI
from .config import get_settings

def get_openai_client() -> OpenAI:
    s = get_settings()
    return OpenAI(api_key=s.api_key, organization=s.org_id, project=s.project_id)
