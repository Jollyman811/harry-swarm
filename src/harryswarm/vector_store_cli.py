from __future__ import annotations
import argparse, os, glob
from typing import Iterable
from openai import OpenAI
from .config import get_settings

def _client() -> OpenAI:
    s = get_settings()
    return OpenAI(api_key=s.api_key, organization=s.org_id, project=s.project_id)

def _iter_paths(folder: str) -> Iterable[str]:
    patterns = ("*.pdf","*.txt","*.md","*.csv","*.docx","*.pptx","*.html")
    for p in patterns:
        yield from glob.glob(os.path.join(folder, p))

def create_store_with_files(name: str, folder: str) -> str:
    client = _client()
    vs = client.vector_stores.create(name=name)

    files_to_upload = [open(p, "rb") for p in _iter_paths(folder)]
    if not files_to_upload:
        print("No supported files found.")
        return vs.id

    try:
        client.vector_stores.files.batch_create(
            vector_store_id=vs.id,
            files=[{"file": f} for f in files_to_upload],
        )
    except Exception:
        # Fallback path: upload then attach
        uploaded = [client.files.create(file=f, purpose="assistants").id for f in files_to_upload]
        for fid in uploaded:
            client.vector_stores.files.create(vector_store_id=vs.id, file_id=fid)

    return vs.id

def main():
    ap = argparse.ArgumentParser(description="Create a vector store & upload a folder of docs.")
    ap.add_argument("--name", required=True)
    ap.add_argument("--folder", required=True)
    args = ap.parse_args()
    vid = create_store_with_files(args.name, args.folder)
    print(vid)

if __name__ == "__main__":
    main()
