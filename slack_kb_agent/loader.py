import os
from pathlib import Path
from typing import List, Dict


def load_docs(docs_dir: str) -> List[Dict]:
    """Load all .md and .txt files from docs_dir, return list of {filename, content}"""
    docs = []
    for path in Path(docs_dir).rglob("*"):
        if path.suffix in (".md", ".txt") and path.is_file():
            content = path.read_text(encoding="utf-8", errors="replace")
            docs.append({"filename": str(path), "content": content})
    return docs
