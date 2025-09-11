# src/loaders.py
from __future__ import annotations
import re
from pathlib import Path
from typing import List, Dict, Any

def guess_filename(name: str) -> str:
    base = re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_")
    return base or "file.html"

def save_upload_as_html(files, raw_dir: str) -> int:
    """Save uploaded PDFs/HTMLs to data/raw/ (no conversion; keeps extension)."""
    out_dir = Path(raw_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for f in files:
        data = f.read()
        ext = (f.name.split(".")[-1] or "html").lower()
        if ext not in ("pdf", "html", "htm"):
            ext = "html"
        fn = guess_filename(f.name)
        (out_dir / fn).write_bytes(data)
        count += 1
    return count

def load_raw_files(raw_dir: str, section_filter: str = "", min_year: int = 0) -> List[Dict[str, Any]]:
    """
    Returns a list of {"path": str, "title": str, "year": int}.
    HTML year is unknown → 0; PDF year extraction is skipped for speed/reliability.
    """
    paths: List[Path] = []
    for p in Path(raw_dir).glob("**/*"):
        if p.is_file() and p.suffix.lower() in {".pdf", ".html", ".htm"}:
            paths.append(p)

    out: List[Dict[str, Any]] = []
    for p in sorted(paths):
        title = p.stem.replace("_", " ").strip() or p.name
        year = 0  # keep neutral; fast and robust
        out.append({"path": str(p), "title": title, "year": year})

    # We DON’T actually filter by section/min_year here; that happens at read time
    return out


