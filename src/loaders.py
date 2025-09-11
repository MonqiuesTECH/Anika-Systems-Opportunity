# src/loaders.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict
from pypdf import PdfReader
from bs4 import BeautifulSoup

RAW_DIR = Path("data/raw")
SEED_DIR = Path("data/seed")

def _read_pdf(path: Path) -> str:
    text = []
    with open(path, "rb") as f:
        pdf = PdfReader(f)
        for page in pdf.pages:
            txt = page.extract_text() or ""
            if txt:
                text.append(txt)
    return "\n".join(text).strip()

def _read_html(path: Path) -> str:
    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")
    # Keep visible text only
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return " ".join(soup.get_text(separator=" ").split())

def _enumerate_files() -> List[Path]:
    paths: List[Path] = []
    for root in (RAW_DIR, SEED_DIR):
        if root.exists():
            paths += list(root.rglob("*.pdf"))
            paths += list(root.rglob("*.html"))
            paths += list(root.rglob("*.htm"))
    # De-dup by name in case seed == raw
    uniq: Dict[str, Path] = {}
    for p in paths:
        uniq[str(p.resolve())] = p
    return list(uniq.values())

def load_raw_documents(raw_dir: Path | None = None) -> List[Dict]:
    docs: List[Dict] = []
    files = _enumerate_files()
    if not files:
        print("[loader] No .pdf/.html files found in data/raw/ or data/seed/")
        return docs

    for p in files:
        try:
            if p.suffix.lower() == ".pdf":
                text = _read_pdf(p)
            else:
                text = _read_html(p)
            if not text:
                continue
            docs.append({
                "id": str(p),
                "title": p.stem.replace("-", " ").replace("_", " ").title(),
                "source": str(p.parent.name),
                "url": "",
                "text": text,
            })
        except Exception as e:
            print(f"[loader] skip {p}: {e}")
    return docs
