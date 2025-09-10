import pathlib
from typing import List, Dict
from bs4 import BeautifulSoup
from pypdf import PdfReader

ALLOWED_EXT = {".pdf", ".html", ".htm"}

def _read_pdf(path: pathlib.Path) -> str:
    text_parts = []
    try:
        pdf = PdfReader(str(path))
        for page in pdf.pages:
            t = page.extract_text() or ""
            text_parts.append(t)
    except Exception as e:
        # If extraction fails, still return empty string so guards trigger gracefully
        print(f"[loader] PDF read failed {path}: {e}")
    return "\n".join(text_parts).strip()

def _read_html(path: pathlib.Path) -> str:
    try:
        html = path.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(html, "lxml")
        for s in soup(["script", "style"]):
            s.extract()
        return soup.get_text(separator="\n").strip()
    except Exception as e:
        print(f"[loader] HTML read failed {path}: {e}")
        return ""

def load_raw_documents(raw_dir: pathlib.Path) -> List[Dict]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    docs: List[Dict] = []

    paths = [p for p in sorted(raw_dir.glob("**/*")) if p.is_file() and p.suffix.lower() in ALLOWED_EXT]
    if not paths:
        print("[loader] No .pdf/.html files found in data/raw/")
        return docs

    for p in paths:
        text = ""
        if p.suffix.lower() == ".pdf":
            text = _read_pdf(p)
        else:
            text = _read_html(p)

        if not text:
            print(f"[loader] Skipped empty text: {p}")
            continue

        docs.append({
            "id": str(p.relative_to(raw_dir)),
            "title": p.stem.replace("_", " ").strip(),
            "source": str(p),
            "url": "",
            "text": text
        })
    return docs

