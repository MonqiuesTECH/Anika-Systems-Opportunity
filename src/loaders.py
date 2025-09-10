import pathlib
from typing import List, Dict
from bs4 import BeautifulSoup
from pypdf import PdfReader

ALLOWED_EXT = {".pdf", ".html", ".htm"}

def _read_pdf(path: pathlib.Path) -> str:
    text_parts = []
    pdf = PdfReader(str(path))
    for page in pdf.pages:
        t = page.extract_text() or ""
        text_parts.append(t)
    return "\n".join(text_parts)

def _read_html(path: pathlib.Path) -> str:
    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")
    for s in soup(["script", "style"]):
        s.extract()
    return soup.get_text(separator="\n")

def load_raw_documents(raw_dir: pathlib.Path) -> List[Dict]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    docs = []
    for p in sorted(raw_dir.glob("**/*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in ALLOWED_EXT:
            continue
        try:
            if p.suffix.lower() == ".pdf":
                text = _read_pdf(p)
            else:
                text = _read_html(p)
        except Exception as e:
            print(f"[loader] Skipped {p}: {e}")
            continue

        docs.append({
            "id": str(p.relative_to(raw_dir)),
            "title": p.stem.replace("_", " ").strip(),
            "source": str(p),
            "url": "",
            "text": text
        })
    return docs
