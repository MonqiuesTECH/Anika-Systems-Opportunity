# src/loaders.py
from __future__ import annotations
import os, pathlib, re, time, random
from typing import List, Dict, Tuple
from bs4 import BeautifulSoup
from pypdf import PdfReader
import requests

USER_AGENT = "Mozilla/5.0 (compatible; anika-rag/1.0)"
STARTER_URLS = [
    # Public, low-risk pages (HTML). Feel free to add more.
    "https://www.anikasystems.com/",
    "https://www.anikasystems.com/who-we-are",
    "https://www.anikasystems.com/capabilities",
    "https://www.anikasystems.com/insights",
    "https://www.anikasystems.com/careers",
    "https://www.anikasystems.com/partners",
    "https://www.anikasystems.com/contact",
    # Federal AI/automation references (HTML, generally allowed)
    "https://www.whitehouse.gov/ostp/ai/",
    "https://www.ai.gov/",
    "https://www.whitehouse.gov/briefing-room/presidential-actions/2023/10/30/executive-order-on-the-safe-secure-and-trustworthy-development-and-use-of-artificial-intelligence/",
]

def ensure_dirs(raw_dir: pathlib.Path, idx_dir: pathlib.Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    idx_dir.mkdir(parents=True, exist_ok=True)

def _fetch_html(url: str, timeout=20) -> str | None:
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
        if resp.status_code == 200 and "text/html" in resp.headers.get("Content-Type",""):
            return resp.text
    except Exception:
        return None
    return None

def fetch_starter_corpus(raw_dir: pathlib.Path, target_count: int = 35) -> Tuple[List[str], List[str]]:
    saved, failed = [], []
    for url in STARTER_URLS:
        html = _fetch_html(url)
        if not html:
            failed.append(url); continue
        name = re.sub(r"[^a-z0-9]+", "_", url.lower()).strip("_")[:80] + ".html"
        (raw_dir / name).write_text(html, encoding="utf-8")
        saved.append(name)
        if len(saved) >= target_count:
            break
    return saved, failed

def _read_pdf(path: pathlib.Path) -> str:
    text = []
    try:
        r = PdfReader(str(path))
        for p in r.pages:
            t = p.extract_text() or ""
            text.append(t)
    except Exception:
        return ""
    return "\n".join(text)

def _read_html(path: pathlib.Path) -> str:
    try:
        soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
        # remove nav/footers/scripts
        for tag in soup(["script","style","noscript","nav","footer","header"]):
            tag.decompose()
        txt = soup.get_text(separator=" ")
        return re.sub(r"\s+", " ", txt).strip()
    except Exception:
        return ""

def load_local_and_uploaded(raw_dir: pathlib.Path) -> List[Dict]:
    docs: List[Dict] = []
    for p in sorted(raw_dir.glob("*")):
        if p.suffix.lower() in {".pdf"}:
            txt = _read_pdf(p)
        elif p.suffix.lower() in {".html", ".htm"}:
            txt = _read_html(p)
        else:
            continue
        if not txt or len(txt) < 200:
            continue
        docs.append({"source": p.name, "text": txt})
    return docs

