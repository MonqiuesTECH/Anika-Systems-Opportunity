import json
import pathlib
import re
from typing import List, Dict, Iterable

PROC_FILENAME = "chunks.jsonl"

def _normalize_whitespace(t: str) -> str:
    t = re.sub(r"[ \t]+", " ", t or "")
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def _split_by_heading_or_sentences(text: str, max_chars: int = 1200) -> Iterable[str]:
    if not text:
        return []
    # Prefer block/heading splits; fall back to paragraph gaps.
    blocks = re.split(r"\n\s*(?P<h>#+\s+|[A-Z][A-Za-z0-9 ]{2,}\n[-=]{3,}\n)", text)
    if len(blocks) == 1:
        blocks = re.split(r"\n{2,}", text)

    out: List[str] = []
    for block in blocks:
        b = _normalize_whitespace(block)
        if not b:
            continue
        while len(b) > max_chars:
            cut = b.rfind(". ", 0, max_chars)
            if cut == -1:
                cut = max_chars
            out.append(b[:cut].strip())
            b = b[cut:].strip()
        if b:
            out.append(b)
    return out

def _infer_year(text: str) -> int:
    candidates = re.findall(r"\b(20\d{2}|19\d{2})\b", text or "")
    if not candidates:
        return 0
    try:
        years = [int(c) for c in candidates]
        return max(years)
    except Exception:
        return 0

def clean_and_chunk(docs: List[Dict], save_dir: pathlib.Path) -> List[Dict]:
    save_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = save_dir / PROC_FILENAME

    chunks: List[Dict] = []
    for d in docs or []:
        text = _normalize_whitespace(d.get("text", ""))
        if not text:
            continue

        # Precompute year once per doc
        year = _infer_year(text)

        for i, chunk in enumerate(_split_by_heading_or_sentences(text)):
            chunks.append({
                "doc_id": d.get("id", ""),
                "title": d.get("title", ""),
                "source": d.get("source", ""),
                "url": d.get("url", ""),
                "section": "",
                "year": year,
                "text": chunk
            })

    # Persist even if empty so downstream logic can handle gracefully
    with open(chunks_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    return chunks
