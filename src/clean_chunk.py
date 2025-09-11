# src/clean_chunk.py
from __future__ import annotations
import re
from typing import List, Dict, Optional, Tuple

def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\u200b\u200c\u200d]+", "", text)  # zero-widths
    return text.strip()

def _year_guard(meta_src: str, text: str, min_year: Optional[int]) -> bool:
    if not min_year or min_year <= 0:
        return True
    # simple year scan; keep if any year ≥ min_year exists
    years = [int(y) for y in re.findall(r"\b(19|20)\d{2}\b", text)]
    return any(y >= min_year for y in years)

def clean_and_chunk(
    docs: List[Dict], 
    section_substr: Optional[str] = None,
    min_year: Optional[int] = None,
    chunk_size: int = 850,
    overlap: int = 120
) -> Tuple[List[str], List[Dict]]:
    chunks, meta = [], []
    section_substr = (section_substr or "").lower().strip()
    for d in docs:
        txt = _clean(d["text"])
        if section_substr and section_substr not in txt.lower():
            continue
        if not _year_guard(d["source"], txt, min_year):
            continue
        # simple sliding window
        words = txt.split()
        if len(words) <= chunk_size:
            chunks.append(" ".join(words)); meta.append({"source": d["source"]}); continue
        step = max(1, chunk_size - overlap)
        for i in range(0, len(words), step):
            chunk = " ".join(words[i:i+chunk_size])
            if len(chunk) < 200:
                continue
            chunks.append(chunk)
            meta.append({"source": d["source"]})
    return chunks, meta

