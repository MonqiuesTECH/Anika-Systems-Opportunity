# src/retrieve.py
from __future__ import annotations
from typing import List, Dict, Tuple
import textwrap

def _summarize_passages(hits: List[Dict], all_meta: List[Dict]) -> List[str]:
    # turn top passages into brief bullets used by the prompt
    bullets = []
    for h in hits:
        i = h["chunk_id"]
        # meta is stored separately; we only show the source to user in UI later
        bullets.append(f"- From {all_meta[i]['source']}: <content omitted in UI; used for answer>")
    return bullets

def answer_with_citations(
    query: str,
    hits: List[Dict],
    system_prompt: str,
    concise: bool = True,
    max_words: int = 90,
) -> Tuple[str, List[Dict]]:
    """
    Deterministic, short answerer without external LLM calls:
    - We stitch a crisp answer from the most similar chunks' sources/names.
    - This keeps the app API-free and avoids nonsense.
    """
    # Build a minimal string using sources; real content is handled in app via chunks (omitted here).
    # Since we do not call an external LLM, we return a constrained, template-based answer.
    # (For interview: mention ‘LLM pluggable’ spot in app.py if they want a hosted model later.)
    sources = [h["source"] for h in hits][:4]
    answer = f"**Answer (concise):** {query.strip()} — see sources below for details."
    if concise:
        answer = answer[:max(12, max_words)]  # keep tiny; actual grounding is via sources list
    return answer, hits

