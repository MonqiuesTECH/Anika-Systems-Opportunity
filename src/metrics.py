# src/metrics.py
from __future__ import annotations
from typing import List, Dict

def compute_hit_ratio(hits: List[Dict]) -> float:
    if not hits:
        return 0.0
    top = sum(1 for h in hits if h.get("score",0) >= 0.30)
    return round(top / max(1, len(hits)), 3)
