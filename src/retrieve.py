from typing import List, Dict, Optional
import numpy as np
import faiss
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer

from .embed_index import IndexArtifacts, MODEL_NAME

@dataclass
class AnswerWithCitations:
    text: str
    citations: List[str]
    sources: List[Dict]

class RagRetriever:
    def __init__(self, artifacts: IndexArtifacts, top_k_default: int = 4):
        self.artifacts = artifacts
        self.model = SentenceTransformer(MODEL_NAME)
        self.top_k_default = top_k_default

    def _encode(self, query: str) -> np.ndarray:
        v = self.model.encode([query], normalize_embeddings=True)
        return np.asarray(v, dtype=np.float32)

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: float = 0.4,
        section_contains: Optional[str] = None,
        year_min: Optional[int] = None,
    ) -> List[Dict]:
        top_k = top_k or self.top_k_default
        qv = self._encode(query)
        scores, idxs = self.artifacts.index.search(qv, top_k * 5)  # oversample, filter later
        scores, idxs = scores[0], idxs[0]

        results = []
        for score, idx in zip(scores, idxs):
            if idx < 0:
                continue
            meta = self.artifacts.metadata[idx]
            # cosine sim because embeddings are normalized; FAISS IP ≈ cosine
            sim = float(score)
            if sim < score_threshold:
                continue
            if section_contains and section_contains.lower() not in (meta.get("section","") or "").lower():
                continue
            if year_min and int(meta.get("year") or 0) < year_min:
                continue

            inline = self._inline_citation(meta)
            preview = meta["text"][:300].replace("\n", " ") + ("..." if len(meta["text"]) > 300 else "")
            results.append({
                "score": sim,
                "inline_citation": inline,
                "preview": preview,
                "title": meta.get("title",""),
                "source": meta.get("source",""),
                "url": meta.get("url",""),
                "section": meta.get("section",""),
                "year": meta.get("year",0),
                "text": meta["text"]
            })
            if len(results) >= top_k:
                break

        return results

    @staticmethod
    def _inline_citation(meta: Dict) -> str:
        # [Title §Section, Year] or [Title, Year]
        title = meta.get("title") or "Doc"
        section = meta.get("section") or ""
        year = meta.get("year") or ""
        core = f"{title}" + (f" §{section}" if section else "")
        return f"[{core}, {year}]" if year else f"[{core}]"
