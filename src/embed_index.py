# src/embed_index.py
from __future__ import annotations
from typing import List, Dict, Tuple, Optional
import pathlib, pickle
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer

EMB_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def _embed(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    vecs = model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    return np.asarray(vecs, dtype="float32")

def build_or_load_faiss(
    chunks: List[str],
    meta: List[Dict],
    idx_dir: pathlib.Path,
    force_rebuild: bool=False
) -> Tuple[faiss.IndexFlatIP, List[Dict]]:
    idx_path = idx_dir / "chunks.faiss"
    meta_path = idx_dir / "meta.pkl"

    if idx_path.exists() and meta_path.exists() and not force_rebuild:
        index = faiss.read_index(str(idx_path))
        with open(meta_path, "rb") as fh:
            loaded_meta = pickle.load(fh)
        return index, loaded_meta

    # build fresh
    model = SentenceTransformer(EMB_NAME)
    vecs = _embed(model, chunks)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)

    idx_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(idx_path))
    with open(meta_path, "wb") as fh:
        pickle.dump(meta, fh)
    return index, meta

def search_index(
    query: str,
    idx_dir: pathlib.Path,
    top_k: int,
    score_threshold: float,
    meta: List[Dict]
) -> List[Dict]:
    idx_path = idx_dir / "chunks.faiss"
    if not idx_path.exists():
        return []
    index = faiss.read_index(str(idx_path))
    model = SentenceTransformer(EMB_NAME)
    qv = _embed(model, [query])
    scores, ids = index.search(qv, top_k)
    out: List[Dict] = []
    for s, i in zip(scores[0], ids[0]):
        if int(i) < 0: 
            continue
        if float(1.0 - s) > (1.0 - (1.0 - score_threshold)):  # simple gating
            pass
        out.append({"score": float(s), "chunk_id": int(i), "source": meta[i]["source"]})
    # dedupe by source, keep best first
    seen = set(); filtered=[]
    for r in sorted(out, key=lambda x: -x["score"]):
        if r["source"] in seen: 
            continue
        seen.add(r["source"]); filtered.append(r)
    return filtered
