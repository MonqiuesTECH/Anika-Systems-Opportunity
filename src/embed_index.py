import json
import pathlib
from dataclasses import dataclass
from typing import List, Dict

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

INDEX_FILE = "faiss.index"
META_FILE = "meta.jsonl"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

@dataclass
class IndexArtifacts:
    index: faiss.IndexFlatIP
    metadata: List[Dict]
    embedding_model_name: str
    dim: int

def _embed_texts(texts: List[str], model: SentenceTransformer, batch: int = 128) -> np.ndarray:
    if not texts:
        # Return an empty (0, d) array when we don't have texts
        d = model.get_sentence_embedding_dimension()
        return np.zeros((0, d), dtype=np.float32)
    vecs = model.encode(
        texts,
        batch_size=batch,
        show_progress_bar=False,
        normalize_embeddings=True
    )
    vecs = np.asarray(vecs, dtype=np.float32)
    if vecs.ndim == 1:  # single vector → (1, d)
        vecs = vecs.reshape(1, -1)
    return np.ascontiguousarray(vecs, dtype=np.float32)

def ensure_faiss_index(chunks: List[Dict], index_dir: pathlib.Path) -> IndexArtifacts:
    index_path = index_dir / INDEX_FILE
    meta_path = index_dir / META_FILE

    model = SentenceTransformer(MODEL_NAME)
    dim = model.get_sentence_embedding_dimension()

    # Reuse if present
    if index_path.exists() and meta_path.exists():
        index = faiss.read_index(str(index_path))
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = [json.loads(l) for l in f]
        return IndexArtifacts(index=index, metadata=meta, embedding_model_name=MODEL_NAME, dim=dim)

    # Fresh build
    texts = [c["text"] for c in chunks if c.get("text")]
    meta = [c for c in chunks if c.get("text")]

    vecs = _embed_texts(texts, model)

    if vecs.size == 0 or vecs.shape[0] == 0:
        # No data → create an empty index so the app still runs with messages
        index = faiss.IndexFlatIP(dim)
        with open(meta_path, "w", encoding="utf-8") as f:
            pass  # empty file
        faiss.write_index(index, str(index_path))
        return IndexArtifacts(index=index, metadata=[], embedding_model_name=MODEL_NAME, dim=dim)

    index = faiss.IndexFlatIP(dim)
    index.add(vecs)  # vecs is guaranteed (n, d) float32

    with open(meta_path, "w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    faiss.write_index(index, str(index_path))

    # reload metadata to be safe
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_reload = [json.loads(l) for l in f]
    return IndexArtifacts(index=index, metadata=meta_reload, embedding_model_name=MODEL_NAME, dim=dim)
