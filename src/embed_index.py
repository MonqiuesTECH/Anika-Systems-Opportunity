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
    d = model.get_sentence_embedding_dimension()
    if not texts:
        return np.zeros((0, d), dtype=np.float32)
    vecs = model.encode(
        texts,
        batch_size=batch,
        show_progress_bar=False,
        normalize_embeddings=True
    )
    vecs = np.asarray(vecs, dtype=np.float32)
    if vecs.ndim == 1:  # single vector
        vecs = vecs.reshape(1, -1)
    return np.ascontiguousarray(vecs, dtype=np.float32)

def ensure_faiss_index(chunks: List[Dict], index_dir: pathlib.Path) -> IndexArtifacts:
    index_dir.mkdir(parents=True, exist_ok=True)
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
    texts = [c.get("text", "") for c in chunks if c.get("text")]
    meta = [c for c in chunks if c.get("text")]

    vecs = _embed_texts(texts, model)

    index = faiss.IndexFlatIP(dim)
    if vecs.size and vecs.shape[0] > 0:
        index.add(vecs)

    with open(meta_path, "w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    faiss.write_index(index, str(index_path))
    return IndexArtifacts(index=index, metadata=meta, embedding_model_name=MODEL_NAME, dim=dim)
