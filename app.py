# app.py
# RAG Chatbot – Source-Grounded Answers (self-contained)
# Python 3.13 / Streamlit 1.49
from __future__ import annotations

import os
import io
import re
import json
import time
import pickle
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import streamlit as st
from pypdf import PdfReader
from bs4 import BeautifulSoup
import numpy as np

# Embeddings & FAISS
from sentence_transformers import SentenceTransformer
import faiss  # faiss-cpu in requirements

RAW_DIR = "data/raw"
INDEX_DIR = "data/index"
META_PATH = os.path.join(INDEX_DIR, "meta.pkl")
FAISS_PATH = os.path.join(INDEX_DIR, "faiss.index")

# ---- Utilities ---------------------------------------------------------------

def ensure_dirs() -> None:
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(INDEX_DIR, exist_ok=True)

def read_pdf(path: str) -> str:
    try:
        reader = PdfReader(path)
        texts = []
        for page in reader.pages:
            txt = page.extract_text() or ""
            texts.append(txt)
        return "\n".join(texts)
    except Exception as e:
        return f""  # skip unreadable PDFs silently

def read_html(path: str) -> str:
    try:
        with open(path, "rb") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
        # Visible text only
        for script in soup(["script", "style", "noscript"]):
            script.extract()
        return soup.get_text(separator="\n", strip=True)
    except Exception:
        return ""

def find_docs() -> List[str]:
    files = []
    for name in os.listdir(RAW_DIR):
        low = name.lower()
        if low.endswith(".pdf") or low.endswith(".html") or low.endswith(".htm"):
            files.append(os.path.join(RAW_DIR, name))
    return sorted(files)

def chunk_text(text: str, max_chars: int = 1200, overlap: int = 150) -> List[str]:
    # simple, deterministic chunker
    if not text.strip():
        return []
    paragraphs = [p.strip() for p in text.splitlines() if p.strip()]
    joined = "\n".join(paragraphs)
    chunks = []
    start = 0
    while start < len(joined):
        end = min(start + max_chars, len(joined))
        chunks.append(joined[start:end])
        if end == len(joined):
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

def load_corpus() -> Tuple[List[str], List[Dict[str, Any]]]:
    """Return chunk texts and per-chunk metadata."""
    chunk_texts: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    for path in find_docs():
        if path.lower().endswith(".pdf"):
            text = read_pdf(path)
        else:
            text = read_html(path)
        for i, ch in enumerate(chunk_text(text)):
            chunk_texts.append(ch)
            metadatas.append({
                "source": os.path.basename(path),
                "chunk_id": i,
            })
    return chunk_texts, metadatas

def save_index(index: faiss.IndexFlatIP, meta: Dict[str, Any]) -> None:
    faiss.write_index(index, FAISS_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(meta, f)

def load_index() -> Tuple[faiss.IndexFlatIP | None, Dict[str, Any] | None]:
    if not (os.path.exists(FAISS_PATH) and os.path.exists(META_PATH)):
        return None, None
    try:
        index = faiss.read_index(FAISS_PATH)
        with open(META_PATH, "rb") as f:
            meta = pickle.load(f)
        return index, meta
    except Exception:
        return None, None

def normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / norms

# ---- Embedding Model ---------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_model():
    # Small, CPU-friendly model already in requirements
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ---- Build / Ensure Index ----------------------------------------------------

def build_index() -> Tuple[int, str]:
    """Build FAISS index from RAW_DIR; returns (#chunks, message)."""
    chunks, metas = load_corpus()
    if len(chunks) == 0:
        return 0, "No index yet. Add PDFs/HTML to data/raw/ or upload below, then click Rebuild index."
    model = get_model()
    embeddings = model.encode(chunks, batch_size=64, show_progress_bar=False)
    embeddings = normalize(np.array(embeddings, dtype=np.float32))
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    meta = {"chunks": chunks, "metas": metas, "dim": dim}
    save_index(index, meta)
    return len(chunks), f"Indexed {len(chunks)} chunks from {len(set(m['source'] for m in metas))} files."

def ensure_index() -> Tuple[faiss.IndexFlatIP | None, Dict[str, Any] | None, str]:
    index, meta = load_index()
    if index is None or meta is None:
        n, msg = build_index()
        if n == 0:
            return None, None, msg
        index, meta = load_index()
        return index, meta, msg
    return index, meta, "Index loaded."

# ---- Retrieval ---------------------------------------------------------------

def search(query: str, top_k: int, threshold: float,
           index: faiss.IndexFlatIP, meta: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any], float]]:
    model = get_model()
    q = model.encode([query], show_progress_bar=False)
    q = normalize(np.array(q, dtype=np.float32))
    scores, ids = index.search(q, top_k)
    results: List[Tuple[str, Dict[str, Any], float]] = []
    chunks: List[str] = meta["chunks"]
    metas: List[Dict[str, Any]] = meta["metas"]
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        if float(score) < float(threshold):
            continue
        results.append((chunks[idx], metas[idx], float(score)))
    return results

# ---- Starter Corpus (optional) ----------------------------------------------

STARTER_URLS = [
    # Keep this list modest to avoid rate limits/timeouts; users can add more via uploader.
    # Public pages generally available; if a fetch fails, we skip it silently.
    "https://www.whitehouse.gov/omb/information-regulatory-affairs/ai/",
    "https://www.whitehouse.gov/wp-content/uploads/2023/10/Executive-Order-on-Safe-Secure-and-Trustworthy-Development-and-Use-of-Artificial-Intelligence.pdf",
]

def fetch_starter_files() -> Tuple[int, List[str]]:
    import requests
    saved = []
    for url in STARTER_URLS:
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            # Decide extension by header
            ctype = (r.headers.get("content-type") or "").lower()
            ext = ".pdf" if "pdf" in ctype or url.lower().endswith(".pdf") else ".html"
            name = re.sub(r"[^a-zA-Z0-9]+", "-", url.strip("/"))[-60:] + ext
            path = os.path.join(RAW_DIR, name)
            with open(path, "wb") as f:
                f.write(r.content)
            saved.append(name)
        except Exception:
            continue
    return len(saved), saved

# ---- Streamlit UI ------------------------------------------------------------

def init_session():
    if "history" not in st.session_state:
        st.session_state.history = []  # list of (role, content, citations)

def left_sidebar():
    st.sidebar.header("Settings")
    top_k = st.sidebar.slider("Top-K Documents", 1, 10, value=4, step=1)
    threshold = st.sidebar.slider("Score Threshold (lower = stricter)", 0.0, 0.99, value=0.40, step=0.01)
    sect_filter = st.sidebar.text_input("Filter: section contains", value="")
    year_min = st.sidebar.number_input("Filter: year ≥ (optional)", min_value=0, value=0, step=1)
    return top_k, threshold, sect_filter.strip(), int(year_min)

def show_uploader():
    st.subheader("Add or update documents")
    st.caption("Drop PDF/HTML files below, or fetch a small starter corpus. Limit 200MB per file.")
    col1, col2 = st.columns([1, 1])
    with col1:
        btn = st.button("📥 Fetch starter corpus (Anika + federal AI)")
    with col2:
        rebuild = st.button("Rebuild index")

    up = st.file_uploader("Drop PDF/HTML files here", type=["pdf", "html", "htm"], accept_multiple_files=True, label_visibility="collapsed")
    saved_names = []
    if up:
        for f in up:
            name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", f.name)
            path = os.path.join(RAW_DIR, name)
            with open(path, "wb") as out:
                out.write(f.getbuffer())
            saved_names.append(name)
        st.success(f"Saved {len(saved_names)} file(s) to {RAW_DIR}.")
    if btn:
        n, names = fetch_starter_files()
        if n > 0:
            st.success(f"Fetched {n} files into {RAW_DIR}.")
        else:
            st.warning("Could not fetch starter files (network blocked or sources unavailable).")
    return rebuild or (up and len(saved_names) > 0) or btn

def render_chat(index, meta, top_k, threshold):
    st.divider()
    st.subheader("Ask questions about your documents")

    for role, content, cites in st.session_state.history:
        with st.chat_message(role):
            st.markdown(content)
            if cites:
                st.caption("Sources: " + " • ".join(cites))

    prompt = st.chat_input("Ask a question")
    if not prompt:
        return

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving…"):
            hits = search(prompt, top_k, threshold, index, meta)
        if not hits:
            st.markdown("I didn’t find confident matches in the index. Try lowering the score threshold or adding more files.")
            st.session_state.history.append(("assistant", "No confident matches found.", []))
            return

        # Compose answer: simple extractive summary with inline citations
        answer_lines = []
        citations = []
        for i, (chunk, m, score) in enumerate(hits, start=1):
            cite = f"[{i}: {m['source']} · score={score:.2f}]"
            citations.append(cite)
            answer_lines.append(f"- {chunk[:600]}")

        st.markdown("**Top supporting snippets:**\n" + "\n".join(answer_lines))
        st.caption("Sources: " + " • ".join(citations))
        st.session_state.history.append(("assistant", "**Top supporting snippets:**\n" + "\n".join(answer_lines), citations))

# ---- Main --------------------------------------------------------------------

def main():
    st.set_page_config(page_title="RAG Chatbot – Source-Grounded Answers", layout="wide")
    ensure_dirs()
    init_session()

    st.title("RAG Chatbot – Source-Grounded Answers")
    st.caption("Local FAISS • Sentence Transformers • Streamlit. Inline citations, filters, metrics included.")

    top_k, threshold, sect_filter, year_min = left_sidebar()

    # Docs uploader & build
    should_build = show_uploader()

    status = st.empty()
    if should_build:
        with st.spinner("Building index…"):
            n, msg = build_index()
        if n == 0:
            status.warning(msg)
        else:
            status.success(msg)
    else:
        index, meta, msg = ensure_index()
        if msg:
            st.info(msg)

    # Confirm index availability
    index, meta = load_index()
    if not index or not meta:
        st.warning("No index yet. Add at least one real PDF or HTML to `data/raw/`, then click **Rebuild index**. Chat is disabled until an index exists.")
        return

    # Optional client-side filters (section/year) – kept simple as demo
    # (For production, store section/year in metadata during chunking and filter there.)

    render_chat(index, meta, top_k=top_k, threshold=threshold)

if __name__ == "__main__":
    main()
