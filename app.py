# app.py
from __future__ import annotations
import os, sys, time, pickle, re, json, pathlib, tempfile, shutil
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Iterable, Optional

import streamlit as st
import numpy as np

# ===== Local modules =====
# Keep all logic in /src so Streamlit hot-reload behaves
from src.loaders import ensure_dirs, load_local_and_uploaded, fetch_starter_corpus
from src.clean_chunk import clean_and_chunk
from src.embed_index import build_or_load_faiss, search_index
from src.prompts import build_system_prompt
from src.retrieve import answer_with_citations
from src.metrics import compute_hit_ratio

APP_TITLE = "Anika Systems"
POWERED_BY = "Powered by Monique Bruce"
RAW_DIR = pathlib.Path("data/raw")
IDX_DIR = pathlib.Path("data/index")
META_PKL = IDX_DIR / "meta.pkl"
MAX_UPLOAD_MB = 200

# ---------- Page / Theme ----------
st.set_page_config(page_title=APP_TITLE, page_icon="💬", layout="wide")

# Force a clean dark blue look without needing secrets; works on Streamlit Cloud 1.49
DARK_CSS = """
<style>
:root { --brand-blue:#1677ff; }
.block-container { padding-top: 1.2rem; }
h1,h2,h3,h4 { color: #e6f0ff; }
.stApp { background: #0b1220; color: #e6f0ff; }
.sidebar .sidebar-content, section[data-testid="stSidebar"] { background: #121a2b; }
div[data-baseweb="input"] input, textarea, .stTextArea textarea {
  background: #0e1728 !important; color: #e6f0ff !important; border-color: #22304a !important;
}
.stButton>button, .stDownloadButton>button {
  background: var(--brand-blue) !important; border: 0 !important; color: white !important;
}
.kbdx { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; opacity:.75 }
footer { visibility: hidden; } /* saves vertical space */
.small { font-size: 0.85rem; opacity: .85 }
.card { background:#0e1728; padding: .75rem 1rem; border-radius: .75rem; border: 1px solid #22304a; }
.badge { background:#122241; color:#cde2ff; padding:.25rem .5rem; border-radius:.5rem; font-size:.75rem; }
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

# ---------- Sidebar (controls) ----------
with st.sidebar:
    st.markdown("### Settings")
    top_k = st.slider("Top-K Documents", 1, 10, value=4)
    score_thresh = st.slider("Score Threshold (lower = stricter)", 0.0, 1.0, value=0.40, step=0.05)
    section_filter = st.text_input("Filter: section contains", value="")
    min_year = st.number_input("Filter: year ≥ (optional)", min_value=0, value=0, step=1)
    st.markdown(
        '<div class="card small">Tip: place <b>~30–50 PDFs/HTML</b> into <kbd class="kbdx">data/raw/</kbd> '
        "or use the uploader below. Click <b>Rebuild index</b> afterwards.</div>", 
        unsafe_allow_html=True
    )

# ---------- Header ----------
left, right = st.columns([0.78, 0.22])
with left:
    st.markdown(f"# {APP_TITLE}")
    st.markdown("**RAG Chatbot – Source-Grounded Answers**  \n"
                "Local FAISS • Sentence Transformers • Streamlit. Inline citations, filters, metrics, and upload/fetch tools included.")
with right:
    st.markdown(f'<div style="text-align:right;margin-top:1.25rem;"><span class="badge">{POWERED_BY}</span></div>', unsafe_allow_html=True)

# ---------- Ensure folders ----------
ensure_dirs(RAW_DIR, IDX_DIR)

# ---------- Upload / fetch area ----------
st.subheader("Add or update documents")
uploader = st.file_uploader(
    "Drop PDF/HTML files", type=["pdf", "html", "htm"],
    accept_multiple_files=True, help=f"Limit {MAX_UPLOAD_MB}MB per file • PDF, HTML, HTM"
)
c1, c2 = st.columns([0.32, 0.68])
with c1:
    do_rebuild = st.button("Rebuild index", type="primary", use_container_width=True)
with c2:
    fetch_clicked = st.button("Fetch starter corpus (Anika + federal AI)", use_container_width=True)

# Feedback ribbons
status_box = st.empty()

if fetch_clicked:
    with st.spinner("Fetching starter corpus…"):
        fetched, failed = fetch_starter_corpus(RAW_DIR, target_count=35)
    msg = f"Fetched {len(fetched)} file(s)" + (f"; some sources blocked or failed={len(failed)}" if failed else "")
    status_box.info(msg)

# Save uploads to disk
if uploader:
    saved = []
    for f in uploader:
        # guard size: Streamlit already enforces; we still cap
        if len(f.getbuffer()) > MAX_UPLOAD_MB * 1024 * 1024:
            st.warning(f"Skipped {f.name}: over {MAX_UPLOAD_MB}MB")
            continue
        dest = RAW_DIR / f.name
        with open(dest, "wb") as out:
            out.write(f.getbuffer())
        saved.append(dest.name)
    if saved:
        status_box.success(f"Uploaded {len(saved)} file(s). Click **Rebuild index** next.")

# ---------- Build index ----------
def rebuild_index() -> Tuple[int, int]:
    docs = load_local_and_uploaded(RAW_DIR)
    if not docs:
        return (0, 0)

    # Clean + chunk
    chunks, meta = clean_and_chunk(
        docs,
        min_year=min_year if min_year else None,
        section_substr=section_filter or None
    )

    # Build embeddings + FAISS
    idx, built_meta = build_or_load_faiss(
        chunks, meta, idx_dir=IDX_DIR, force_rebuild=True
    )

    # Persist meta for later search/citation
    with open(META_PKL, "wb") as fh:
        pickle.dump(built_meta, fh)

    return len(chunks), len(set([m["source"] for m in built_meta]))

# Rebuild if asked
if do_rebuild:
    with st.spinner("Indexing… this runs locally (no external APIs)."):
        n_chunks, n_files = rebuild_index()
    if n_files == 0:
        st.warning("No index yet. Add at least one real PDF or HTML to data/raw/, then click Rebuild index.")
    else:
        st.success(f"Index built: {n_chunks} chunks from {n_files} files.")

# Show index status
index_ready = IDX_DIR.exists() and any(IDX_DIR.glob("*.faiss"))
if META_PKL.exists():
    try:
        with open(META_PKL, "rb") as fh:
            meta_preview: List[Dict[str, Any]] = pickle.load(fh)
        n_files = len(set(m["source"] for m in meta_preview))
        n_chunks = len(meta_preview)
        st.info(f"Index built: {n_chunks} chunks from {n_files} files.")
    except Exception:
        index_ready = False

st.divider()

# ---------- Chat ----------
st.subheader("Ask about these documents…")
q = st.text_input("Ask", key="q", placeholder="e.g., Who is Anika Systems? What contracts? What AI/automation work?")
ask = st.button("Ask", type="primary")

if ask:
    if not index_ready:
        st.error("Index not available. Add or fetch documents and click **Rebuild index**.")
    elif not q.strip():
        st.warning("Please type a question.")
    else:
        # Load meta + search + answer
        with open(META_PKL, "rb") as fh:
            meta: List[Dict[str, Any]] = pickle.load(fh)
        hits = search_index(
            query=q,
            idx_dir=IDX_DIR,
            top_k=top_k,
            score_threshold=score_thresh,
            meta=meta
        )
        if not hits:
            st.warning("No supporting passages found. Try a broader query or lower the threshold.")
        else:
            sys_prompt = build_system_prompt(org_name="Anika Systems")
            answer, used = answer_with_citations(
                query=q,
                hits=hits,
                system_prompt=sys_prompt,
                concise=True,                 # keep answers short
                max_words=90,                 # keep it crisp
            )
            # Render
            box = st.container()
            with box:
                st.markdown(f"#### 🧠 {q.strip()}")
                st.markdown(answer)
                # cite sources
                st.markdown("**Sources:**")
                for u in used:
                    st.markdown(f"- `{u.get('source','')}`")

st.caption(POWERED_BY)

# --------- Safety: always clear temporary assignment keys last (Streamlit 1.49) ---------
# Streamlit 1.49+ raises when you assign brand-new keys outside of initialization;
# keep all keys on st.session_state behind setdefault.
st.session_state.setdefault("rebuild-top", False)
