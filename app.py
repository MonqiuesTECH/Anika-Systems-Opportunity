# app.py
# Anika Systems – RAG Chatbot (FAISS + SentenceTransformers + Streamlit)
# UI: black & blue, concise grounded answers, no external LLMs
# Author: Monique Bruce

from __future__ import annotations

import os
import io
import re
import json
import time
import shutil
import string
import datetime as dt
from dataclasses import dataclass
from typing import List, Tuple, Dict, Iterable

import numpy as np
import pandas as pd
import streamlit as st

# Lightweight parsing
from bs4 import BeautifulSoup
from pypdf import PdfReader

# Embeddings + Vector index
from sentence_transformers import SentenceTransformer
import faiss

# ----------- Constants / Paths -----------
APP_TITLE = "Anika Systems"
POWERED_BY = "Powered by Monique Bruce"

RAW_DIR = "data/raw"
IDX_DIR = "data/index"
DOCS_MAP_PATH = os.path.join(IDX_DIR, "docs.parquet")
FAISS_PATH = os.path.join(IDX_DIR, "faiss.index")
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # small, fast

# Starter corpus (public pages/PDFs). We expect some to fail on locked networks; that's ok.
STARTER_URLS: List[str] = [
    # Anika Systems public site (HTML pages)
    "https://anikasystems.com/",
    "https://anikasystems.com/capabilities/",
    "https://anikasystems.com/careers/",
    "https://anikasystems.com/insights/",
    "https://anikasystems.com/contracts/",
    "https://anikasystems.com/contact/",
    # Federal AI / policy primers (HTML or PDFs) — safe/public examples
    "https://www.ai.gov/",
    "https://www.whitehouse.gov/ostp/ai-bill-of-rights/",
    "https://www.nist.gov/artificial-intelligence",
    "https://www.nist.gov/system/files/documents/2023/01/26/AI_RMF_1.0.pdf",
    "https://www.gsa.gov/technology/government-it-initiatives/emerging-citizen-technology/artificial-intelligence",
    "https://www.doi.gov/sites/default/files/ai-governance-framework.pdf",
    # Add a few generic MLOps/transformers references (public PDFs) for body
    "https://arxiv.org/pdf/2106.04554.pdf",
    "https://arxiv.org/pdf/2005.14165.pdf",
    "https://arxiv.org/pdf/1706.03762.pdf",
]

# ----------- Utilities -----------
def ensure_dirs() -> None:
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(IDX_DIR, exist_ok=True)

def sanitize_filename(name: str) -> str:
    name = re.sub(r"[^\w\-.]+", "_", name.strip())
    return name[:200] if len(name) > 200 else name

def year_from_text(text: str) -> int:
    years = [int(y) for y in re.findall(r"\b(19|20)\d{2}\b", text)]
    return max(years) if years else 0

def clean_text(t: str) -> str:
    t = re.sub(r"\s+", " ", t).strip()
    return t

def chunk_text(text: str, chunk_size: int = 1100, overlap: int = 150) -> List[str]:
    """Simple token-agnostic chunking by characters with overlap."""
    text = clean_text(text)
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks

@dataclass
class Chunk:
    text: str
    source: str
    section: str
    year: int

# ----------- Loaders -----------
def load_pdf(file_bytes: bytes, source_name: str) -> Tuple[str, int]:
    """Extract text per page, return joined text + most recent year guess."""
    reader = PdfReader(io.BytesIO(file_bytes))
    pages_txt = []
    for p in reader.pages:
        try:
            pages_txt.append(p.extract_text() or "")
        except Exception:
            pages_txt.append("")
    full = "\n".join(pages_txt)
    return clean_text(full), year_from_text(full)

def load_html(html_bytes: bytes, source_name: str) -> Tuple[str, int]:
    soup = BeautifulSoup(html_bytes, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    # Prefer article/main if present
    main = soup.find(["article", "main"]) or soup.body or soup
    text = main.get_text(separator=" ")
    return clean_text(text), year_from_text(text)

def load_file(path: str) -> Tuple[str, int]:
    with open(path, "rb") as f:
        raw = f.read()
    if path.lower().endswith(".pdf"):
        return load_pdf(raw, os.path.basename(path))
    else:
        return load_html(raw, os.path.basename(path))

# ----------- Embeddings / Index -----------
@st.cache_resource(show_spinner=False)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMB_MODEL_NAME, device="cpu")

def embed_texts(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384), dtype="float32")
    vecs = model.encode(texts, batch_size=64, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
    return vecs.astype("float32")

def build_faiss(d: int) -> faiss.IndexFlatIP:
    # inner product index; we use normalized vectors so IP == cosine
    return faiss.IndexFlatIP(d)

def save_index(index: faiss.IndexFlatIP) -> None:
    faiss.write_index(index, FAISS_PATH)

def load_index() -> faiss.IndexFlatIP | None:
    if os.path.exists(FAISS_PATH):
        return faiss.read_index(FAISS_PATH)
    return None

# ----------- Document pipeline -----------
def discover_raw_files() -> List[str]:
    files = []
    for root, _, fnames in os.walk(RAW_DIR):
        for fn in fnames:
            if fn.lower().endswith((".pdf", ".html", ".htm")):
                files.append(os.path.join(root, fn))
    files.sort()
    return files

def ingest_to_chunks(files: List[str], section_contains: str = "", year_min: int = 0) -> List[Chunk]:
    chunks: List[Chunk] = []
    for path in files:
        try:
            txt, y = load_file(path)
            if year_min and y and y < year_min:
                continue
            # Derive a light "section" from filename and H1-ish patterns
            section = os.path.splitext(os.path.basename(path))[0].replace("_", " ")
            if section_contains and section_contains.lower() not in section.lower():
                # Try inside text, too
                if section_contains.lower() not in txt.lower():
                    continue
            for c in chunk_text(txt):
                chunks.append(Chunk(text=c, source=os.path.basename(path), section=section, year=y))
        except Exception:
            # Skip unreadable files
            continue
    return chunks

def persist_docmap(chunks: List[Chunk]) -> pd.DataFrame:
    df = pd.DataFrame([vars(c) for c in chunks])
    df.to_parquet(DOCS_MAP_PATH, index=False)
    return df

def load_docmap() -> pd.DataFrame | None:
    if os.path.exists(DOCS_MAP_PATH):
        return pd.read_parquet(DOCS_MAP_PATH)
    return None

def rebuild_index(section_filter: str, year_min: int, status_placeholder) -> Tuple[int, int]:
    ensure_dirs()
    files = discover_raw_files()
    if not files:
        return 0, 0

    status_placeholder.info("Reading and chunking documents…")
    chunks = ingest_to_chunks(files, section_filter, year_min)
    if not chunks:
        # Clear any previous index if filters exclude everything
        if os.path.exists(FAISS_PATH):
            os.remove(FAISS_PATH)
        if os.path.exists(DOCS_MAP_PATH):
            os.remove(DOCS_MAP_PATH)
        return 0, 0

    df = persist_docmap(chunks)
    texts = df["text"].tolist()

    status_placeholder.info(f"Embedding {len(texts):,} chunks…")
    model = get_embedder()
    vecs = embed_texts(model, texts)  # (n, 384)
    index = build_faiss(vecs.shape[1])
    index.add(vecs)
    save_index(index)

    status_placeholder.success(f"Index built: {len(texts):,} chunks from {df['source'].nunique()} files.")
    return len(texts), df["source"].nunique()

# ----------- Retrieval & Answering -----------
def search(query: str, top_k: int, score_thresh: float) -> List[Dict]:
    model = get_embedder()
    index = load_index()
    docs = load_docmap()
    if index is None or docs is None or docs.empty:
        return []
    qv = embed_texts(model, [query])
    if qv.shape[0] == 0:
        return []
    scores, idxs = index.search(qv, top_k)
    scores = scores.flatten().tolist()
    idxs = idxs.flatten().tolist()
    hits = []
    for s, i in zip(scores, idxs):
        if i < 0:
            continue
        if s < score_thresh:  # cosine similarity threshold (0..1)
            continue
        row = docs.iloc[i]
        hits.append(
            dict(
                score=float(s),
                text=row["text"],
                source=row["source"],
                section=row["section"],
                year=int(row["year"]) if row["year"] else 0,
            )
        )
    return hits

def concise_answer(query: str, hits: List[Dict]) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Extractive, short answer composed from top hits.
    Returns (answer_text, [(source, section), ...])
    """
    if not hits:
        return ("I couldn’t find that in the indexed documents. Try lowering the score threshold or adding more files.", [])

    # Merge top few chunks and pick salient sentences that overlap with query terms.
    terms = set([w.lower().strip(string.punctuation) for w in query.split() if len(w) > 2])
    chosen: List[str] = []
    citations: List[Tuple[str, str]] = []
    for h in hits[:6]:
        sents = re.split(r"(?<=[.!?])\s+", h["text"])
        # score sentences by term overlap + length prior
        scored = []
        for s in sents:
            toks = set([w.lower().strip(string.punctuation) for w in s.split() if len(w) > 2])
            inter = len(terms & toks)
            if inter == 0:
                continue
            scored.append((inter / (len(toks) + 1e-6), s))
        if scored:
            best = max(scored, key=lambda x: x[0])[1]
            best = clean_text(best)
            if best and best not in chosen:
                chosen.append(best)
                citations.append((h["source"], h["section"]))
        if len(chosen) >= 4:
            break

    if not chosen:
        # fallback: take the first lines from best hit
        top = clean_text(hits[0]["text"])
        chosen = [top[:300]]

    # Constrain to 2–4 sentences overall
    text = " ".join(chosen)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    summary = " ".join(sentences[:4])

    # Enforce shortness
    if len(summary) > 750:
        summary = summary[:730].rsplit(" ", 1)[0] + "…"

    # Deduplicate citations (show up to 4)
    seen = set()
    uniq_cites = []
    for src, sec in citations:
        key = (src, sec)
        if key in seen:
            continue
        seen.add(key)
        uniq_cites.append(key)
        if len(uniq_cites) >= 4:
            break

    return summary, uniq_cites

# ----------- Fetch helper (optional) -----------
def http_get(url: str, timeout: int = 15) -> Tuple[bytes | None, str]:
    """
    Minimal requests-like fetch using urllib to avoid extra deps.
    Returns (bytes, ext) where ext is '.html' or '.pdf'. Returns (None, '') on failure.
    """
    try:
        import urllib.request
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (RAGBot/1.0 MoniqueBruce)"},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
            ctype = resp.headers.get("Content-Type", "")
            if "pdf" in ctype or url.lower().endswith(".pdf"):
                return data, ".pdf"
            return data, ".html"
    except Exception:
        return None, ""

def fetch_starter(status_placeholder) -> Tuple[int, int]:
    ensure_dirs()
    saved = 0
    failed = 0
    for i, url in enumerate(STARTER_URLS, 1):
        status_placeholder.info(f"Fetching {i}/{len(STARTER_URLS)}…")
        data, ext = http_get(url)
        if not data or not ext:
            failed += 1
            continue
        fname = sanitize_filename(url.split("://", 1)[-1])
        path = os.path.join(RAW_DIR, f"{fname}{ext}")
        try:
            with open(path, "wb") as f:
                f.write(data)
            saved += 1
        except Exception:
            failed += 1
    if saved == 0:
        status_placeholder.warning("Could not fetch starter files (network blocked or sources unavailable). You can upload files instead.")
    else:
        status_placeholder.success(f"Fetched {saved} file(s). Some sources may be blocked on this network (failed={failed}). You can add more via upload.")
    return saved, failed

# ----------- Streamlit App -----------
def set_theme():
    st.set_page_config(page_title=APP_TITLE, page_icon="🤖", layout="wide")
    st.markdown(
        """
        <style>
        :root { --mb-blue: #0b74de; --mb-dark: #0e1117; --mb-slate: #1b2230; --mb-text: #e6eef8; }
        .stApp { background: var(--mb-dark); color: var(--mb-text); }
        header, .css-18ni7ap, .e16nr0p30 { background: var(--mb-dark) !important; }
        .block-container { padding-top: 1.5rem; }
        .stButton>button, .stDownloadButton>button { background: var(--mb-blue); color: white; border: 0; border-radius: 8px; }
        .stSlider>div>div>div>div { background: var(--mb-blue) !important; }
        .stTextInput>div>div>input, .stTextArea>div>div>textarea { background: var(--mb-slate); color: var(--mb-text); }
        .stFileUploader>div { background: var(--mb-slate); border-radius: 8px; }
        .badge { display:inline-block; padding:2px 8px; border-radius: 8px; background:#0b74de22; color:#7fb7ff; border:1px solid #0b74de55; font-size: 0.8rem; }
        .footer { color:#9fb7d6; font-size:0.9rem; text-align:center; margin-top: 0.5rem; }
        .title { font-size: 2.2rem; font-weight: 800; color: #cfe6ff; text-shadow: 0 2px 8px rgba(11,116,222,0.35); }
        .panel { background: #101622; border:1px solid #1f2b42; border-radius: 12px; padding: 1rem 1.2rem; }
        .code { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; }
        </style>
        """,
        unsafe_allow_html=True,
    )

def sidebar_controls():
    st.sidebar.header("Settings")
    top_k = st.sidebar.slider("Top-K Documents", 1, 12, 4, 1)
    score = st.sidebar.slider("Score Threshold (lower = stricter)", 0.30, 0.90, 0.40, 0.01)
    section_contains = st.sidebar.text_input("Filter: section contains", value="")
    year_min = st.sidebar.number_input("Filter: year ≥ (optional)", min_value=0, value=0, step=1)
    st.sidebar.markdown(
        f"""
        <div class="panel">
          <div class="badge">Tip</div> Place <span class="code">30–50</span> PDFs/HTML into <span class="code">{RAW_DIR}/</span>,
          or use the uploader below.
        </div>
        """,
        unsafe_allow_html=True,
    )
    return top_k, score, section_contains, year_min

def ui_header():
    st.markdown(f'<div class="title">{APP_TITLE}</div>', unsafe_allow_html=True)
    st.caption("RAG Chatbot – Source-Grounded Answers")
    st.markdown("<div class='badge'>Local FAISS • Sentence Transformers • Streamlit. Inline citations, filters, metrics, and upload/fetch tools included.</div>", unsafe_allow_html=True)
    st.write("")

def ui_uploader_and_fetch() -> None:
    with st.container():
        st.subheader("Add or update documents")
        col_up, col_btn = st.columns([6, 2])
        with col_up:
            uploads = st.file_uploader("Drop PDF/HTML files", type=["pdf", "html", "htm"], accept_multiple_files=True, label_visibility="collapsed")
            if uploads:
                ensure_dirs()
                saved = 0
                for f in uploads:
                    ext = ".pdf" if f.type == "application/pdf" or f.name.lower().endswith(".pdf") else ".html"
                    path = os.path.join(RAW_DIR, sanitize_filename(f.name))
                    if not path.lower().endswith(ext):
                        path += ext
                    with open(path, "wb") as out:
                        out.write(f.read())
                    saved += 1
                st.success(f"Saved {saved} file(s) into {RAW_DIR}/. Click Rebuild index next.")
        with col_btn:
            fetch_area = st.empty()
            if st.button("Fetch starter corpus (Anika + federal AI)"):
                with st.spinner("Fetching starter corpus…"):
                    saved, failed = fetch_starter(fetch_area)
            st.button("Rebuild index", key="rebuild-top")

        # Status banners
        files = discover_raw_files()
        if files:
            st.info(f"Files detected in {RAW_DIR}/. Click Rebuild index after updates.")
        else:
            st.warning(f"No index yet. Add PDFs/HTML to {RAW_DIR}/ or upload below, then click Rebuild index.")
        if not os.path.exists(FAISS_PATH):
            st.info("Index not available. Add or fetch documents and click Rebuild index. Chat is disabled until an index exists.")

def ui_rebuild(section_filter: str, year_min: int) -> None:
    if st.button("Rebuild index", key="rebuild-bottom"):
        pass  # handled below (we wire both buttons to the same action path)

    # One action handler for either button
    if st.session_state.get("rebuild-top") or st.session_state.get("rebuild-bottom"):
        status = st.empty()
        with st.spinner("Running bootstrap_index()…"):
            n_chunks, n_files = rebuild_index(section_filter, year_min, status)
        if n_files > 0:
            st.success(f"Index ready: {n_chunks:,} chunks from {n_files} file(s).")
        else:
            st.warning("No documents indexed. Adjust filters or add files.")

def ui_chat(top_k: int, score_thresh: float):
    st.write("---")
    st.subheader("Chat")
    if not os.path.exists(FAISS_PATH):
        st.info("Chat is disabled until an index exists.")
        return

    # Chat history
    if "history" not in st.session_state:
        st.session_state.history = []

    # Render history
    for role, msg in st.session_state.history:
        with st.chat_message(role):
            st.write(msg)

    prompt = st.chat_input("Ask about these documents…")
    if not prompt:
        return

    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.history.append(("user", prompt))

    with st.chat_message("assistant"):
        with st.spinner("Retrieving…"):
            hits = search(prompt, top_k=top_k, score_thresh=score_thresh)
            answer, cites = concise_answer(prompt, hits)

        st.write(answer)
        if cites:
            # Group citations & render
            srcs = [f"• **{src}** — _{sec}_" for src, sec in cites]
            st.caption("Sources:\n" + "\n".join(srcs))
        else:
            st.caption("No sources met the threshold; response withheld to avoid speculation.")

        st.session_state.history.append(("assistant", answer))

def footer():
    st.markdown(f"<div class='footer'>{POWERED_BY}</div>", unsafe_allow_html=True)

def main():
    set_theme()
    top_k, score, section_filter, year_min = sidebar_controls()
    ui_header()
    ui_uploader_and_fetch()

    # If either rebuild button was pressed, rebuild now
    if st.session_state.get("rebuild-top") or st.session_state.get("rebuild-bottom"):
        status = st.empty()
        with st.spinner("Building index…"):
            rebuild_index(section_filter, year_min, status)
        # Clear the pressed state so rebuild is one-shot
        st.session_state["rebuild-top"] = False
        st.session_state["rebuild-bottom"] = False

    ui_chat(top_k=top_k, score_thresh=score)
    footer()

if __name__ == "__main__":
    ensure_dirs()
    main()
