import os
import re
import json
import time
import math
import hashlib
import pathlib
import random
import itertools
from dataclasses import dataclass
from typing import List, Dict, Tuple

import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from pypdf import PdfReader

import numpy as np
import faiss

from sentence_transformers import SentenceTransformer

# -----------------------------
# Paths & constants
# -----------------------------
RAW_DIR = pathlib.Path("data/raw")
CLEAN_DIR = pathlib.Path("data/clean")
INDEX_DIR = pathlib.Path("data/index")
INDEX_PATH = INDEX_DIR / "faiss.index"
META_PATH = INDEX_DIR / "meta.json"

EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K_DEFAULT = 4
SCORE_THRESH_DEFAULT = 0.40
CHUNK_SIZE = 1200        # characters
CHUNK_OVERLAP = 200      # characters
REQUEST_TIMEOUT = 12     # seconds
MAX_RETRIES = 3

# -----------------------------
# Streamlit page setup (dark / blue)
# -----------------------------
st.set_page_config(
    page_title="Anika Systems",
    page_icon="🤖",
    layout="wide",
)

DARK_CSS = """
<style>
/* Global */
:root { --primary:#1f6feb; --bg:#0b1620; --panel:#111d2a; --text:#e6f0ff; }
#MainMenu, header { visibility: hidden; }
section.main { background: var(--bg); color: var(--text); }
div.stButton>button { background: var(--primary); color:white; border-radius:10px; border:0; }
div.stSlider > div[data-baseweb="slider"] { color: var(--text); }
.sidebar .sidebar-content { background: #0f1b27 !important; }
.css-1cypcdb, .stMarkdown, .stTextInput, .stTextArea { color: var(--text) !important; }
.stChatMessage { background: var(--panel); color: var(--text); border-radius: 12px; }
.stChatMessage .stMarkdown p { color: var(--text); }
.st-bf { color: var(--text); }
a { color: #78a6ff; }
footer { visibility: hidden; }
.rag-footer { color: #b8c7e0; font-size:12px; text-align:center; padding: 12px 0 2px 0; }
.rag-card { background: var(--panel); border-radius: 12px; padding: 14px; border: 1px solid #1b2a3a; }
.rag-badge { background:#0e274e; color:#bcd7ff; padding:.25rem .5rem; border-radius:8px; font-size:12px; }
.rag-muted { color:#9fb0c8; font-size:12px; }
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

# -----------------------------
# Utilities
# -----------------------------
def ensure_dirs():
    for p in [RAW_DIR, CLEAN_DIR, INDEX_DIR]:
        p.mkdir(parents=True, exist_ok=True)

def slugify(s: str) -> str:
    s = re.sub(r"https?://", "", s)
    s = re.sub(r"[^a-zA-Z0-9._-]+", "-", s).strip("-")
    return s[:120] if s else hashlib.md5(s.encode()).hexdigest()

def save_file_bytes(url: str, data: bytes, ext: str) -> pathlib.Path:
    fname = slugify(url) or hashlib.md5(url.encode()).hexdigest()
    path = RAW_DIR / f"{fname}.{ext}"
    path.write_bytes(data)
    return path

def get_with_retries(url: str) -> requests.Response | None:
    headers_pool = [
        {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Safari/537.36"},
        {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X) Chrome/126"},
        {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Firefox/128"},
    ]
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            headers = random.choice(headers_pool)
            r = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT, allow_redirects=True)
            if r.status_code == 200 and r.content:
                return r
        except requests.RequestException:
            pass
        time.sleep(0.75 * attempt)  # backoff
    return None

def is_pdf_response(resp: requests.Response) -> bool:
    ctype = resp.headers.get("Content-Type", "").lower()
    return ("application/pdf" in ctype) or resp.url.lower().endswith(".pdf")

def pick_links(base_url: str, html: bytes) -> Tuple[List[str], List[str]]:
    soup = BeautifulSoup(html, "lxml")
    pdfs, pages = [], []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        full = urljoin(base_url, href)
        if any(full.lower().endswith(suf) for suf in [".pdf"]):
            pdfs.append(full)
        elif any(full.lower().endswith(suf) for suf in [".html", ".htm", "/"]):
            pages.append(full)
    # de-dup & keep within sensible hosts
    def norm(u): return u.split("#")[0]
    pdfs = list(dict.fromkeys(map(norm, pdfs)))
    pages = list(dict.fromkeys(map(norm, pages)))
    return pdfs, pages

# -----------------------------
# Fetch corpus (auto-fallback to reach ~50 docs)
# -----------------------------
SEED_SOURCES: Dict[str, List[str]] = {
    # 1) Anika public pages
    "anika": [
        "https://www.anikasystems.com/",
        "https://www.anikasystems.com/who-we-are",
        "https://www.anikasystems.com/capabilities",
        "https://www.anikasystems.com/insights",
        "https://www.anikasystems.com/careers",
        "https://www.anikasystems.com/contact",
        "https://www.anikasystems.com/privacy-policy"
    ],
    # 2) Federal AI / data / cloud guidance (public, stable)
    "nist": [
        "https://www.nist.gov/artificial-intelligence",
        "https://www.nist.gov/publications?type=all&topics=214666",
    ],
    "gsa": ["https://www.gsa.gov/technology/artificial-intelligence"],
    "omb": ["https://www.whitehouse.gov/omb/briefing-room/"],
    "cio": ["https://www.cio.gov/"],
    # 3) Backup sources with many PDFs
    "nasa": ["https://www.nasa.gov/ai/"],
    "doe": ["https://www.energy.gov/ai-and-technology-office/artificial-intelligence-and-technology-office"],
}

ALLOWED_HOSTS = {
    "anikasystems.com", "www.anikasystems.com",
    "nist.gov", "www.nist.gov",
    "gsa.gov", "www.gsa.gov",
    "whitehouse.gov", "www.whitehouse.gov",
    "cio.gov", "www.cio.gov",
    "nasa.gov", "www.nasa.gov",
    "energy.gov", "www.energy.gov",
}

def host_allowed(url: str) -> bool:
    return urlparse(url).netloc in ALLOWED_HOSTS

def crawl_and_save(target_count: int = 50) -> Dict[str, int]:
    ensure_dirs()
    saved, failed = 0, 0
    visited: set[str] = set()

    queues = list(itertools.chain.from_iterable(SEED_SOURCES.values()))
    random.shuffle(queues)

    while queues and saved < target_count:
        url = queues.pop(0)
        if url in visited or not host_allowed(url):
            continue
        visited.add(url)

        resp = get_with_retries(url)
        if not resp:
            failed += 1
            continue

        if is_pdf_response(resp):
            try:
                save_file_bytes(resp.url, resp.content, "pdf")
                saved += 1
            except Exception:
                failed += 1
            continue

        # HTML page
        try:
            html = resp.content
            # Save page too (some are valuable)
            save_file_bytes(resp.url, html, "html")
            saved += 1

            pdf_links, page_links = pick_links(resp.url, html)

            # Prefer PDFs first
            for purl in pdf_links[:10]:
                if saved >= target_count: break
                if purl in visited or not host_allowed(purl): continue
                visited.add(purl)
                r2 = get_with_retries(purl)
                if r2 and is_pdf_response(r2):
                    try:
                        save_file_bytes(r2.url, r2.content, "pdf")
                        saved += 1
                    except Exception:
                        failed += 1

            # Enqueue a few more pages to crawl breadth-first
            random.shuffle(page_links)
            for surl in page_links[:8]:
                if surl not in visited and host_allowed(surl):
                    queues.append(surl)

        except Exception:
            failed += 1

    return {"saved": saved, "failed": failed}

# -----------------------------
# Loaders: PDF/HTML -> clean text
# -----------------------------
def extract_text_from_pdf(path: pathlib.Path) -> str:
    try:
        reader = PdfReader(str(path))
        pages = [p.extract_text() or "" for p in reader.pages]
        return "\n".join(pages)
    except Exception:
        return ""

def extract_text_from_html(path: pathlib.Path) -> str:
    try:
        html = path.read_bytes()
        soup = BeautifulSoup(html, "lxml")
        # remove nav/script/style
        for tag in soup(["nav", "script", "style", "header", "footer", "noscript"]):
            tag.extract()
        text = soup.get_text(separator=" ")
        text = re.sub(r"\s+", " ", text).strip()
        return text
    except Exception:
        return ""

def clean_and_chunk(doc_id: str, text: str) -> List[Dict]:
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    i, n = 0, len(text)
    while i < n:
        j = min(i + CHUNK_SIZE, n)
        chunk = text[i:j]
        chunks.append({"id": f"{doc_id}:::{i}", "text": chunk})
        i = j - CHUNK_OVERLAP
        if i < 0: i = 0
        if i >= n: break
    return chunks

def build_clean_corpus() -> Tuple[List[str], List[Dict]]:
    ensure_dirs()
    file_paths = [p for p in RAW_DIR.glob("**/*") if p.suffix.lower() in [".pdf", ".html", ".htm"]]
    doc_texts: Dict[str, str] = {}

    for p in file_paths:
        if p.suffix.lower() == ".pdf":
            t = extract_text_from_pdf(p)
        else:
            t = extract_text_from_html(p)
        if len(t) < 400:
            continue
        doc_id = p.name
        (CLEAN_DIR / f"{doc_id}.txt").write_text(t, encoding="utf-8", errors="ignore")
        doc_texts[doc_id] = t

    # chunk
    all_chunks = []
    for doc_id, t in doc_texts.items():
        all_chunks.extend(clean_and_chunk(doc_id, t))
    return list(doc_texts.keys()), all_chunks

# -----------------------------
# Embedding & Index
# -----------------------------
@dataclass
class RagIndex:
    index: faiss.IndexFlatIP
    model: SentenceTransformer
    ids: List[str]
    meta: Dict[str, Dict]  # chunk_id -> {doc_id, text}

def normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / norms

def embed_texts(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    embs = model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    return np.array(embs).astype("float32")

def build_index(chunks: List[Dict]) -> RagIndex:
    model = SentenceTransformer(EMB_MODEL_NAME)
    texts = [c["text"] for c in chunks]
    embs = embed_texts(model, texts)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)

    ids = [c["id"] for c in chunks]
    meta = {c["id"]: {"doc_id": c["id"].split(":::")[0], "text": c["text"]} for c in chunks}
    # persist
    faiss.write_index(index, str(INDEX_PATH))
    META_PATH.write_text(json.dumps({"ids": ids, "meta": meta}, ensure_ascii=False))
    return RagIndex(index=index, model=model, ids=ids, meta=meta)

def load_index() -> RagIndex | None:
    if not (INDEX_PATH.exists() and META_PATH.exists()):
        return None
    index = faiss.read_index(str(INDEX_PATH))
    d = json.loads(META_PATH.read_text())
    model = SentenceTransformer(EMB_MODEL_NAME)
    return RagIndex(index=index, model=model, ids=d["ids"], meta=d["meta"])

# -----------------------------
# Retrieval & deterministic “short answer”
# -----------------------------
def retrieve(rag: RagIndex, query: str, top_k: int) -> List[Tuple[str, float]]:
    q_emb = embed_texts(rag.model, [query])
    D, I = rag.index.search(q_emb, top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx == -1: continue
        chunk_id = rag.ids[idx]
        results.append((chunk_id, float(score)))
    return results

def format_sources(rag: RagIndex, hits: List[Tuple[str, float]]) -> List[str]:
    # Show doc file names (de-duplicated)
    seen, out = set(), []
    for cid, _ in hits:
        doc = rag.meta[cid]["doc_id"]
        if doc not in seen:
            seen.add(doc)
            out.append(doc)
    return out[:5]

def synthesize_short_answer(rag: RagIndex, query: str, hits: List[Tuple[str, float]], score_thresh: float) -> str:
    # Extractive: pick 2–3 sentences with highest keyword overlap, keeps it concise & grounded
    key_terms = set([w.lower() for w in re.findall(r"[a-zA-Z]{3,}", query)])
    cand_sentences: List[Tuple[float, str]] = []
    for cid, score in hits:
        if score < score_thresh: 
            continue
        text = rag.meta[cid]["text"]
        # split to short sentences
        for s in re.split(r"(?<=[.?!])\s+", text):
            if len(s) < 30 or len(s) > 300: 
                continue
            words = set([w.lower() for w in re.findall(r"[a-zA-Z]{3,}", s)])
            overlap = len(key_terms & words)
            cand_sentences.append((overlap + score, s.strip()))
    cand_sentences.sort(key=lambda x: x[0], reverse=True)
    picks = [s for _, s in cand_sentences[:3]]
    if not picks:
        # fallback: first chunk first 2 sentences
        fallback = rag.meta[hits[0][0]]["text"]
        picks = re.split(r"(?<=[.?!])\s+", fallback)[:2]
    # keep super short
    joined = " ".join(picks)
    return re.sub(r"\s+", " ", joined)[:600]

# -----------------------------
# Sidebar controls
# -----------------------------
def init_state():
    defaults = {
        "top_k": TOP_K_DEFAULT,
        "score_thresh": SCORE_THRESH_DEFAULT,
        "rebuild_top": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

with st.sidebar:
    st.subheader("Settings")
    st.session_state["top_k"] = st.slider("Top-K Documents", 1, 10, st.session_state["top_k"])
    st.session_state["score_thresh"] = st.slider("Score Threshold (lower = stricter)", 0.10, 0.95, st.session_state["score_thresh"])
    st.caption("Tip: place ~30–50 PDFs/HTML into `data/raw/`, or use the uploader below. Then click Rebuild index.")
    st.markdown("<div class='rag-muted'>Be concise; answers cite sources.</div>", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown("## Anika Systems")
st.markdown("<span class='rag-badge'>RAG Chatbot — Source-Grounded Answers (local FAISS + MiniLM)</span>", unsafe_allow_html=True)
st.write("")

# -----------------------------
# Upload & Fetch row
# -----------------------------
ensure_dirs()

upcol, rebuildcol, fetchcol = st.columns([3, 1, 1])

with upcol:
    upload = st.file_uploader("Drag and drop files here", type=["pdf","html","htm"], accept_multiple_files=True, label_visibility="collapsed")
    if upload:
        saved_n = 0
        for f in upload:
            ext = f.name.split(".")[-1].lower()
            if ext not in ["pdf","html","htm"]: 
                continue
            path = RAW_DIR / f.name
            path.write_bytes(f.getvalue())
            saved_n += 1
        st.success(f"Saved {saved_n} file(s). Click Rebuild index next.")

with rebuildcol:
    if st.button("Rebuild index", use_container_width=True):
        docs, chunks = build_clean_corpus()
        if not chunks:
            st.error("No indexable content found. Add valid PDFs/HTML first.")
        else:
            rag = build_index(chunks)
            st.session_state["rebuild_top"] = True
            st.success(f"Index built: {len(chunks)} chunks from {len(set([c['id'].split(':::')[0] for c in chunks]))} files.")

with fetchcol:
    if st.button("Fetch 50 docs (auto-fallback)", use_container_width=True):
        stats = crawl_and_save(target_count=50)
        st.toast(f"Saved={stats['saved']}  Failed={stats['failed']}. Now click Rebuild index.")

# Corpus status
def corpus_status() -> Tuple[int,int]:
    files = [p for p in RAW_DIR.glob("**/*") if p.suffix.lower() in [".pdf",".html",".htm"]]
    pdfs = [p for p in files if p.suffix.lower()==".pdf"]
    htmls = [p for p in files if p.suffix.lower() in [".html",".htm"]]
    return len(pdfs), len(htmls)

pdf_ct, html_ct = corpus_status()
st.markdown(f"<div class='rag-card'>Corpus: <b>{pdf_ct}</b> PDF, <b>{html_ct}</b> HTML — target ≥ 50</div>", unsafe_allow_html=True)

# -----------------------------
# Load index (if exists)
# -----------------------------
rag = load_index()

# -----------------------------
# Chat box (only if index exists)
# -----------------------------
st.write("")
st.markdown("### Ask about these documents…")
query = st.text_input("Ask", placeholder="Summarize Anika Systems capabilities in 2 sentences", label_visibility="collapsed")

if st.button("Ask", type="primary"):
    if not rag:
        st.error("Index not available. Add/fetch documents and click Rebuild index to enable chat.")
    elif not query or len(query.strip()) < 3:
        st.warning("Please enter a question.")
    else:
        top_k = int(st.session_state["top_k"])
        score_thresh = float(st.session_state["score_thresh"])

        hits = retrieve(rag, query.strip(), top_k=top_k)
        if not hits:
            st.warning("No relevant passages found. Try a broader query.")
        else:
            answer = synthesize_short_answer(rag, query, hits, score_thresh)
            sources = format_sources(rag, hits)
            with st.container():
                st.markdown("<div class='rag-card'>", unsafe_allow_html=True)
                st.markdown(f"#### 🔍 {query.strip()}")
                st.markdown(answer)
                if sources:
                    st.markdown("**Sources:**")
                    for s in sources:
                        st.markdown(f"- `{s}`")
                st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("<div class='rag-footer'>Powered by Monique Bruce</div>", unsafe_allow_html=True)
