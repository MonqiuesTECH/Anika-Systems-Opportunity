# app.py — Anika Systems RAG Chatbot
# Black/blue branding • Auto-ingest all PDFs/HTML • Enforce ≥50 docs • Concise answers

import re, json, time
from pathlib import Path
from typing import List, Dict, Any, Tuple

import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from bs4 import BeautifulSoup
from pypdf import PdfReader

# ----------------------
# Constants / Paths
# ----------------------
APP_TITLE = "Anika Systems"
POWERED_BY = "Powered by Monique Bruce"
RAW_DIR = Path("data/raw")
INDEX_DIR = Path("data/index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
MIN_DOCS = 50  # requirement check

# ----------------------
# CSS Theme (black + blue)
# ----------------------
st.set_page_config(page_title=APP_TITLE, page_icon="🤖", layout="wide")
st.markdown(f"""
<style>
:root {{
  --bg: #0b1220;
  --panel: #111827;
  --text: #dbeafe;
  --muted: #9ca3af;
  --accent: #2563eb;
}}
.stApp {{
  background: var(--bg);
  color: var(--text);
}}
[data-testid="stSidebar"] {{
  background: #0a1220;
  color: var(--text);
  border-right: 1px solid #1f2937;
}}
h1,h2,h3,h4,h5,h6,p,span,label {{
  color: var(--text) !important;
}}
.stButton>button {{
  background: var(--accent) !important;
  color: white !important;
  border: none !important;
  border-radius: 8px !important;
}}
.stSlider label, .stNumberInput label {{
  color: var(--text) !important;
}}
.stTextInput input, .stTextArea textarea {{
  background: var(--panel) !important;
  color: var(--text) !important;
  border: 1px solid #1f2937 !important;
  border-radius: 6px;
}}
.chat-bubble {{
  background: var(--panel);
  border: 1px solid #1f2937;
  border-radius: 12px;
  padding: 0.8rem;
  margin-top: 0.5rem;
  color: var(--text);
}}
.source-badge {{
  background: #1e3a8a;
  color: #bfdbfe;
  padding: 2px 6px;
  border-radius: 8px;
  font-size: 12px;
  margin-right: 4px;
}}
.footer {{
  color: var(--muted);
  text-align: center;
  font-size: 12px;
  margin-top: 14px;
}}
</style>
""", unsafe_allow_html=True)

# ----------------------
# Utils
# ----------------------
def load_model():
    if "model" not in st.session_state:
        st.session_state.model = SentenceTransformer(MODEL_NAME)
    return st.session_state.model

def read_pdf(path: Path) -> str:
    try:
        pdf = PdfReader(str(path))
        out = []
        for page in pdf.pages:
            out.append(page.extract_text() or "")
        return " ".join(out)
    except Exception:
        return ""

def read_html(path: Path) -> str:
    try:
        soup = BeautifulSoup(path.read_text(errors="ignore"), "lxml")
        for tag in soup(["script","style","noscript","nav","footer","header"]):
            tag.decompose()
        return soup.get_text(" ", strip=True)
    except Exception:
        return ""

def chunk_text(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    chunks, i = [], 0
    while i < len(text):
        chunks.append(text[i:i+CHUNK_SIZE])
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

# ----------------------
# Index build / load
# ----------------------
def build_index() -> Tuple[int,int]:
    texts, meta = [], []
    files = [p for p in RAW_DIR.glob("**/*") if p.suffix.lower() in [".pdf",".html",".htm"]]
    for p in files:
        txt = read_pdf(p) if p.suffix.lower()==".pdf" else read_html(p)
        if not txt or len(txt)<200: 
            continue
        chunks = chunk_text(txt)
        for j,c in enumerate(chunks):
            texts.append(c)
            meta.append({"source": p.name, "chunk": j})
    if not texts:
        return 0,0
    model = load_model()
    embs = model.encode(texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embs.shape[1])
    faiss.normalize_L2(embs)
    index.add(embs.astype(np.float32))
    faiss.write_index(index, str(INDEX_DIR/"faiss.index"))
    (INDEX_DIR/"meta.json").write_text(json.dumps(meta))
    st.session_state.index = index
    st.session_state.meta = meta
    st.session_state.texts = texts
    return len(files), len(texts)

def ensure_index():
    if "index" in st.session_state:
        return True
    fidx, fmeta = INDEX_DIR/"faiss.index", INDEX_DIR/"meta.json"
    if fidx.exists() and fmeta.exists():
        index = faiss.read_index(str(fidx))
        meta = json.loads(fmeta.read_text())
        st.session_state.index = index
        st.session_state.meta = meta
        st.session_state.texts = []  # not reloaded (lightweight)
        return True
    return False

# ----------------------
# Search / Answer
# ----------------------
def search(query: str, k:int=4, threshold:float=0.4) -> List[Dict[str,Any]]:
    if not ensure_index(): return []
    model = load_model()
    q = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    D,I = st.session_state.index.search(q.astype(np.float32), k)
    out=[]
    for score,idx in zip(D[0],I[0]):
        if idx==-1 or score<threshold: continue
        meta = st.session_state.meta[idx]
        text = st.session_state.texts[idx] if st.session_state.texts else ""
        out.append({"score":float(score),"meta":meta,"text":text})
    return out

def concise_answer(q: str, hits: List[Dict[str,Any]]) -> str:
    if not hits: return "I couldn’t find a grounded answer."
    joined = " ".join(h["text"] for h in hits[:2])
    sents = re.split(r"(?<=[.!?])\s+", joined)
    out = " ".join(sents[:3])[:400]
    return out.strip()

# ----------------------
# UI
# ----------------------
st.markdown(f"# {APP_TITLE}")
st.caption("RAG Chatbot — Source-Grounded Answers (local FAISS + MiniLM)")

k = st.sidebar.slider("Top-K Documents",1,10,4)
threshold = st.sidebar.slider("Score Threshold",0.0,1.0,0.4,0.01)

# Corpus check
files = [p for p in RAW_DIR.glob("**/*") if p.suffix.lower() in [".pdf",".html",".htm"]]
st.info(f"Corpus: {len(files)} file(s). Target ≥ {MIN_DOCS}")
if len(files)<MIN_DOCS:
    st.warning(f"⚠️ Only {len(files)} docs found. Please add more until you have at least {MIN_DOCS}.")

if st.button("Rebuild index"):
    n_files, n_chunks = build_index()
    if n_files==0:
        st.error("No valid text found. Add more docs.")
    else:
        st.success(f"Index built: {n_chunks} chunks from {n_files} files.")

q = st.text_input("Ask about these documents:", placeholder="E.g., Summarize Anika Systems capabilities")
if st.button("Ask"):
    hits = search(q,k,threshold)
    if not hits:
        st.warning("No answer above threshold.")
    else:
        ans = concise_answer(q,hits)
        st.markdown(f"<div class='chat-bubble'>{ans}</div>", unsafe_allow_html=True)
        st.markdown("**Sources:**")
        for h in hits:
            st.markdown(f"<span class='source-badge'>{h['meta']['source']}</span>", unsafe_allow_html=True)

st.markdown(f"<div class='footer'>{POWERED_BY}</div>", unsafe_allow_html=True)
