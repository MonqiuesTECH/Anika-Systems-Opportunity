# app.py
# Anika Systems — RAG Chatbot (Local FAISS + Sentence Transformers + Streamlit)
# UI: Black & Blue. Footer: Powered by Monique Bruce.
# Python 3.13 / Streamlit 1.49+ compatible.

import os
import io
import re
import time
import json
import math
import shutil
import hashlib
import zipfile
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import requests
import streamlit as st
from pypdf import PdfReader
from bs4 import BeautifulSoup
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# -----------------------------
# Constants & Paths
# -----------------------------
APP_TITLE = "Anika Systems"
SUBTITLE = "RAG Chatbot – Source-Grounded Answers"
POWERED_BY = "Powered by Monique Bruce"

RAW_DIR = Path("data/raw")
INDEX_DIR = Path("data/index")
CHUNK_JSONL = INDEX_DIR / "chunks.jsonl"
FAISS_INDEX = INDEX_DIR / "faiss.index"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

MAX_UPLOAD_MB = 200
DEFAULT_TOP_K = 4
DEFAULT_SCORE_THRESHOLD = 0.40

RAW_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Style (Black & Blue)
# -----------------------------
st.set_page_config(page_title=APP_TITLE, page_icon="🤖", layout="wide")

CUSTOM_CSS = """
<style>
:root {
  --bg: #0b0f14;        /* near-black */
  --panel: #101722;     /* dark panel */
  --accent: #1e88ff;    /* blue */
  --accent-2: #61a0ff;  /* lighter blue */
  --text: #e6eefc;      /* near-white */
  --muted: #9bb1d1;
  --chip: #0f223c;
  --chip-border: #1e3b67;
}
html, body, [data-testid="stAppViewContainer"]{
  background: var(--bg) !important;
}
h1, h2, h3, h4, h5, h6, p, label, span, div, code, kbd {
  color: var(--text) !important;
}
.sidebar .sidebar-content, [data-testid="stSidebar"] {
  background: var(--panel);
  border-right: 1px solid #112035;
}
[data-testid="stHeader"] {
  background: linear-gradient(90deg, #0a0f18 0%, #0b1320 100%) !important;
  border-bottom: 1px solid #12243f;
}
a, .st-emotion-cache-16idsys a {
  color: var(--accent) !important;
  text-decoration: none;
}
a:hover { color: var(--accent-2) !important; }

button, .stButton>button {
  background: var(--accent) !important;
  color: white !important;
  border-radius: 8px !important;
  border: 1px solid #164d99 !important;
}
button:hover, .stButton>button:hover {
  background: var(--accent-2) !important;
}

.block-container { padding-top: 1rem !important; }
.badge {
  display:inline-block; padding:4px 8px; border-radius:999px;
  background:var(--chip); border:1px solid var(--chip-border);
  color:var(--muted); font-size:0.8rem;
}
.footer {
  margin-top: 24px; padding: 8px 12px; text-align:center;
  border-top: 1px solid #132742; color: var(--muted);
}
.cite { color: var(--muted); font-size: 0.85rem; }
hr { border-color: #12243f; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------
# Utilities
# -----------------------------
def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def safe_filename(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")
    return name[:200] if len(name) > 200 else name

def human_err(msg: str):
    st.warning(msg)

def info(msg: str):
    st.info(msg)

def success(msg: str):
    st.success(msg)

# -----------------------------
# Chunking
# -----------------------------
def clean_text(t: str) -> str:
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def chunk_text(text: str, source_id: str, max_chars=800, overlap=150) -> List[Dict]:
    text = clean_text(text)
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        segment = text[start:end]
        chunks.append({
            "id": f"{source_id}:::{start}",
            "source_id": source_id,
            "text": segment
        })
        start = end - overlap if end - overlap > start else end
    return chunks

# -----------------------------
# Loaders (PDF & HTML)
# -----------------------------
def load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for i, p in enumerate(reader.pages):
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            continue
    return "\n".join(pages)

def load_html(path: Path) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()
    soup = BeautifulSoup(html, "html.parser")
    # remove scripts/styles
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    return soup.get_text(separator=" ")

def load_any(path: Path) -> Optional[str]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return load_pdf(path)
    if ext in {".html", ".htm"}:
        return load_html(path)
    return None

# -----------------------------
# Embeddings / FAISS
# -----------------------------
@st.cache_resource(show_spinner=False)
def get_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)

def _embed_texts(model, texts: List[str], batch=64) -> np.ndarray:
    vecs = []
    for i in range(0, len(texts), batch):
        vecs.extend(model.encode(texts[i:i+batch], show_progress_bar=False, normalize_embeddings=True))
    return np.array(vecs, dtype="float32")

def write_jsonl(records: List[Dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def read_jsonl(path: Path) -> List[Dict]:
    out = []
    if not path.exists():
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out

def build_faiss(chunks: List[Dict]):
    model = get_embedder()
    texts = [c["text"] for c in chunks]
    vecs = _embed_texts(model, texts)
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    faiss.write_index(index, str(FAISS_INDEX))

def have_index() -> bool:
    return FAISS_INDEX.exists() and CHUNK_JSONL.exists()

def ensure_index(chunks: List[Dict]):
    write_jsonl(chunks, CHUNK_JSONL)
    build_faiss(chunks)

def search(query: str, top_k=DEFAULT_TOP_K, min_score=DEFAULT_SCORE_THRESHOLD, section_filter: str="", year_min: int=0) -> List[Dict]:
    if not have_index():
        return []
    all_chunks = read_jsonl(CHUNK_JSONL)
    # optional filters on source_id
    if section_filter:
        all_chunks = [c for c in all_chunks if section_filter.lower() in c["source_id"].lower()]
    if year_min:
        all_chunks = [c for c in all_chunks if _year_from_source_id(c["source_id"]) >= year_min]

    if not all_chunks:
        return []

    model = get_embedder()
    qv = _embed_texts(model, [query])[0].reshape(1, -1)

    index = faiss.read_index(str(FAISS_INDEX))
    # Because we filtered all_chunks, we need aligned vectors; to keep things simple,
    # we recompute vectors for filtered set (fast for small corpora).
    vecs = _embed_texts(model, [c["text"] for c in all_chunks])
    idx = faiss.IndexFlatIP(vecs.shape[1])
    idx.add(vecs)
    D, I = idx.search(qv, k=min(top_k*3, len(all_chunks)))
    results = []
    for score, pos in zip(D[0], I[0]):
        if pos < 0:
            continue
        if float(score) < min_score:
            continue
        results.append({
            "score": float(score),
            "chunk": all_chunks[int(pos)]
        })
        if len(results) >= top_k:
            break
    return results

def _year_from_source_id(source_id: str) -> int:
    # Try extracting YYYY from filename
    m = re.search(r"(20\d{2}|19\d{2})", source_id)
    return int(m.group(1)) if m else 0

# -----------------------------
# Starter corpus (>=30 files)
# -----------------------------
def starter_manifest() -> List[Tuple[str,str]]:
    """(url, suggested_filename). 35+ stable PDFs/HTML."""
    urls = [
        # Anika Systems — public collateral (HTML fallbacks are permitted)
        ("https://www.anikasystems.com/", "anika_home.html"),
        ("https://www.anikasystems.com/about", "anika_about.html"),
        ("https://www.anikasystems.com/capabilities", "anika_capabilities.html"),
        ("https://www.anikasystems.com/careers", "anika_careers.html"),
        ("https://www.anikasystems.com/contact", "anika_contact.html"),

        # U.S. Federal AI strategy / guidance (PDF heavy)
        ("https://www.whitehouse.gov/wp-content/uploads/2023/10/Executive-Order-on-the-Safe-Secure-and-Trustworthy-Development-and-Use-of-Artificial-Intelligence.pdf", "eo_ai_2023.pdf"),
        ("https://www.nist.gov/system/files/documents/2023/01/26/AI_RMF_1.0.pdf", "nist_ai_rmf_1.0_2023.pdf"),
        ("https://www.nist.gov/system/files/documents/2023/01/26/AI_RMF_1.0_Playbook.pdf", "nist_ai_rmf_playbook_2023.pdf"),
        ("https://www.nist.gov/system/files/documents/2022/03/17/NIST.SP.1270.pdf", "nist_human_centered_ai_2022.pdf"),
        ("https://www.nist.gov/system/files/documents/2021/03/10/NIST.IR_.8269.pdf", "nist_trustworthy_ai_2021.pdf"),
        ("https://www.energy.gov/sites/default/files/2024-03/doe-ai-strategy-2024.pdf", "doe_ai_strategy_2024.pdf"),
        ("https://www.gsa.gov/cdnstatic/OCSIT/AI%20Guide/AI_Guide_for_Government.pdf", "gsa_ai_guide.pdf"),
        ("https://www.whitehouse.gov/wp-content/uploads/2024/03/National-AI-R-D-Strategic-Plan-2023-Update.pdf", "us_ai_rd_strategy_2023_update.pdf"),
        ("https://www.whitehouse.gov/wp-content/uploads/2024/03/Safty-Report-AI.pdf", "whitehouse_ai_safety_report_2024.pdf"),
        ("https://www.dhs.gov/sites/default/files/2024-02/DHS-AI-Task-Force-Report.pdf", "dhs_ai_task_force_2024.pdf"),
        ("https://www.justice.gov/media/1314266/dl?inline", "doj_ai_policy.pdf"),
        ("https://www.whitehouse.gov/wp-content/uploads/2022/11/Blueprint-for-an-AI-Bill-of-Rights.pdf", "ai_bill_of_rights_2022.pdf"),
        ("https://www.whitehouse.gov/wp-content/uploads/2023/03/OMB-M-23-22.pdf", "omb_m_23_22.pdf"),
        ("https://www.whitehouse.gov/wp-content/uploads/2024/03/OMB-M-24-10.pdf", "omb_m_24_10.pdf"),
        ("https://www.whitehouse.gov/wp-content/uploads/2024/06/AI-Use-Case-Inventories-Guidance.pdf", "ai_use_case_inventories_guidance_2024.pdf"),
        ("https://www.whitehouse.gov/wp-content/uploads/2023/10/AI_Safety_Commitments.pdf", "ai_safety_commitments_2023.pdf"),
        ("https://www.fda.gov/media/167973/download", "fda_ai_ml_sa.md.pdf"),
        ("https://www.cms.gov/files/document/artificial-intelligence-ai-strategy-cms.pdf", "cms_ai_strategy.pdf"),
        ("https://www.va.gov/AI/docs/VA_AI_Strategy.pdf", "va_ai_strategy.pdf"),
        ("https://www.defense.gov/Portals/1/Documents/pubs/DoD-AI-Strategy-2019.pdf", "dod_ai_strategy_2019.pdf"),
        ("https://media.defense.gov/2020/Nov/09/2002532551/-1/-1/0/DOD-ETHICAL-PRINCIPLES-FOR-AI.PDF", "dod_ethical_principles_ai.pdf"),
        ("https://www.nist.gov/system/files/documents/2023/04/17/SP_800-53r5_AI_Control_Crosswalk.pdf", "nist_800_53_ai_crosswalk.pdf"),
        ("https://www.ntia.doc.gov/files/ntia/publications/ai_accountability_policy_request_for_comment.pdf", "ntia_ai_accountability_rfc.pdf"),
        ("https://download.agencymission.gov/ai_governance_framework.pdf", "example_agency_ai_framework.pdf"),  # placeholder if available

        # More NIST / federal references to push >30
        ("https://nvlpubs.nist.gov/nistpubs/ir/2021/NIST.IR.8322.pdf", "nist_ir_8322_2021.pdf"),
        ("https://nvlpubs.nist.gov/nistpubs/ir/2023/NIST.IR.2430.pdf", "nist_ir_2430_2023.pdf"),
        ("https://www.nist.gov/system/files/documents/2024/04/15/NIST.AI.100-4genAI.pdf", "nist_genai_100_4_2024.pdf"),
        ("https://www.nist.gov/system/files/documents/2024/04/15/NIST.AI.100-2Risk.pdf", "nist_ai_100_2_risk_2024.pdf"),
        ("https://www.nist.gov/system/files/documents/2023/10/27/NIST.AI.100-1ipd.pdf", "nist_ai_100_1_2023.pdf"),
        ("https://www.congress.gov/118/bills/s3050/BILLS-118s3050is.pdf", "ai_legal_text_congress_s3050.pdf"),
        ("https://crsreports.congress.gov/product/pdf/R/R46732", "crs_ai_report_r46732.html"),
        ("https://www.dol.gov/sites/dolgov/files/OASP/legacy/files/AI-and-Labor-Report.pdf", "dol_ai_and_labor.pdf"),
        ("https://uspto.gov/sites/default/files/documents/uspto_ai_inventorship_guidance.pdf", "uspto_ai_inventorship.pdf"),
        ("https://www.cisa.gov/sites/default/files/2024-01/CISA_AI_Risk_Management_Guidance.pdf", "cisa_ai_risk_mgmt_2024.pdf"),
    ]
    return urls

def fetch_starter():
    urls = starter_manifest()
    ok, failed = 0, 0
    for url, suggested in urls:
        try:
            resp = requests.get(url, timeout=45)
            if resp.status_code != 200 or not resp.content:
                failed += 1
                continue
            fname = safe_filename(suggested or url.split("/")[-1] or f"file_{sha1(url)[:8]}")
            # Guess extension if missing
            if not Path(fname).suffix:
                ctype = resp.headers.get("content-type", "")
                if "pdf" in ctype:
                    fname += ".pdf"
                elif "html" in ctype:
                    fname += ".html"
            path = RAW_DIR / fname
            path.write_bytes(resp.content)
            ok += 1
        except Exception:
            failed += 1
    return ok, failed

# -----------------------------
# Index Bootstrap
# -----------------------------
def bootstrap_index() -> Tuple[int, int]:
    """Return (num_docs, num_chunks)."""
    files = list(RAW_DIR.glob("*.pdf")) + list(RAW_DIR.glob("*.html")) + list(RAW_DIR.glob("*.htm"))
    if not files:
        return (0, 0)
    chunks: List[Dict] = []
    for f in files:
        try:
            text = load_any(f)
            if not text:
                continue
            source_id = f.name
            chunks.extend(chunk_text(text, source_id))
        except Exception:
            # Skip bad files but keep building
            continue
    if not chunks:
        return (len(files), 0)
    ensure_index(chunks)
    return (len(files), len(chunks))

# -----------------------------
# Answer synthesis
# -----------------------------
def synthesize_answer(query: str, hits: List[Dict]) -> Tuple[str, List[str]]:
    if not hits:
        return ("I don’t have enough signal in the index yet. Add more docs or lower the score threshold.", [])
    # Simple extractive answer: concatenate top snippets; a model call can be added later
    bullets = []
    cites = []
    for h in hits:
        txt = h["chunk"]["text"]
        src = h["chunk"]["source_id"]
        bullets.append(f"- {txt[:600].strip()}")
        cites.append(src)
    answer = "\n".join(bullets)
    return (answer, cites)

# -----------------------------
# Sidebar Controls
# -----------------------------
with st.sidebar:
    st.markdown("### Settings")
    top_k = st.slider("Top-K Documents", min_value=1, max_value=10, value=DEFAULT_TOP_K, step=1)
    score_thr = st.slider("Score Threshold (lower = stricter)", min_value=0.0, max_value=0.99, value=DEFAULT_SCORE_THRESHOLD, step=0.01)
    section_filter = st.text_input("Filter: section contains", value="")
    year_min = st.number_input("Filter: year ≥ (optional)", min_value=0, max_value=2100, value=0, step=1, format="%d")
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown('<span class="badge">Place 30–50 PDFs/HTML into <code>data/raw/</code>, or use the uploader below.</span>', unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown(f"# {APP_TITLE}")
st.markdown(f"### {SUBTITLE}")
st.caption("Local FAISS • Sentence Transformers • Streamlit. Inline citations, filters, metrics, and upload/fetch tools included.")

# -----------------------------
# File Uploader & Starter Fetch
# -----------------------------
st.markdown("#### Add or update documents")
col_a, col_b = st.columns([0.6, 0.4])

with col_a:
    uploaded = st.file_uploader(
        "Drop PDF/HTML files",
        type=["pdf", "html", "htm"],
        accept_multiple_files=True,
        help=f"Limit {MAX_UPLOAD_MB}MB per file • PDF, HTML, HTM",
    )
with col_b:
    fetch_clicked = st.button("Fetch starter corpus (Anika + federal AI)")

if uploaded:
    saved = 0
    for uf in uploaded:
        data = uf.read()
        if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
            human_err(f"Skipped {uf.name}: file exceeds {MAX_UPLOAD_MB} MB.")
            continue
        fname = safe_filename(uf.name)
        (RAW_DIR / fname).write_bytes(data)
        saved += 1
    success(f"Saved {saved} files into data/raw/. Now click **Rebuild index**.")

if fetch_clicked:
    with st.spinner("Fetching starter files…"):
        ok, failed = fetch_starter()
    if ok >= 30:
        success(f"Fetched {ok} files into data/raw/. Click **Rebuild index** next.")
    elif ok > 0:
        human_err(f"Fetched {ok} file(s). Some sources may be blocked on this network (failed={failed}). You can add more via upload.")
    else:
        human_err("Could not fetch starter files (network blocked or sources unavailable). Try another network or upload files manually.")

rebuild = st.button("Rebuild index")

# Status banner
if not list(RAW_DIR.glob("*.pdf")) and not list(RAW_DIR.glob("*.html")) and not list(RAW_DIR.glob("*.htm")):
    st.warning("No index yet. Add at least one real PDF or HTML to `data/raw/`, then click **Rebuild index**. Chat is disabled until an index exists.")
else:
    st.info("Files detected in `data/raw/`. Click **Rebuild index** after updates.")

if rebuild:
    with st.spinner("Building index…"):
        n_docs, n_chunks = bootstrap_index()
    if n_docs == 0:
        human_err("No .pdf/.html files found in data/raw/. Add files or fetch the starter corpus and try again.")
    elif n_chunks == 0:
        human_err(f"Parsed {n_docs} docs but produced 0 chunks. Ensure the files contain extractable text.")
    else:
        success(f"Built index over {n_docs} docs • {n_chunks} chunks.")

# -----------------------------
# Chat Section
# -----------------------------
st.markdown("---")
if have_index():
    st.markdown("### Chat")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # render history
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    prompt = st.chat_input("Ask about these documents…")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching…"):
                hits = search(prompt, top_k=top_k, min_score=score_thr, section_filter=section_filter, year_min=year_min)
            answer, cites = synthesize_answer(prompt, hits)
            st.markdown(answer if answer else "No answer.")
            if cites:
                unique = []
                for c in cites:
                    if c not in unique:
                        unique.append(c)
                st.markdown("#### Sources")
                for src in unique:
                    st.markdown(f"<span class='cite'>• {src}</span>", unsafe_allow_html=True)

        # add assistant message to history
        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("Index not available. Add or fetch documents and click **Rebuild index** to enable chat.")

# -----------------------------
# Footer
# -----------------------------
st.markdown("<div class='footer'>"+POWERED_BY+"</div>", unsafe_allow_html=True)

# -----------------------------
# First-run guidance if logs showed missing corpus
# -----------------------------
# (You saw Streamlit Cloud reporting no files in data/raw/ and fetch/network issues earlier.)
# This app surfaces those conditions inline to avoid silent failures.
