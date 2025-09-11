# app.py — Anika Systems (local FAISS + MiniLM)
# Streamlit 1.49 / Python 3.13 compatible

from __future__ import annotations

import os
import time
import random
from pathlib import Path
from typing import List, Tuple

import streamlit as st

# Local modules
from src.loaders import load_raw_files, save_url_as_html, guess_filename
from src.embed_index import ensure_faiss_index, wipe_index, index_dir, chunks_dir
from src.retrieve import retrieve
from src.clean_chunk import clean_and_chunk
from src.prompts import answer_prompt
from src.metrics import highlight_spans

# ----------------------------
# App constants / theming
# ----------------------------
APP_TITLE = "Anika Systems"
RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

MAX_FETCH = 50
TOP_K_DEFAULT = 4
SCORE_THRESHOLD_DEFAULT = 0.40

DARK_BG = "#0f172a"   # slate-900
DARK_PANEL = "#111827" # gray-900
PRIMARY = "#2563eb"   # blue-600
TEXT = "#e5e7eb"      # gray-200
SUBTLE = "#9ca3af"    # gray-400

# ----------------------------
# Utilities
# ----------------------------
def css():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: radial-gradient(1200px 600px at 20% 10%, #0b1222, {DARK_BG}) !important;
            color: {TEXT};
        }}
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #0b1222 0%, {DARK_PANEL} 100%);
        }}
        .stButton>button {{
            background: {PRIMARY} !important; color: white !important; border: none;
        }}
        .stTextInput input, .stTextArea textarea {{
            background: #0b1222; color: {TEXT}; border: 1px solid #1f2937;
        }}
        .stChatInput input {{
            background: #0b1222 !important; color: {TEXT} !important; border: 1px solid #1f2937 !important;
        }}
        .chunk-badge {{ color: {SUBTLE}; font-size: 0.8rem; }}
        .footnote {{ color: {SUBTLE}; font-size: 0.8rem; text-align:center; padding-top:6px; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

def ensure_state():
    defaults = dict(
        top_k=TOP_K_DEFAULT,
        score_thresh=SCORE_THRESHOLD_DEFAULT,
        contains_filter="",
        year_filter=0,
        fetched=0,
        fetched_failed=0,
        fetched_synth=0,
        index_ready=False,
        rebuild_top=False,  # flag to trigger a rebuild at top of script
        chat_history=[],
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# ----------------------------
# Robust 50-doc fetcher
# ----------------------------
SEED_URLS = [
    # Company & gov/AI adjacent—safe to fetch anywhere as agreed
    "https://anika-systems.com/",
    "https://anika-systems.com/who-we-are/",
    "https://anika-systems.com/what-we-do/",
    "https://anika-systems.com/careers/",
    "https://anika-systems.com/careers/#open-positions",
    "https://anika-systems.com/contact/",
    # Federal AI / policy primers (HTML pages are fine)
    "https://www.whitehouse.gov/ostp/ai-bill-of-rights/",
    "https://www.nist.gov/itl/ai-risk-management-framework",
    "https://www.ai.gov/strategic-pillars/",
    "https://www.usds.gov/impact",
    # Cloud partner overviews
    "https://azure.microsoft.com/en-us/solutions/ai/",
    "https://aws.amazon.com/machine-learning/",
    "https://cloud.google.com/ai",
    "https://www.databricks.com/solutions/machine-learning",
    "https://www.redhat.com/en/topics/ai",
]

def fetch_50_docs(target: int = MAX_FETCH) -> Tuple[int, int, int]:
    """
    Tries to save at least `target` HTML/PDF docs into data/raw/.
    Returns (saved_ok, synthesized, failed).
    """
    saved, synthesized, failed = 0, 0, 0
    seen = set()

    # 1) Start with our seed list
    candidates = list(dict.fromkeys(SEED_URLS))  # stable dedupe, keep order

    # 2) Expand: if we still need more, create simple “about + ai + gov” queries from seeds
    expands = [
        "https://aws.amazon.com/what-is/fedramp/",
        "https://learn.microsoft.com/en-us/azure/compliance/offerings/offering-fedramp",
        "https://cloud.google.com/security/compliance/fedramp",
        "https://www.cisa.gov/secure-by-design",
        "https://www.gsa.gov/technology/technology-products-services/ai-center-of-excellence",
    ]
    candidates.extend([u for u in expands if u not in candidates])

    # 3) If still < target, pad with public PDFs that usually succeed
    pdf_pad = [
        "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1e2024.pdf",
        "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.ipd.pdf",
        "https://csrc.nist.gov/csrc/media/Publications/white-paper/2023/01/26/trustworthy-and-responsible-ai/documents/trustworthy-responsible-ai.pdf",
        "https://download.microsoft.com/download/f/6/7/f6710c15-1010-4f0c-8f63-0a35f9cd6c7a/microsoft-ai-principles.pdf",
        "https://ai.google/static/documents/principles/google_ai_principles.pdf",
        "https://pages.awscloud.com/rs/112-TZM-766/images/AI-ML-in-the-Public-Sector.pdf",
        "https://www.redhat.com/cms/managed-files/cl-cloud-services-public-sector-security-ebook-f28368-202105-en.pdf",
    ]
    candidates.extend([u for u in pdf_pad if u not in candidates])

    # Try until we hit target or run out
    for url in candidates:
        if saved >= target:
            break
        try:
            fname = guess_filename(url)
            out = RAW_DIR / fname
            if out.exists():
                continue
            ok = save_url_as_html(url, out)  # handles pdf/html; returns True/False
            if ok:
                saved += 1
            else:
                failed += 1
        except Exception:
            failed += 1

    return saved, synthesized, failed

# ----------------------------
# Index pipeline
# ----------------------------
def bootstrap_index():
    """
    Build chunks + FAISS index from any PDFs/HTML in data/raw/.
    """
    # Clean old chunk dir, keep index dir but re-create inside ensure_faiss_index
    wipe_index()  # clears old artifacts safely

    # Load raw docs
    docs = load_raw_files(RAW_DIR)
    if not docs:
        return 0, 0

    # Clean & chunk
    total_chunks = 0
    for doc in docs:
        chunks = clean_and_chunk(doc)
        total_chunks += len(chunks)

    # Embed + build FAISS
    ensure_faiss_index(chunks_dir(), index_dir())
    st.session_state["index_ready"] = True
    return len(docs), total_chunks

# ----------------------------
# Chat answering
# ----------------------------
def answer_question(q: str, k: int, threshold: float, contains: str, min_year: int):
    """
    Retrieve -> short, source-grounded answer.
    """
    results = retrieve(
        query=q,
        k=k,
        score_threshold=threshold,
        section_contains=contains.strip() or None,
        year_min=int(min_year) if min_year else None,
        index_dir=index_dir(),
    )

    if not results:
        return "I couldn't find a grounded answer in the corpus.", []

    # Build a short, safe answer
    answer = answer_prompt(
        question=q,
        passages=[r.text for r in results],
        max_words=70,                   # short and crisp
        forbid_speculation=True,        # no nonsense
        style="concise factual bullets",# consistent tone
    )
    return answer, results

# ----------------------------
# Callbacks (avoid state writes mid-render)
# ----------------------------
def on_fetch_click():
    saved, synth, failed = fetch_50_docs(MAX_FETCH)
    st.session_state["fetched"] = saved
    st.session_state["fetched_failed"] = failed
    st.session_state["fetched_synth"] = synth
    st.toast(f"Saved={saved} Failed={failed}. Now click Rebuild index.")

def on_rebuild_click():
    st.session_state["rebuild_top"] = True
    # Streamlit 1.49+: use st.rerun(), not experimental_rerun
    st.rerun()

# ----------------------------
# Main UI
# ----------------------------
def sidebar():
    with st.sidebar:
        st.header("Settings")
        st.session_state["top_k"] = st.slider("Top-K Documents", 1, 10, st.session_state["top_k"])
        st.session_state["score_thresh"] = st.slider("Score Threshold (lower = stricter)", 0.0, 1.0, float(st.session_state["score_thresh"]), 0.01)
        st.session_state["contains_filter"] = st.text_input("Filter: section contains", st.session_state["contains_filter"])
        st.session_state["year_filter"] = st.number_input("Filter: year ≥ (optional)", min_value=0, max_value=2100, value=int(st.session_state["year_filter"]))
        st.caption("Be concise; answers cite sources.")

def header():
    st.markdown(f"<h1 style='color:{TEXT};'>{APP_TITLE}</h1>", unsafe_allow_html=True)
    st.caption("RAG Chatbot — Source-Grounded Answers (local FAISS + MiniLM)")

def corpus_box():
    st.subheader("Add or update documents")
    st.file_uploader(
        "Drag and drop files here",
        type=["pdf", "html", "htm"],
        accept_multiple_files=True,
        key="uploader",
        help="Limit 200MB per file • PDF, HTML, HTM",
    )

    c1, c2 = st.columns([1,1])
    with c1:
        st.button("Rebuild index", on_click=on_rebuild_click, use_container_width=True)
    with c2:
        st.button("Fetch 50 docs (auto-fallback)", on_click=on_fetch_click, use_container_width=True)

    # Corpus status line
    pdfs = len(list(RAW_DIR.glob("*.pdf")))
    htmls = len(list(RAW_DIR.glob("*.html"))) + len(list(RAW_DIR.glob("*.htm")))
    st.info(f"Corpus: {pdfs} PDF, {htmls} HTML — target ≥ {MAX_FETCH}")

def chat_box():
    st.subheader("Ask about these documents…")
    q = st.text_input("Ask", placeholder="Summarize Anika Systems capabilities in 2 sentences", label_visibility="collapsed")
    if st.button("Ask"):
        if not st.session_state.get("index_ready", False):
            st.warning("Index not available. Fetch/upload documents and click **Rebuild index** first.")
            return
        with st.spinner("Retrieving…"):
            answer, results = answer_question(
                q=q,
                k=int(st.session_state["top_k"]),
                threshold=float(st.session_state["score_thresh"]),
                contains=st.session_state["contains_filter"],
                min_year=int(st.session_state["year_filter"]),
            )
        with st.container(border=True):
            st.markdown(answer)
            st.markdown("**Sources:**")
            for r in results:
                st.markdown(f"- {r.meta.get('source_name','?')}  <span class='chunk-badge'>score: {r.score:.2f}</span>", unsafe_allow_html=True)

def footer():
    st.markdown("<div class='footnote'>Powered by Monique Bruce</div>", unsafe_allow_html=True)

def handle_top_rebuild():
    """
    If a rebuild was requested in a previous run, do it at the very top of the script,
    then reset the flag and rerun again to render the fresh UI.
    """
    if st.session_state.get("rebuild_top", False):
        with st.spinner("Building index…"):
            n_docs, n_chunks = bootstrap_index()
        st.success(f"Index built: {n_chunks} chunks from {n_docs} files.")
        st.session_state["rebuild_top"] = False
        st.session_state["index_ready"] = True
        st.rerun()

# ----------------------------
# Streamlit App
# ----------------------------
def main():
    st.set_page_config(page_title=f"{APP_TITLE}", page_icon="🤖", layout="wide")
    css()
    ensure_state()
    handle_top_rebuild()         # <-- do any rebuild immediately at the top

    sidebar()
    header()
    corpus_box()
    chat_box()
    footer()

if __name__ == "__main__":
    main()

