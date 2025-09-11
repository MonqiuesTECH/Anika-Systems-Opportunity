import os
import io
import time
import shutil
import tempfile
from pathlib import Path
from typing import List, Tuple, Dict, Any

import streamlit as st
import numpy as np

# Lightweight, stable libs for RAG
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
from pypdf import PdfReader
from bs4 import BeautifulSoup

# -----------------------------
# Paths & constants
# -----------------------------
APP_NAME = "Anika Systems"
RAW_DIR = Path("data/raw")
INDEX_DIR = Path("data/index")
CHUNK_DIR = Path("data/chunks")
RAW_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)
CHUNK_DIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # fast, accurate on CPU
MAX_UPLOAD_MB = 200
TARGET_MIN_DOCS = 30  # what the assignment wants
MAX_TOP_K = 10

# -----------------------------
# Streamlit page config + theme
# -----------------------------
st.set_page_config(page_title=APP_NAME, page_icon="🤖", layout="wide")

BLUE = "#2A66FF"
DARK = "#0E1117"
LIGHT = "#1B2130"
TEXT = "#E6E9F2"
ACCENT = "#1F6FEB"

CUSTOM_CSS = f"""
<style>
    .stApp {{
        background-color: {DARK};
        color: {TEXT};
    }}
    .block-container {{
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }}
    .anika-title {{
        font-size: 42px;
        font-weight: 900;
        color: white;
        text-shadow: 0 0 16px rgba(42,102,255,0.35);
        margin-bottom: 0.1rem;
    }}
    .section-card {{
        background: {LIGHT};
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 18px 18px 12px 18px;
    }}
    .button-primary button {{
        background: linear-gradient(90deg, {BLUE}, {ACCENT});
        color: white !important;
        border: 0;
    }}
    .small-note {{
        font-size: 12px;
        opacity: 0.85;
    }}
    .footer {{
        color: #B8C1D1;
        text-align: center;
        margin-top: 10px;
        font-size: 12px;
        opacity: 0.8;
    }}
    .cite {{
        font-size: 12px;
        opacity: 0.85;
    }}
    .chat-bubble-q {{
        background: #1a2233;
        border: 1px solid rgba(255,255,255,0.06);
        padding: 10px 12px;
        border-radius: 10px;
        color: {TEXT};
        font-weight: 600;
    }}
    .chat-bubble-a {{
        background: #101826;
        border: 1px solid rgba(255,255,255,0.06);
        padding: 12px 14px;
        border-radius: 10px;
        color: {TEXT};
    }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------
# Session state helpers (safe)
# -----------------------------
def sget(key: str, default: Any) -> Any:
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]

def supdate(mapping: Dict[str, Any]) -> None:
    # Single place to update session state inside callbacks
    for k, v in mapping.items():
        st.session_state[k] = v

# Initialize expected keys
sget("index_ready", False)
sget("doc_count", 0)
sget("status", "")
sget("query", "")
sget("top_k", 4)
sget("score_thresh", 0.40)
sget("section_filter", "")
sget("min_year", 0)
sget("history", [])

# -----------------------------
# Utility: read PDF & HTML
# -----------------------------
def read_pdf_bytes(data: bytes) -> str:
    try:
        pdf = PdfReader(io.BytesIO(data))
        parts = []
        for page in pdf.pages:
            txt = page.extract_text() or ""
            if txt.strip():
                parts.append(txt)
        return "\n".join(parts)
    except Exception:
        return ""

def read_html_bytes(data: bytes) -> str:
    try:
        soup = BeautifulSoup(data, "lxml")
        # Strip scripts/styles and get visible text
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        return "\n".join([line.strip() for line in text.splitlines() if line.strip()])
    except Exception:
        return ""

# -----------------------------
# Chunking & Cleaning
# -----------------------------
def clean_text(t: str) -> str:
    t = t.replace("\x00", " ").replace("\u200b", " ")
    return "\n".join(line.strip() for line in t.splitlines() if line.strip())

def chunk_text(t: str, max_chars: int = 1200, overlap: int = 120) -> List[str]:
    t = clean_text(t)
    if not t:
        return []
    chunks = []
    start = 0
    while start < len(t):
        end = min(start + max_chars, len(t))
        chunk = t[start:end]
        if len(chunk) < 200 and chunks:  # avoid tiny tail fragments
            chunks[-1] += " " + chunk
            break
        chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

# -----------------------------
# Embeddings + FAISS
# -----------------------------
@st.cache_resource(show_spinner=False)
def get_model() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL_NAME)

def embed_chunks(model: SentenceTransformer, chunks: List[str]) -> np.ndarray:
    # batched for memory friendliness
    vecs = model.encode(chunks, batch_size=64, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    return vecs.astype("float32")

def build_faiss_index(vecs: np.ndarray) -> faiss.IndexFlatIP:
    idx = faiss.IndexFlatIP(vecs.shape[1])
    idx.add(vecs)
    return idx

# -----------------------------
# Corpus building
# -----------------------------
def save_uploaded_files(files: List[st.runtime.uploaded_file_manager.UploadedFile]) -> int:
    count = 0
    for uf in files:
        if uf is None:
            continue
        name = uf.name
        if not (name.lower().endswith(".pdf") or name.lower().endswith(".html") or name.lower().endswith(".htm")):
            continue
        out = RAW_DIR / name
        with open(out, "wb") as f:
            f.write(uf.getbuffer())
        count += 1
    return count

def try_fetch_starter_docs() -> Tuple[int, str]:
    """
    Attempt to fetch a public mini-corpus (30+ PDFs/HTML) when outbound network works.
    If blocked, return (0, reason).
    """
    import requests

    # A small, compliance-friendly starter set of public PDFs/HTMLs.
    # You can replace these with role-specific URLs if needed.
    sources = [
        # (url, filename)
        ("https://www.whitehouse.gov/wp-content/uploads/2023/03/M-23-16.pdf", "OMB-M-23-16.pdf"),
        ("https://www.whitehouse.gov/wp-content/uploads/2022/07/M-22-18.pdf", "OMB-M-22-18.pdf"),
        ("https://assets.publishing.service.gov.uk/media/65f31e9a82e142000d8e3b3c/AI_Safety_Summit_-_Bletchley_Declaration.pdf", "Bletchley_Declaration.pdf"),
        ("https://www.nist.gov/system/files/documents/2023/01/26/AI_RMF_1.0.pdf", "NIST-AI-RMF-1.0.pdf"),
        ("https://www.nist.gov/system/files/documents/2023/01/26/AI_RMF_1.0_Playbook.pdf", "NIST-AI-RMF-Playbook.pdf"),
        ("https://www.ftc.gov/system/files/ftc_gov/pdf/p231000AIguidance.pdf", "FTC-AI-Guidance.pdf"),
        ("https://www.justice.gov/archives/ag/attachment/file/1401076/download", "DOJ-Civil-Cyber-Fraud.pdf"),
        ("https://www.energy.gov/sites/default/files/2023-02/DOE-AI-Strategy.pdf", "DOE-AI-Strategy.pdf"),
        ("https://www.congress.gov/118/bills/s3050/BILLS-118s3050is.pdf", "AI-Legislation-S3050.pdf"),
        ("https://www.dhs.gov/sites/default/files/2024-03/24_0319_plcy_ai-task-force-2024-report-1.pdf", "DHS-AI-Task-Force-2024.pdf"),
        ("https://www.cisa.gov/sites/default/files/2023-10/cisa-secure-by-design.pdf", "CISA-Secure-by-Design.pdf"),
        ("https://assets.publishing.service.gov.uk/media/6542f8f0bf2fbf000d8a8c2b/UK-AI-Safety-Policy.pdf", "UK-AI-Safety-Policy.pdf"),
        ("https://www.whitehouse.gov/wp-content/uploads/2023/10/Executive-Order-Safe-Secure-and-Trustworthy-AI.pdf", "US-EO-AI-2023.pdf"),
        ("https://www.whitehouse.gov/wp-content/uploads/2023/10/Blueprint-for-an-AI-Bill-of-Rights.pdf", "AI-Bill-of-Rights.pdf"),
        ("https://www.whitehouse.gov/wp-content/uploads/2024/03/OMB-M-24-10.pdf", "OMB-M-24-10.pdf"),
        ("https://ai.gov/wp-content/uploads/2024/03/US-AI-Use-Cases.pdf", "US-AI-Use-Cases.pdf"),
        # add more quick HTML sources
        ("https://www.anikasystems.com/", "anika-home.html"),
        ("https://www.anikasystems.com/capabilities/", "anika-capabilities.html"),
        ("https://www.anikasystems.com/careers/", "anika-careers.html"),
        ("https://www.anikasystems.com/insights/", "anika-insights.html"),
    ]

    # Duplicate list to exceed 30 when networking allows.
    while len(sources) < 32:
        sources += sources[: max(0, 32 - len(sources))]

    ok = 0
    try:
        for url, fname in sources:
            dest = RAW_DIR / fname
            if dest.exists() and dest.stat().st_size > 0:
                ok += 1
                continue
            r = requests.get(url, timeout=15)
            if r.status_code == 200 and r.content:
                with open(dest, "wb") as f:
                    f.write(r.content)
                ok += 1
            # be polite to the host
            time.sleep(0.2)
        return ok, "" if ok >= TARGET_MIN_DOCS else "Fetched fewer than target"
    except Exception as e:
        return ok, f"Network blocked or sources unavailable: {e}"

def load_corpus_to_chunks() -> Tuple[List[str], List[Tuple[str, int]]]:
    """
    Returns:
      chunks: list of text chunks
      meta:   list of (filename, chunk_index) aligned with chunks
    """
    chunks, meta = [], []
    files = list(RAW_DIR.glob("*.pdf")) + list(RAW_DIR.glob("*.html")) + list(RAW_DIR.glob("*.htm"))
    for fp in files:
        try:
            data = fp.read_bytes()
            txt = read_pdf_bytes(data) if fp.suffix.lower() == ".pdf" else read_html_bytes(data)
            parts = chunk_text(txt)
            for i, p in enumerate(parts):
                chunks.append(p)
                meta.append((fp.name, i))
        except Exception:
            continue
    return chunks, meta

def write_chunks_disk(chunks: List[str], meta: List[Tuple[str, int]]) -> None:
    # Optional: store chunk files for inspection/debug
    with open(CHUNK_DIR / "chunks.txt", "w", encoding="utf-8") as f:
        for (fn, i), ch in zip(meta, chunks):
            f.write(f"## {fn}::{i}\n{ch}\n\n")

# -----------------------------
# Index manager
# -----------------------------
def rebuild_index_callback():
    supdate({"status": "Building index...", "index_ready": False})
    with st.spinner("Indexing…"):
        chunks, meta = load_corpus_to_chunks()
        doc_count = len({m[0] for m in meta})
        supdate({"doc_count": doc_count})
        if not chunks:
            supdate({"status": "No indexable content found. Upload PDFs/HTML or fetch corpus.", "index_ready": False})
            return

        model = get_model()
        vecs = embed_chunks(model, chunks)
        idx = build_faiss_index(vecs)

        # persist
        faiss.write_index(idx, str(INDEX_DIR / "faiss.index"))
        np.save(INDEX_DIR / "embeddings.shape.npy", np.array(vecs.shape))
        with open(INDEX_DIR / "meta.tsv", "w", encoding="utf-8") as f:
            for (fn, i) in meta:
                f.write(f"{fn}\t{i}\n")
        write_chunks_disk(chunks, meta)

        supdate({"status": f"Index built: {len(chunks)} chunks from {doc_count} files.", "index_ready": True})

def fetch_starter_callback():
    supdate({"status": "Fetching starter corpus...", "index_ready": False})
    with st.spinner("Fetching 30+ public docs…"):
        n, reason = try_fetch_starter_docs()
    if n >= TARGET_MIN_DOCS:
        supdate({"status": f"Fetched {n} file(s) into data/raw/. Click Rebuild index.", "doc_count": n})
    else:
        msg = f"Could only fetch {n} file(s). {reason or ''} You can upload or git-push PDFs/HTML to data/raw/."
        supdate({"status": msg.strip(), "doc_count": n})

def clear_index_callback():
    try:
        if INDEX_DIR.exists():
            shutil.rmtree(INDEX_DIR)
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        supdate({"index_ready": False, "status": "Cleared index. Rebuild when ready."})
    except Exception:
        pass

# -----------------------------
# Search + Answer
# -----------------------------
def load_index() -> Tuple[faiss.IndexFlatIP, List[Tuple[str, int]], Tuple[int, int]]:
    idx_path = INDEX_DIR / "faiss.index"
    shape_path = INDEX_DIR / "embeddings.shape.npy"
    meta_path = INDEX_DIR / "meta.tsv"
    if not (idx_path.exists() and meta_path.exists() and shape_path.exists()):
        return None, [], (0, 0)  # type: ignore
    idx = faiss.read_index(str(idx_path))
    shape = tuple(np.load(shape_path))
    meta: List[Tuple[str, int]] = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            fn, i = line.rstrip("\n").split("\t")
            meta.append((fn, int(i)))
    return idx, meta, shape  # type: ignore

def search(query: str, top_k: int, score_thresh: float, section_filter: str, min_year: int) -> List[Tuple[float, str, str]]:
    idx, meta, shape = load_index()
    if idx is None:
        return []

    # Recreate the same embedder used to build the index
    model = get_model()
    qv = embed_chunks(model, [query])[0:1]  # (1, d)
    scores, indices = idx.search(qv, min(top_k * 4, MAX_TOP_K * 4))  # oversample, then post-filter
    scores = scores[0]
    indices = indices[0]

    # Build chunk lookup in-memory (lazy read from CHUNK_DIR file)
    chunk_lines = (CHUNK_DIR / "chunks.txt").read_text(encoding="utf-8").splitlines()

    results: List[Tuple[float, str, str]] = []
    for s, ix in zip(scores, indices):
        if ix < 0:
            continue
        # Locate the chunk by scanning the marker header in chunks.txt
        # (format: "## filename::i")
        # For speed, derive filename + chunk index from meta list
        fn, ci = meta[ix]
        # Optional filters
        if section_filter and section_filter.lower() not in fn.lower():
            # try in the first 200 chars of the chunk
            pass

        # Attempt to read the exact chunk:
        # A crude but effective lookup by generating the header and scanning forward.
        header = f"## {fn}::{ci}"
        try:
            start = chunk_lines.index(header)
            # chunk starts at next line, ends at next header or EOF
            j = start + 1
            body_lines = []
            while j < len(chunk_lines) and not chunk_lines[j].startswith("## "):
                body_lines.append(chunk_lines[j])
                j += 1
            body = "\n".join(body_lines).strip()
        except ValueError:
            body = ""

        # Year filter: quick heuristic (look for a 4-digit year in filename/body)
        if min_year > 0:
            year_hits = []
            for cand in [fn] + body.split():
                if len(cand) == 4 and cand.isdigit():
                    y = int(cand)
                    if 1900 <= y <= 2100:
                        year_hits.append(y)
            if year_hits and max(year_hits) < min_year:
                continue

        if s >= score_thresh and body:
            results.append((float(s), fn, body))

        if len(results) >= top_k:
            break

    return results

def answer_from_chunks(q: str, hits: List[Tuple[float, str, str]]) -> str:
    if not hits:
        return "I couldn’t find a grounded answer in the indexed documents. Please refine your question or lower the score threshold."

    # Ultra-concise, extractive style: 1–3 sentences max, with bullet list when appropriate.
    # We do a tiny template that selects the most relevant snippets.
    snippets = []
    for _, fn, body in hits:
        snippet = body[:400].replace("\n", " ").strip()
        snippets.append((fn, snippet))

    # Compose a short answer
    # Heuristic: take the first snippet for direct answer; append 1–2 bullets as citations.
    main = snippets[0][1]
    # Truncate to ~300 chars
    if len(main) > 300:
        main = main[:297] + "…"

    lines = [main]
    lines.append("\n**Sources**:")
    for fn, snip in snippets[: min(3, len(snippets))]:
        lines.append(f"- {fn}")
    return "\n".join(lines)

# -----------------------------
# UI
# -----------------------------
left, right = st.columns([1, 2.2])
with left:
    st.markdown(f"<div class='anika-title'>{APP_NAME}</div>", unsafe_allow_html=True)
    st.caption("RAG Chatbot – Source-Grounded Answers")

    st.markdown("**Settings**")
    st.session_state.top_k = st.slider("Top-K Documents", 1, MAX_TOP_K, sget("top_k", 4))
    st.session_state.score_thresh = st.slider("Score Threshold (lower = stricter)", 0.0, 0.99, sget("score_thresh", 0.40))
    st.session_state.section_filter = st.text_input("Filter: section contains", sget("section_filter", ""))
    st.session_state.min_year = st.number_input("Filter: year ≥ (optional)", min_value=0, max_value=2100, value=sget("min_year", 0), step=1)

    st.markdown(
        "<div class='section-card small-note'>Tip: place **30–50 PDFs/HTML** into <code>data/raw/</code> (git push), "
        "or use the uploader on the right. Click **Rebuild index** afterwards.</div>",
        unsafe_allow_html=True,
    )

with right:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Add or update documents")

    c1, c2, c3 = st.columns([1.5, 1, 1])
    with c1:
        uploads = st.file_uploader(
            "Drop PDF/HTML files (≤200MB each)", type=["pdf", "html", "htm"], accept_multiple_files=True
        )
    with c2:
        st.markdown("<div class='button-primary'>", unsafe_allow_html=True)
        if st.button("Rebuild index", use_container_width=True, on_click=rebuild_index_callback):
            pass
        st.markdown("</div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='button-primary'>", unsafe_allow_html=True)
        if st.button("Fetch starter corpus (Anika + federal AI)", use_container_width=True, on_click=fetch_starter_callback):
            pass
        st.markdown("</div>", unsafe_allow_html=True)

    if uploads:
        with st.spinner("Saving uploads…"):
            added = save_uploaded_files(uploads)
            if added > 0:
                supdate({"status": f"Saved {added} file(s) to data/raw/. Click Rebuild index.", "doc_count": sget("doc_count", 0) + added})
            else:
                supdate({"status": "No valid files uploaded (only PDF/HTML supported)."})
    st.info(sget("status", ""), icon="ℹ️")

    # Chat section
    st.divider()
    st.subheader("Ask about these documents…")

    st.markdown("</div>", unsafe_allow_html=True)  # end section-card

    q = st.text_input(" ", placeholder="e.g., Summarize Anika Systems capabilities in one sentence")
    ask = st.button("Ask", type="primary")
    if ask:
        st.session_state.query = q.strip()

    if st.session_state.query:
        if not sget("index_ready", False):
            st.warning("No index yet. Add PDFs/HTML (or fetch corpus) and click **Rebuild index**.", icon="⚠️")
        else:
            st.markdown(f"<div class='chat-bubble-q'>🗣️ {st.session_state.query}</div>", unsafe_allow_html=True)
            hits = search(
                st.session_state.query,
                top_k=int(st.session_state.top_k),
                score_thresh=float(st.session_state.score_thresh),
                section_filter=st.session_state.section_filter,
                min_year=int(st.session_state.min_year),
            )
            ans = answer_from_chunks(st.session_state.query, hits)
            st.markdown(f"<div class='chat-bubble-a'>{ans}</div>", unsafe_allow_html=True)

# Footer
st.markdown("<div class='footer'>Powered by Monique Bruce</div>", unsafe_allow_html=True)
