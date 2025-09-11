# app.py
# Anika Systems • RAG Chatbot — Source-Grounded Answers
# Streamlit 1.49+ / Python 3.13 compatible — no session_state writes inside cached funcs.

from __future__ import annotations
import os, io, json, re, time, hashlib, pathlib, textwrap
from typing import List, Tuple, Dict, Any
import streamlit as st

# Lightweight parsing
from bs4 import BeautifulSoup
from pypdf import PdfReader

# Embeddings / Vector store
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

########################################################################################
# Paths / constants
########################################################################################
APP_NAME = "Anika Systems"
POWERED_BY = "Powered by Monique Bruce"
RAW_DIR = pathlib.Path("data/raw")
IDX_DIR = pathlib.Path("data/index")
CHUNK_JSON = IDX_DIR / "chunks.json"
FAISS_FILE = IDX_DIR / "faiss.index"
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

RAW_DIR.mkdir(parents=True, exist_ok=True)
IDX_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_EXT = {".pdf", ".html", ".htm"}

DEFAULT_TOPK = 4
DEFAULT_THRESH = 0.40

########################################################################################
# Styling (black + blue)
########################################################################################
st.set_page_config(
    page_title=f"{APP_NAME} — RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

PRIMARY = "#65A9FF"      # blue
BG_DARK = "#0f172a"      # slate-900
PANEL = "#111827"        # gray-900
TEXT = "#E5E7EB"         # gray-200
MUTED = "#94A3B8"        # slate-400
ACCENT = "#1e40af"       # blue-800

st.markdown(f"""
<style>
/* base */
.stApp {{ background: radial-gradient(50% 50% at 50% 0%, #0b1222 0%, {BG_DARK} 60%); color:{TEXT}; }}
h1, h2, h3, h4, h5, h6 {{ color: {TEXT}; }}
.block-container {{ padding-top: 1.2rem; padding-bottom: 2rem; }}
/* cards */
.card {{
  background:{PANEL};
  border:1px solid #1f2937;
  border-radius:12px; padding:14px 16px; box-shadow:0 0 0 1px rgba(255,255,255,0.02) inset;
}}
small, .muted {{ color:{MUTED}; }}
/* buttons */
.stButton>button {{
  background:{ACCENT}; border-radius:8px; border:0; color:white; font-weight:600;
}}
/* inputs */
[data-baseweb="input"] input {{ color:{TEXT}; }}
/* sliders text */
.css-184tjsw, .css-17z6c7b, .css-q8sbsg {{ color:{TEXT} !important; }}
/* footer */
.footer {{ margin-top:10px; text-align:center; color:{MUTED}; font-size:12px; }}
</style>
""", unsafe_allow_html=True)

########################################################################################
# Session state (init only here; never mutate inside cached/resource funcs)
########################################################################################
def init_state():
    ss = st.session_state
    defaults = dict(
        top_k=DEFAULT_TOPK,
        threshold=DEFAULT_THRESH,
        filter_contains="",
        filter_year_min=0,
        index_ready=False,
        index_stats="",
        # runtime
        last_build_started=None,
        last_build_finished=None,
        model_loaded=False,
    )
    for k, v in defaults.items():
        if k not in ss:
            ss[k] = v

init_state()

########################################################################################
# Utilities
########################################################################################
def clean_text(t: str) -> str:
    t = re.sub(r"\s+", " ", t or "").strip()
    return t

def file_sha1(path: pathlib.Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def chunk_text(t: str, max_tokens: int = 220, overlap: int = 40) -> List[str]:
    """Simple token-ish chunking by words (MiniLM tolerates this well)."""
    words = t.split()
    if not words:
        return []
    chunks, start = [], 0
    step = max_tokens - overlap
    while start < len(words):
        end = min(len(words), start + max_tokens)
        chunks.append(" ".join(words[start:end]))
        start += step
    return chunks

def read_html(path: pathlib.Path) -> str:
    try:
        with open(path, "rb") as f:
            soup = BeautifulSoup(f.read(), "lxml")
        # common noise removal
        for s in soup(["script", "style", "noscript", "header", "footer", "nav", "form"]):
            s.extract()
        txt = clean_text(soup.get_text(" "))
        return txt
    except Exception as e:
        return ""

def read_pdf(path: pathlib.Path) -> str:
    try:
        reader = PdfReader(str(path))
        pages = [clean_text(page.extract_text() or "") for page in reader.pages]
        return " ".join(p for p in pages if p)
    except Exception:
        return ""

def load_file_text(path: pathlib.Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return read_pdf(path)
    elif ext in {".html", ".htm"}:
        return read_html(path)
    return ""

########################################################################################
# Model & Index (never write to session_state inside these)
########################################################################################
@st.cache_resource(show_spinner=False)
def load_model(model_name: str = EMB_MODEL_NAME):
    model = SentenceTransformer(model_name)
    dim = model.get_sentence_embedding_dimension()
    return model, dim

@st.cache_resource(show_spinner=False)
def new_index(dim: int):
    # Flat L2 index; cosine via normalized vectors
    return faiss.IndexFlatIP(dim)

def embed_texts(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    # normalize for cosine similarity (then use inner product)
    embs = model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    return np.asarray(embs, dtype="float32")

def scan_raw_files(min_year: int = 0, contains: str = "") -> List[pathlib.Path]:
    files = []
    for p in RAW_DIR.glob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXT:
            if min_year:
                try:
                    # guess year from name
                    yrs = re.findall(r"(20\d{2}|19\d{2})", p.name)
                    if yrs and int(yrs[0]) < min_year:
                        continue
                except Exception:
                    pass
            if contains:
                if contains.lower() not in p.name.lower():
                    continue
            files.append(p)
    return sorted(files, key=lambda q: q.name.lower())

def build_index_ui(files: List[pathlib.Path]) -> Tuple[faiss.Index, List[Dict[str, Any]], int]:
    """
    Parse -> chunk -> embed -> FAISS
    Returns (index, chunks_meta, n_chunks)
    """
    model, dim = load_model()
    index = new_index(dim)

    all_chunks: List[str] = []
    meta: List[Dict[str, Any]] = []

    for fp in files:
        txt = load_file_text(fp)
        if not txt:
            continue
        # gentle guard against html nav spam
        txt = re.sub(r"(?:\b(Privacy|Cookies|Accept|Subscribe|Share)\b.{0,30}){2,}", " ", txt, flags=re.I)
        chunks = chunk_text(txt, max_tokens=220, overlap=50)
        for i, c in enumerate(chunks):
            all_chunks.append(c)
            meta.append({
                "file": fp.name,
                "path": str(fp),
                "chunk_id": f"{fp.name}:::{i}",
            })

    if not all_chunks:
        return index, [], 0

    embs = embed_texts(model, all_chunks)
    index.add(embs)  # shape (N, dim)

    # persist to disk (best effort; not relied on for cache)
    try:
        faiss.write_index(index, str(FAISS_FILE))
        with open(CHUNK_JSON, "w") as f:
            json.dump({"meta": meta}, f)
    except Exception:
        pass

    return index, meta, len(all_chunks)

@st.cache_resource(show_spinner=False)
def load_persisted_index() -> Tuple[faiss.Index|None, List[Dict[str, Any]]]:
    if FAISS_FILE.exists() and CHUNK_JSON.exists():
        try:
            index = faiss.read_index(str(FAISS_FILE))
            meta = json.load(open(CHUNK_JSON))["meta"]
            return index, meta
        except Exception:
            return None, []
    return None, []

def ensure_index(min_year: int, contains: str) -> Tuple[faiss.Index|None, List[Dict[str, Any]], str]:
    """Decides whether to rebuild or reuse; returns (index, meta, stats_str)."""
    files = scan_raw_files(min_year=min_year, contains=contains)
    stats = f"{len(files)} files detected in data/raw/"
    if not files:
        return None, [], stats + " — add PDFs/HTML then click Rebuild index."

    # Always (re)build live to reflect filters; no session_state writes here.
    idx, meta, n = build_index_ui(files)
    stats = f"Index built: {n} chunks from {len(set([m['file'] for m in meta]))} files."
    return idx if n else None, meta, stats

def search(index: faiss.Index, meta: List[Dict[str, Any]], query: str, k: int) -> List[Tuple[float, Dict[str, Any]]]:
    model, _ = load_model()
    q = embed_texts(model, [query])
    sims, idxs = index.search(q, k)  # inner product on normalized embeddings -> cosine
    hits: List[Tuple[float, Dict[str, Any]]] = []
    for score, i in zip(sims[0], idxs[0]):
        if i == -1: 
            continue
        m = meta[int(i)]
        hits.append((float(score), m))
    return hits

########################################################################################
# Small, grounded answerer (concise; 2–4 sentences; no hallucinations)
########################################################################################
def generate_answer(query: str, hits: List[Tuple[float, Dict[str, Any]]], chunks: List[str], thresh: float) -> Tuple[str, List[Tuple[float,str]]]:
    kept = []
    for score, m in hits:
        if score >= thresh:
            kept.append((score, m))
    if not kept:
        return ("I couldn’t find a high-confidence answer in the uploaded sources. "
                "Try lowering the score threshold or adding more documents." , [])

    # pull text for kept chunks
    evidences = []
    for score, m in kept:
        # m['chunk_id'] ends with :::i
        try:
            i = int(m["chunk_id"].split(":::")[-1])
        except Exception:
            i = 0
        evidences.append((score, m["file"], chunks[i]))

    # Compose short answer by extractive synthesis
    # Take top 2–3 spans that contain the main keywords
    spans = []
    q_terms = [w.lower() for w in re.findall(r"[a-zA-Z0-9]+", query) if len(w) > 2]
    for sc, fname, txt in evidences[:6]:
        # pick a 1–2 sentence window around most overlap
        sents = re.split(r"(?<=[.!?])\s+", txt)
        best = sorted(sents, key=lambda s: -sum(t in s.lower() for t in q_terms))[:2]
        spans.extend(best)

    summary = " ".join(spans[:3])
    summary = clean_text(summary)
    if len(summary) < 40:
        # fall back to first evidence chunk trimmed
        summary = clean_text(evidences[0][2])[:400]

    # Trim to ~3 sentences
    summary_sents = re.split(r"(?<=[.!?])\s+", summary)[:3]
    answer = " ".join(summary_sents)
    # Citations (top 3)
    cits = [(sc, fn) for sc, fn, _ in evidences[:3]]
    return answer, cits

########################################################################################
# UI
########################################################################################
st.markdown(f"<h1 style='margin:0 0 8px 0;'>{APP_NAME}</h1>", unsafe_allow_html=True)
st.markdown("<div class='muted'>RAG Chatbot — Source-Grounded Answers</div>", unsafe_allow_html=True)
st.write("")

with st.sidebar:
    st.subheader("Settings")
    st.session_state.top_k = st.slider("Top-K Documents", 1, 10, st.session_state.top_k)
    st.session_state.threshold = st.slider("Score Threshold (lower = stricter)", 0.10, 0.95, st.session_state.threshold, 0.01)
    st.session_state.filter_contains = st.text_input("Filter: section contains", st.session_state.filter_contains)
    st.session_state.filter_year_min = st.number_input("Filter: year ≥ (optional)", min_value=0, max_value=2100, step=1, value=st.session_state.filter_year_min)
    st.markdown(
        "<div class='card'><small>Tip: place <b>30–50 PDFs/HTML</b> into <code>data/raw/</code> (git push), or use the uploader below. Then click <b>Rebuild index</b>.</small></div>",
        unsafe_allow_html=True
    )

st.markdown("### Add or update documents")
col_u, col_rebuild, col_fetch = st.columns([6,2,3])

with col_u:
    uploaded = st.file_uploader("Drop PDF/HTML files", type=["pdf","html","htm"], accept_multiple_files=True, label_visibility="collapsed")
    if uploaded:
        saved = 0
        for uf in uploaded:
            ext = os.path.splitext(uf.name)[1].lower()
            if ext not in SUPPORTED_EXT: 
                continue
            target = RAW_DIR / uf.name
            with open(target, "wb") as f:
                f.write(uf.getbuffer())
            saved += 1
        st.success(f"Saved {saved} file(s) into data/raw/.")

with col_rebuild:
    do_rebuild = st.button("Rebuild index", use_container_width=True)

with col_fetch:
    # Best effort starter corpus — if network blocks, we just explain the failure.
    def fetch_starter_files() -> Tuple[int,int,List[str]]:
        import requests
        # Curated, crawl-friendly public pages (HTML) only.
        urls = [
            "https://www.anikasystems.com/", 
            "https://www.anikasystems.com/who-we-are",
            "https://www.anikasystems.com/capabilities",
            "https://www.anikasystems.com/careers",
            "https://www.anikasystems.com/insights",
            "https://www.anikasystems.com/contact",
            "https://ai.gov/strategic-plan/",  # federal AI (public)
            "https://www.whitehouse.gov/ostp/ai-bill-of-rights/",
            "https://www.whitehouse.gov/wp-content/uploads/2023/10/Executive-Order-Safe-Secure-and-Trustworthy-Development-and-Use-of-AI.pdf",
        ]
        ok, fail, failed = 0, 0, []
        for u in urls:
            try:
                r = requests.get(u, timeout=12)
                r.raise_for_status()
                name = re.sub(r"[^a-z0-9]+", "_", u.lower()).strip("_")[:70]
                if u.lower().endswith(".pdf"):
                    path = RAW_DIR / f"{name}.pdf"
                    open(path, "wb").write(r.content)
                else:
                    path = RAW_DIR / f"{name}.html"
                    open(path, "wb").write(r.content)
                ok += 1
            except Exception:
                fail += 1
                failed.append(u)
        return ok, fail, failed

    if st.button("Fetch starter corpus (Anika + federal AI)", use_container_width=True):
        with st.spinner("Fetching starter files…"):
            try:
                ok, fail, failed = fetch_starter_files()
                if ok:
                    st.success(f"Fetched {ok} file(s) into data/raw/.")
                if fail:
                    st.warning(f"Some sources may be blocked or unavailable on this network (failed={fail}). You can add more via upload.")
            except Exception:
                st.warning("Could not fetch starter files (network blocked or sources unavailable).")

# Rebuild index if asked (do not write to session_state inside cached funcs)
index, meta, stats = None, [], ""
if do_rebuild:
    with st.spinner("Building index…"):
        index, meta, stats = ensure_index(
            min_year=st.session_state.filter_year_min,
            contains=st.session_state.filter_contains.strip()
        )
    st.session_state.index_ready = index is not None
    st.session_state.index_stats = stats

# If we have a persisted index and no rebuild request, try to load and report
if not do_rebuild and not st.session_state.index_ready:
    idx_loaded, meta_loaded = load_persisted_index()
    if idx_loaded is not None and meta_loaded:
        index, meta = idx_loaded, meta_loaded
        st.session_state.index_ready = True
        st.session_state.index_stats = f"Index loaded from disk: {len(meta)} chunks from {len(set(m['file'] for m in meta))} files."

# Status banner
if st.session_state.index_stats:
    st.info(st.session_state.index_stats, icon="🔎")
else:
    st.warning("No index yet. Add at least one PDF or HTML to data/raw/ and click Rebuild index.", icon="ℹ️")

########################################################################################
# Ask section
########################################################################################
st.markdown("### Ask about these documents…")

q = st.text_input("Ask about the documents", placeholder="e.g., Who is Anika Systems? What capabilities do they offer for federal agencies?")
ask = st.button("Ask", type="primary")

if ask:
    if not st.session_state.index_ready:
        st.warning("Chat is disabled until an index exists. Add/fetch documents and click Rebuild index.", icon="💡")
    else:
        index_active, meta_active = None, None

        # Prefer the live in-memory index if we just rebuilt; otherwise, load persisted.
        idx_loaded, meta_loaded = load_persisted_index()
        if idx_loaded is not None and meta_loaded:
            index_active, meta_active = idx_loaded, meta_loaded
        else:
            # last build was not persisted; build minimal ephemeral index from current filters
            index_active, meta_active, _ = ensure_index(
                min_year=st.session_state.filter_year_min,
                contains=st.session_state.filter_contains.strip()
            )

        if index_active is None or not meta_active:
            st.warning("No searchable index available yet. Rebuild the index first.", icon="ℹ️")
        else:
            # we also need the chunk texts to construct the answer (read from disk)
            # If CHUNK_JSON present, we pull only meta; we will reconstruct chunk list by re-parsing the referenced files.
            files_map: Dict[str, List[str]] = {}
            for m in meta_active:
                files_map.setdefault(m["file"], []).append(m["chunk_id"])

            # Recreate chunk list deterministically by reading each file and chunking in the same way
            chunks: List[str] = []
            file_to_start: Dict[str, int] = {}
            i_start = 0
            seen = set()
            for m in meta_active:
                fname = m["file"]
                if fname in seen: 
                    continue
                seen.add(fname)
                path = pathlib.Path(m["path"])
                txt = load_file_text(path)
                cs = chunk_text(txt, max_tokens=220, overlap=50)
                file_to_start[fname] = i_start
                chunks.extend(cs)
                i_start += len(cs)

            hits = search(index_active, meta_active, q, k=max(1, int(st.session_state.top_k)))
            answer, cits = generate_answer(q, hits, chunks, float(st.session_state.threshold))

            with st.container():
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown(f"**🔎 {q.strip()}**")
                st.write(answer)

                if cits:
                    st.markdown("**Sources:**")
                    for sc, fn in cits:
                        st.markdown(f"- `{fn}`  · similarity={sc:.2f}")
                st.markdown("</div>", unsafe_allow_html=True)

########################################################################################
# Footer
########################################################################################
st.markdown(f"<div class='footer'>{POWERED_BY}</div>", unsafe_allow_html=True)
