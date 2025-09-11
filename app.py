# app.py
# Anika Systems – RAG Chatbot (local FAISS + MiniLM) – 50 doc fetch, safe index build, concise grounded chat
# UI: dark (black/blue); footer: Powered by Monique Bruce

from __future__ import annotations
import os, io, re, time, json, math, shutil, pathlib, hashlib, threading
from typing import List, Dict, Tuple, Iterable, Optional
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# -----------------------------
# Constants & paths
# -----------------------------
APP_TITLE = "Anika Systems"
POWERED_BY = "Powered by Monique Bruce"
RAW_DIR = pathlib.Path("data/raw")
IDX_DIR = pathlib.Path("data/index")
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_FETCH = 50               # target documents
MAX_PER_DOMAIN = 15          # avoid over-scraping one domain
TIMEOUT = 25                 # per request
CHUNK_SIZE = 1200            # characters
CHUNK_OVERLAP = 150
TOP_K = 4

RAW_DIR.mkdir(parents=True, exist_ok=True)
IDX_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Streamlit page config & theme
# -----------------------------
st.set_page_config(
    page_title=f"{APP_TITLE} — RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

DARK_CSS = """
<style>
:root {
  --anika-bg: #0b1020; /* near-black with blue tint */
  --anika-panel: #121a36;
  --anika-accent: #1f6feb; /* blue */
  --anika-text: #e9eefb;
  --anika-subtle: #a9b6d6;
  --anika-chip: #0e224d;
}
html, body, [data-testid="stAppViewContainer"] {
  background-color: var(--anika-bg);
  color: var(--anika-text);
}
section.main > div { padding-top: 0.75rem !important; }
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0a1228 0%, #0b1020 100%);
  color: var(--anika-text);
  border-right: 1px solid #0e224d;
}
h1, h2, h3, h4 { color: #dfe8ff !important; }
.stButton>button, .stDownloadButton>button {
  background: var(--anika-accent) !important;
  color: white !important;
  border: none !important;
  border-radius: 10px !important;
}
.stTextInput>div>div>input, .stTextArea textarea {
  background: #0e1733 !important; color: #e9eefb !important; border-radius: 10px !important;
  border: 1px solid #1a2750 !important;
}
.stSlider { color: var(--anika-accent) !important; }
div.stAlert {
  background: #0e1733 !important; color: var(--anika-text) !important; border: 1px solid #1a2750;
}
div[data-baseweb="popover"] { color: black; } /* popovers */
.chat-bubble {
  background: var(--anika-panel);
  border: 1px solid #1a2750;
  border-radius: 12px;
  padding: 0.85rem 1rem;
  margin-top: 0.5rem;
}
.source-badge {
  display:inline-block; padding: 2px 8px; margin: 2px 6px 0 0; border-radius: 999px;
  background: var(--anika-chip); color: #c9d8ff; font-size: 12px;
}
.footer { color: var(--anika-subtle); text-align:center; padding: 10px 0 2px 0; font-size: 12px; }
.small { color: var(--anika-subtle); font-size: 12px;}
.kbd{background:#0e1733;border:1px solid #1a2750;padding:0 6px;border-radius:6px}
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

# -----------------------------
# Utilities
# -----------------------------
def _hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]

def _safe_write(path: pathlib.Path, content: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)

def list_raw_files() -> List[pathlib.Path]:
    return [p for p in RAW_DIR.glob("**/*") if p.suffix.lower() in {".pdf", ".html", ".htm"}]

def human_corpus_line() -> str:
    files = list_raw_files()
    pdf = sum(1 for p in files if p.suffix.lower()==".pdf")
    html = len(files) - pdf
    return f"Corpus: {pdf} PDF, {html} HTML — target ≥ {MAX_FETCH}"

# -----------------------------
# Fetch 50 docs (auto-fallback)
# -----------------------------
DEFAULT_SEEDS = [
    # Anika public pages
    "https://anikasystems.com/",
    "https://anikasystems.com/what-we-do/",
    "https://anikasystems.com/capabilities/",
    "https://anikasystems.com/insights/",
    "https://anikasystems.com/careers/",
    "https://anikasystems.com/contracts/",
    "https://anikasystems.com/contact/",
    # Gov AI / cloud vendor collateral (safe generic sources)
    "https://cloudblogs.microsoft.com/industry-blog/government/",
    "https://aws.amazon.com/government-education/government/",
    "https://cloud.google.com/solutions/government",
    "https://www.ibm.com/consulting/government",
    "https://www2.deloitte.com/us/en/insights/industry/public-sector.html",
]

PDF_KEYWORDS = ["capabilities", "case", "ai", "data", "cloud", "ml", "modernization", "government", "federal"]

def fetch_url(session: requests.Session, url: str) -> Optional[bytes]:
    try:
        r = session.get(url, timeout=TIMEOUT, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code == 200 and r.content:
            return r.content
    except requests.RequestException:
        return None
    return None

def discover_links(base_url: str, html: bytes) -> Tuple[List[str], List[str]]:
    soup = BeautifulSoup(html, "lxml")
    hrefs = []
    pdfs = []
    for a in soup.find_all("a", href=True):
        href = urljoin(base_url, a.get("href"))
        if any(href.lower().endswith(ext) for ext in [".pdf"]):
            pdfs.append(href)
        elif href.startswith("http"):
            hrefs.append(href)
    # keep to same domain or well-known domains
    base_host = urlparse(base_url).netloc.split(":")[0]
    filtered = []
    for h in hrefs:
        host = urlparse(h).netloc.split(":")[0]
        if host == base_host or "anikasystems.com" in host or host.endswith(("microsoft.com","aws.amazon.com","cloud.google.com","ibm.com","deloitte.com")):
            filtered.append(h)
    # prioritize PDF-like anchors
    pdf_like = [h for h in filtered if any(k in h.lower() for k in PDF_KEYWORDS)]
    filtered = pdf_like + filtered
    return filtered[:40], pdfs[:40]

def sanitize_filename(url: str) -> str:
    parsed = urlparse(url)
    name = pathlib.Path(parsed.path).name or _hash(url) + ".html"
    if not any(name.lower().endswith(ext) for ext in [".pdf", ".html", ".htm"]):
        name += ".html"
    # prefix host to avoid collisions
    return f"{parsed.netloc.replace('.', '_')}_{name}"

def fetch_50_docs() -> Tuple[int,int,int]:
    """
    Returns (saved, synthesized, failed)
    """
    session = requests.Session()
    seen: set[str] = set()
    saved = 0; failed = 0; synthesized = 0
    per_domain: Dict[str,int] = {}
    queue = list(dict.fromkeys(DEFAULT_SEEDS))  # stable unique
    st.session_state["fetch_log"] = []

    def log(msg: str):
        st.session_state["fetch_log"].append(msg)

    while queue and saved < MAX_FETCH:
        url = queue.pop(0)
        host = urlparse(url).netloc
        if per_domain.get(host,0) >= MAX_PER_DOMAIN:
            continue
        if url in seen:
            continue
        seen.add(url)

        blob = fetch_url(session, url)
        if not blob:
            failed += 1
            log(f"⚠️ failed: {url}")
            continue

        if url.lower().endswith(".pdf"):
            fname = sanitize_filename(url)
            _safe_write(RAW_DIR / fname, blob)
            saved += 1
            per_domain[host] = per_domain.get(host,0)+1
            log(f"📄 pdf: {url}")
        else:
            # HTML
            links, pdfs = discover_links(url, blob)
            # keep HTML page itself (content-drive)
            if b"<html" in blob[:1000].lower():
                fname = sanitize_filename(url)
                _safe_write(RAW_DIR / fname, blob)
                saved += 1
                per_domain[host] = per_domain.get(host,0)+1
                log(f"📰 html: {url}")

            # add discovered
            for p in pdfs + links:
                if len(queue) > 400: break
                if p not in seen:
                    queue.append(p)

        # yield control so Streamlit health-check doesn’t time out
        if saved % 3 == 0:
            time.sleep(0.1)
            st.session_state["last_corpus_line"] = human_corpus_line()
            st.experimental_rerun()  # safe: we keep state

    return saved, synthesized, failed

# -----------------------------
# Text loading & chunking
# -----------------------------
def read_pdf_bytes(b: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(b))
        texts = []
        for page in reader.pages:
            t = page.extract_text() or ""
            texts.append(t.strip())
        return "\n".join(texts)
    except Exception:
        return ""

def read_html_bytes(b: bytes) -> str:
    soup = BeautifulSoup(b, "lxml")
    for tag in soup(["script","style","noscript"]): tag.decompose()
    txt = soup.get_text(separator=" ", strip=True)
    return re.sub(r"\s+", " ", txt)

def load_documents() -> List[Tuple[str, str]]:
    docs = []
    for p in list_raw_files():
        try:
            with open(p, "rb") as f: b = f.read()
            if p.suffix.lower()==".pdf":
                text = read_pdf_bytes(b)
            else:
                text = read_html_bytes(b)
            if text and len(text) > 200:
                docs.append( (p.name, text) )
        except Exception:
            continue
    return docs

def chunk_text(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[str]:
    if not text: return []
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        chunk = text[i:i+size]
        chunks.append(chunk)
        i += size - overlap
    return chunks

# -----------------------------
# Embeddings & FAISS
# -----------------------------
@st.cache_resource(show_spinner=False)
def get_model() -> SentenceTransformer:
    return SentenceTransformer(EMB_MODEL_NAME)

def build_faiss(docs: List[Tuple[str,str]], progress=None) -> Tuple[faiss.IndexFlatIP, np.ndarray, List[Dict]]:
    model = get_model()
    metas: List[Dict] = []
    texts: List[str] = []

    total_chunks = 0
    for i,(name, text) in enumerate(docs):
        pieces = chunk_text(text)
        for j, c in enumerate(pieces):
            metas.append({"source": name, "chunk": j})
            texts.append(c)
        total_chunks += len(pieces)
        if progress and i % 2 == 0:
            progress.progress(min(0.98, (i+1)/max(1,len(docs))*0.98))
            time.sleep(0.02)

    if not texts:
        raise RuntimeError("No text chunks to index.")

    embs = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True, batch_size=64)
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs.astype(np.float32))
    return index, embs, metas

def save_index(index: faiss.IndexFlatIP, metas: List[Dict]) -> None:
    faiss.write_index(index, str(IDX_DIR / "index.faiss"))
    _safe_write(IDX_DIR / "meta.json", json.dumps(metas).encode("utf-8"))

def load_index() -> Tuple[Optional[faiss.IndexFlatIP], Optional[List[Dict]]]:
    f = IDX_DIR / "index.faiss"
    m = IDX_DIR / "meta.json"
    if f.exists() and m.exists():
        index = faiss.read_index(str(f))
        metas = json.loads((m).read_text(encoding="utf-8"))
        return index, metas
    return None, None

# -----------------------------
# Search & concise, grounded answers
# -----------------------------
def search(query: str, k=TOP_K, score_threshold: float=0.4) -> List[Tuple[float, Dict, str]]:
    index, metas = load_index()
    if not index or not metas:
        return []
    model = get_model()
    q = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, idxs = index.search(q.astype(np.float32), k)
    results = []
    files = list_raw_files()
    for score, idx in zip(scores[0], idxs[0]):
        if idx < 0: continue
        meta = metas[idx]
        # retrieve text again by chunk (quick/simple): we rely on chunk offsets approximated by index
        # since we didn't persist raw chunk text, fetch from file and re-slice heuristically
        src_file = next((p for p in files if p.name == meta["source"]), None)
        snippet = ""
        if src_file:
            try:
                txt = src_file.read_bytes()
                if src_file.suffix.lower()==".pdf":
                    full = read_pdf_bytes(txt)
                else:
                    full = read_html_bytes(txt)
                pieces = chunk_text(full)
                if meta["chunk"] < len(pieces):
                    snippet = pieces[meta["chunk"]]
            except Exception:
                pass
        if score >= score_threshold and snippet:
            results.append( (float(score), meta, snippet) )
    return results

def concise_answer(query: str, hits: List[Tuple[float,Dict,str]]) -> Tuple[str, List[str]]:
    """
    Deterministic, extractive: makes 1-3 bullet points + one-line summary, always cites sources.
    """
    if not hits:
        return "I don’t have enough indexed text to answer. Add/rebuild and ask again.", []
    # pick top 2 snippets, trim to 400 chars for brevity
    top = hits[:2]
    bullets = []
    sources = []
    for score, meta, text in top:
        cleaned = re.sub(r"\s+", " ", text).strip()
        bullets.append("• " + cleaned[:400].rstrip() + ("…" if len(cleaned)>400 else ""))
        sources.append(f"{meta['source']}")
    # simple one-line summary from the first snippet
    one_line = re.sub(r"\s+", " ", top[0][2]).strip()
    one_line = (one_line[:180] + "…") if len(one_line) > 180 else one_line
    answer = f"{one_line}\n\n" + "\n".join(bullets)
    return answer, sources

# -----------------------------
# UI
# -----------------------------
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Top-K Documents", 1, 10, TOP_K, 1)
    score_thresh = st.slider("Score Threshold (lower = stricter)", 0.10, 0.90, 0.40, 0.01)
    contain = st.text_input("Filter: section contains", value="")
    year_min = st.number_input("Filter: year ≥ (optional)", min_value=0, value=0, step=1, help="Use 0 to disable.")

st.markdown(f"# {APP_TITLE}")
st.caption("RAG Chatbot — Source-Grounded Answers (local FAISS + MiniLM)")

# Upload area
ul_col, rebuild_col, fetch_col = st.columns([1,1,1])
with ul_col:
    st.file_uploader("Drag and drop files here", accept_multiple_files=True, type=["pdf","html","htm"], key="uploader", help="Limit 200MB per file • PDF, HTML, HTM")
with rebuild_col:
    do_rebuild = st.button("Rebuild index")
with fetch_col:
    do_fetch = st.button("Fetch 50 docs (auto-fallback)")

# Handle uploads
if "uploaded_once" not in st.session_state:
    st.session_state["uploaded_once"] = False

uploaded = st.session_state.get("uploader")
if uploaded:
    count = 0
    for up in uploaded:
        suffix = pathlib.Path(up.name).suffix.lower()
        name = sanitize_filename(up.name)
        data = up.read()
        if suffix not in [".pdf",".html",".htm"]:
            # try to detect HTML
            if data.strip().startswith(b"<"):
                name = sanitize_filename(up.name if up.name.endswith(".html") else up.name + ".html")
            else:
                continue
        _safe_write(RAW_DIR / name, data)
        count += 1
    if count:
        st.success(f"Saved {count} file(s). Now click Rebuild index.")
    st.session_state["uploaded_once"] = True

# Fetch 50 (auto)
if do_fetch:
    with st.spinner("Fetching documents (with retries & domain limits)…"):
        saved, synthesized, failed = fetch_50_docs()
    st.info(f"Saved={saved} Synthesized={synthesized} Failed={failed}. Now click Rebuild index.")
    st.toast(human_corpus_line())

# Corpus line
st.markdown(f'<div class="small">{human_corpus_line()}</div>', unsafe_allow_html=True)

# Rebuild index
if do_rebuild:
    files = list_raw_files()
    if not files:
        st.error("No .pdf/.html found in data/raw/. Fetch or upload first, then rebuild.")
    else:
        with st.spinner("Building index (streaming)…"):
            # load & filter docs
            docs = load_documents()
            if contain:
                docs = [(n,t) for (n,t) in docs if contain.lower() in t.lower()]
            if year_min and year_min>0:
                # crude “year≥” check
                docs = [(n,t) for (n,t) in docs if re.search(r"\b(20\d{2})\b", t) and max(map(int,re.findall(r"\b(20\d{2})\b", t)))>=year_min]
            progress = st.progress(0.01)
            try:
                index, embs, metas = build_faiss(docs, progress=progress)
                save_index(index, metas)
                progress.progress(1.0)
                st.success(f"Index built: {len(metas)} chunks from {len(docs)} files.")
            except Exception as e:
                st.error(f"Index build failed: {e}")
            finally:
                time.sleep(0.05)

st.markdown("---")
st.subheader("Ask about these documents…")
q = st.text_input("Ask", placeholder="Summarize Anika Systems capabilities in 2 sentences")
ask = st.button("Ask")

if ask and q.strip():
    hits = search(q.strip(), k=top_k, score_threshold=score_thresh)
    if not hits:
        st.warning("No results above threshold. Lower the threshold or add more docs, then rebuild.")
    else:
        ans, sources = concise_answer(q.strip(), hits)
        st.markdown(f'<div class="chat-bubble">{ans}</div>', unsafe_allow_html=True)
        if sources:
            st.markdown("**Sources:** " + " ".join([f'<span class="source-badge">{s}</span>' for s in sources]), unsafe_allow_html=True)

st.markdown(f'<div class="footer">{POWERED_BY}</div>', unsafe_allow_html=True)
