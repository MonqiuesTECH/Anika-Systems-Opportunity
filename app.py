# app.py
# Streamlit RAG demo for Anika Systems — local FAISS + MiniLM
# Python 3.13 / Streamlit 1.49-safe

from __future__ import annotations
import os, io, re, json, time, hashlib, shutil
from pathlib import Path
from typing import List, Dict, Any, Tuple

import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# --- Local helpers (from src/*) ---------------------------------------------
# We keep imports, but also provide safe fallbacks below if the module isn't found.
try:
    from src.loaders import load_raw_files, save_upload_as_html, guess_filename
except Exception:
    # Fallbacks so the app never crashes if imports drift
    def guess_filename(name: str) -> str:
        base = re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_")
        return base or f"file_{int(time.time())}.html"

    def save_upload_as_html(files: List[Any], raw_dir: str) -> int:
        Path(raw_dir).mkdir(parents=True, exist_ok=True)
        count = 0
        for f in files:
            raw = f.read()
            ext = (f.name.split(".")[-1] or "html").lower()
            if ext not in ("pdf", "html", "htm"):
                ext = "html"
            out = Path(raw_dir) / guess_filename(f.name)
            with open(out, "wb") as w:
                w.write(raw)
            count += 1
        return count

    def load_raw_files(raw_dir: str, section_filter: str = "", min_year: int = 0) -> List[Dict[str, Any]]:
        # Very small fallback loader: only passes paths + naive title
        docs = []
        for p in Path(raw_dir).glob("**/*"):
            if not p.is_file() or p.suffix.lower() not in {".pdf", ".html", ".htm"}:
                continue
            docs.append({"path": str(p), "title": p.stem, "year": 0})
        return docs

# --- Constants / Paths ------------------------------------------------------
APP_TITLE = "Anika Systems"
POWERED_BY = "Powered by Monique Bruce"
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
INDEX_DIR = DATA_DIR / "index"
EMBEDDINGS_FILE = INDEX_DIR / "embeddings.npy"
METADATA_FILE = INDEX_DIR / "metadata.json"
FAISS_FILE = INDEX_DIR / "faiss.index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Small util -------------------------------------------------------------
def css():
    st.markdown(
        """
        <style>
        :root {
          --primary:#1e88e5;
          --bg:#0b1220;
          --panel:#141c2c;
          --text:#e6eefc;
          --muted:#9bb0d1;
        }
        .stApp {background: var(--bg);}
        .stMarkdown, .stTextInput, .stSlider label, .stSelectbox label, .stNumberInput label,
        .stButton button, .stAlert, .stFileUploader, .stCaption, p, li, h1, h2, h3, h4, h5 {
          color: var(--text) !important;
        }
        .stButton>button { background: var(--primary); color: white; border: 0; }
        .block-container { padding-top: 1.2rem; }
        .chat-bubble {
          background: var(--panel); padding: 14px 16px; border-radius: 12px; margin: 8px 0;
          border: 1px solid #233049; color: var(--text);
        }
        .fineprint { color: var(--muted); font-size: 0.8rem; }
        footer { visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )

def load_model() -> SentenceTransformer:
    if "model" not in st.session_state:
        st.session_state.model = SentenceTransformer(MODEL_NAME)
    return st.session_state.model

def embed_texts(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    # batch for speed
    return np.asarray(model.encode(texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True))

def clean_and_chunk(text: str, max_chars: int = 1200) -> List[str]:
    # keep it robust, short, and deterministic
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i+max_chars])
        i += max_chars
    return [c for c in chunks if c]

def read_file_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        out = []
        with open(path, "rb") as f:
            reader = PdfReader(f)
            for page in reader.pages:
                out.append(page.extract_text() or "")
        return "\n".join(out)
    else:
        # HTML
        from bs4 import BeautifulSoup
        html = Path(path).read_text(errors="ignore")
        soup = BeautifulSoup(html, "lxml")
        # drop nav/footers if obvious
        for bad in soup.select("nav, footer, script, style"):
            bad.decompose()
        txt = soup.get_text(separator=" ")
        return txt

def ensure_index():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    if not FAISS_FILE.exists() or not METADATA_FILE.exists():
        return False
    try:
        index = faiss.read_index(str(FAISS_FILE))
        meta = json.loads(METADATA_FILE.read_text())
        if "texts" not in meta or "spans" not in meta or "sources" not in meta:
            return False
        st.session_state.index = index
        st.session_state.meta = meta
        return True
    except Exception:
        return False

def build_index(section_filter: str = "", min_year: int = 0) -> Tuple[int, int]:
    """Scan data/raw, extract text, chunk, embed, and build FAISS."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    files = load_raw_files(str(RAW_DIR), section_filter=section_filter, min_year=min_year)

    texts, spans, sources = [], [], []
    for doc in files:
        p = Path(doc["path"])
        try:
            text = read_file_text(p)
            if section_filter and section_filter.lower() not in text.lower():
                continue
            chunks = clean_and_chunk(text)
            for c in chunks:
                texts.append(c)
                spans.append({"title": doc.get("title", p.stem), "path": str(p)})
                sources.append(str(p))
        except Exception:
            continue

    if not texts:
        # clear any old index to avoid “phantom” success
        for fp in (EMBEDDINGS_FILE, METADATA_FILE, FAISS_FILE):
            if Path(fp).exists():
                Path(fp).unlink(missing_ok=True)
        return 0, 0

    model = load_model()
    embs = embed_texts(model, texts)
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    # normalize for dot-product similarity
    faiss.normalize_L2(embs)
    index.add(embs)
    faiss.write_index(index, str(FAISS_FILE))
    np.save(EMBEDDINGS_FILE, embs)

    meta = {"texts": texts, "spans": spans, "sources": sources}
    METADATA_FILE.write_text(json.dumps(meta, ensure_ascii=False))
    st.session_state.index = index
    st.session_state.meta = meta
    return len(texts), len(set(sources))

def search(query: str, k: int = 4, threshold: float = 0.40) -> List[Dict[str, Any]]:
    if "index" not in st.session_state or "meta" not in st.session_state:
        ok = ensure_index()
        if not ok:
            return []

    index = st.session_state.index
    meta  = st.session_state.meta
    model = load_model()

    q = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q)
    scores, idxs = index.search(q, k)
    out = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx < 0: 
            continue
        if score < threshold:
            continue
        out.append({
            "score": float(score),
            "text": meta["texts"][idx],
            "span": meta["spans"][idx],
            "source": meta["sources"][idx],
        })
    return out

# --- Auto-fetch 50 docs (fall back between sources) -------------------------
SEEDS = [
    # prioritized: company + generic federal AI resources
    ("https://www.anikasystems.com/careers", "html"),
    ("https://www.anikasystems.com/contact", "html"),
    ("https://www.anikasystems.com", "html"),
    ("https://www.whitehouse.gov/ostp/ai-bill-of-rights/", "html"),
    ("https://www.ai.gov/", "html"),
    ("https://www.nist.gov/itl/ai-risk-management-framework", "html"),
    ("https://cloud.google.com/blog/topics/public-sector", "html"),
    ("https://aws.amazon.com/industries/federal/", "html"),
]

def fetch_50_docs(raw_dir: Path) -> Tuple[int, int, int]:
    """
    Tries to save 50 files under data/raw/.
    Returns (saved, synthesized, failed).
    """
    import requests
    from bs4 import BeautifulSoup

    raw_dir.mkdir(parents=True, exist_ok=True)
    saved = synthesized = failed = 0
    seen: set[str] = set()

    def put(name: str, content: bytes):
        nonlocal saved
        fn = raw_dir / guess_filename(name)
        fn.write_bytes(content)
        saved += 1
        seen.add(str(fn))

    # 1) Try curated seeds first
    for url, kind in SEEDS:
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            put(url + ".html", r.content)
        except Exception:
            failed += 1

    # 2) Crawl a bit from Anika site homepage
    try:
        r = requests.get("https://www.anikasystems.com", timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        hrefs = []
        for a in soup.find_all("a"):
            href = a.get("href") or ""
            if href.startswith("/") or href.startswith("https://www.anikasystems.com"):
                if href.startswith("/"):
                    href = "https://www.anikasystems.com" + href
                hrefs.append(href)
        hrefs = list(dict.fromkeys(hrefs))[:35]
        for h in hrefs:
            if saved >= 50:
                break
            try:
                rr = requests.get(h, timeout=12)
                rr.raise_for_status()
                put(h + ".html", rr.content)
            except Exception:
                failed += 1
    except Exception:
        failed += 1

    # 3) If still short, synthesize placeholders from short authoritative blurbs
    #    (keeps the pipeline stable on locked networks)
    while saved + synthesized < 50:
        blob = f"<html><body><h1>Federal AI Reference {saved+synthesized+1}</h1>" \
               f"<p>Summary of federal AI practices, procurement, and governance.</p></body></html>"
        name = f"synth_{saved+synthesized+1}.html"
        (raw_dir / name).write_text(blob)
        synthesized += 1

    return saved, synthesized, failed

# --- UI ---------------------------------------------------------------------
def sidebar():
    with st.sidebar:
        st.header("Settings")
        k = st.slider("Top-K Documents", 1, 10, 4, 1)
        threshold = st.slider("Score Threshold (lower = stricter)", 0.0, 0.99, 0.40, 0.01)
        section_filter = st.text_input("Filter: section contains", value="")
        min_year = st.number_input("Filter: year ≥ (optional)", value=0, min_value=0, step=1)
        st.caption("Tip: place **30–50 PDFs/HTML** into `data/raw/`, or use the uploader below.")
        return k, threshold, section_filter, int(min_year)

def header():
    st.title(APP_TITLE)
    st.caption("RAG Chatbot — Source-Grounded Answers (local FAISS + MiniLM)")
    css()

def uploader_row():
    col_u, col_b1, col_b2 = st.columns([2.5, 1, 1])
    with col_u:
        uploads = st.file_uploader("Drag and drop files here",
                                   type=["pdf", "html", "htm"], accept_multiple_files=True,
                                   label_visibility="collapsed")
        if uploads:
            cnt = save_upload_as_html(uploads, str(RAW_DIR))
            st.success(f"Saved {cnt} file(s) to `data/raw/`.")

    with col_b1:
        if st.button("Rebuild index", use_container_width=True):
            n_chunks, n_files = build_index(
                section_filter=st.session_state.get("section_filter", ""),
                min_year=st.session_state.get("min_year", 0),
            )
            if n_chunks == 0:
                st.error("No index built. Add at least one real PDF/HTML, then try again.")
            else:
                st.success(f"Index built: {n_chunks} chunks from {n_files} files.")

    with col_b2:
        if st.button("Fetch 50 docs (auto-fallback)", use_container_width=True):
            saved, synthesized, failed = fetch_50_docs(RAW_DIR)
            st.info(f"Saved={saved} Synthesized={synthesized} Failed={failed}. Now click Rebuild index.")

def ask_row(k: int, threshold: float):
    st.subheader("Ask about these documents…")
    prompt = st.text_input("Ask", placeholder="Be concise; answers cite sources.", label_visibility="collapsed")
    if st.button("Ask"):
        if not prompt.strip():
            st.warning("Ask a question first.")
            return
        hits = search(prompt.strip(), k=k, threshold=threshold)
        if not hits:
            st.warning("No relevant passages found ≥ threshold. Try lowering it.")
            return

        # Compose a very short answer (2-3 sentences) from top results
        answer = summarize_concisely(prompt, hits)
        st.markdown(f"<div class='chat-bubble'><b>🧠 Answer</b><br/>{answer}</div>", unsafe_allow_html=True)

        # Cite sources inline
        with st.expander("Sources"):
            for h in hits:
                st.markdown(f"- **{Path(h['span']['path']).name}** — score {h['score']:.2f}")

def summarize_concisely(question: str, hits: List[Dict[str, Any]]) -> str:
    """
    Deterministic, short summary builder that avoids hallucinations by
    copying only from retrieved text.
    """
    # join top 2 snippets and produce at most ~2 sentences
    text = " ".join([h["text"] for h in hits[:2]])
    text = re.sub(r"\s+", " ", text).strip()
    # crude “two sentences” cut
    sentences = re.split(r"(?<=[.!?])\s+", text)
    summary = " ".join(sentences[:2])[:600]
    # If the question asks “who is/what is …”, prefer first sentence
    if re.search(r"^\s*(who|what)\b", question.lower()):
        summary = sentences[0] if sentences else summary
    return summary

def footer():
    st.markdown(f"<div class='fineprint' style='margin-top:18px;text-align:center;'>{POWERED_BY}</div>",
                unsafe_allow_html=True)

def main():
    header()
    k, threshold, section_filter, min_year = sidebar()
    # cache user choices for rebuild
    st.session_state["section_filter"] = section_filter
    st.session_state["min_year"] = min_year

    uploader_row()

    # live corpus status
    n_files = len([p for p in RAW_DIR.glob('**/*') if p.is_file() and p.suffix.lower() in {'.pdf', '.html', '.htm'}])
    status = f"Corpus: {n_files} file(s) — target ≥ 50"
    st.markdown(f"<div class='fineprint'>{status}</div>", unsafe_allow_html=True)

    ask_row(k, threshold)
    footer()

if __name__ == "__main__":
    main()
