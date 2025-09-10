import time
from pathlib import Path
from typing import Optional, List, Tuple

import streamlit as st
from dotenv import load_dotenv

# -------------------- Inline starter-corpus fetcher --------------------
# Works on Streamlit Cloud and locally; avoids shell scripts.
def _inline_fetch_starter_corpus(raw_root: Path = Path("data/raw")) -> int:
    import requests  # local import so it's optional unless used

    URLS: List[Tuple[str, str]] = [
        # ---- Anika Systems (core pages)
        ("https://www.anikasystems.com/",                           "anika/anika-home.html"),
        ("https://www.anikasystems.com/whoweare.html",              "anika/who-we-are.html"),
        ("https://www.anikasystems.com/capabilities.html",          "anika/capabilities.html"),
        ("https://www.anikasystems.com/contracts.html",             "anika/contracts.html"),
        ("https://www.anikasystems.com/insights.html",              "anika/insights.html"),
        # ---- Federal AI / policy docs
        ("https://www.whitehouse.gov/wp-content/uploads/2024/03/M-24-10-Advancing-Governance-Innovation-and-Risk-Management-for-Agency-Use-of-Artificial-Intelligence.pdf", "policy/OMB-M-24-10.pdf"),
        ("https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf",  "policy/NIST-AI-RMF-1.0.pdf"),
        ("https://www.govinfo.gov/content/pkg/DCPD-202300949/pdf/DCPD-202300949.pdf", "policy/EO-14110.pdf"),
        ("https://media.defense.gov/2024/Apr/15/2003439257/-1/-1/0/CSI-DEPLOYING-AI-SYSTEMS-SECURELY.PDF", "policy/CISA-Deploying-AI-Securely.pdf"),
        ("https://www.dhs.gov/sites/default/files/2024-04/24_0426_dhs_ai-ci-safety-security-guidelines-508c.pdf", "policy/DHS-AI-Guidelines.pdf"),
        # ---- GAO reports
        ("https://www.gao.gov/assets/gao-24-107332.pdf",           "gao/GAO-24-107332.pdf"),
        ("https://www.gao.gov/assets/gao-25-107653.pdf",           "gao/GAO-25-107653.pdf"),
        # ---- GSA AI resources
        ("https://coe.gsa.gov/coe/ai-guide-for-government/print-all/index.html", "gsa/GSA-AI-Guide.html"),
        ("https://coe.gsa.gov/docs/2020/AIServiceCatalog.pdf",     "gsa/GSA-AI-Service-Catalog.pdf"),
        # ---- Contract vehicles
        ("https://www.nitaac.nih.gov/gwacs/cio-sp3",               "vehicles/CIO-SP3.html"),
        ("https://itvmo.gsa.gov/it-vehicles/",                     "vehicles/GSA-IT-Vehicles.html"),
    ]

    raw_root.mkdir(parents=True, exist_ok=True)
    saved = 0
    for url, rel in URLS:
        out_path = raw_root / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            out_path.write_bytes(r.content)
            saved += 1
        except Exception as e:
            print(f"[fetch] skip {url}: {e}")
    return saved

# If you also created scripts/fetch_corpus.py, we’ll try it first:
try:
    from scripts.fetch_corpus import fetch_starter_corpus as _external_fetch
except Exception:
    _external_fetch = None

def fetch_starter_corpus(raw_root: Path = Path("data/raw")) -> int:
    if _external_fetch:
        try:
            return _external_fetch(raw_root)
        except Exception as e:
            print(f"[fetch] external fetcher failed, using inline fallback: {e}")
    return _inline_fetch_starter_corpus(raw_root)

# -------------------- RAG modules --------------------
from src.loaders import load_raw_documents
from src.clean_chunk import clean_and_chunk
from src.embed_index import ensure_faiss_index, IndexArtifacts
from src.retrieve import RagRetriever, AnswerWithCitations
from src.prompts import build_prompt
from src.metrics import TokenCounter, fmt_ms

# -------------------- App config & paths --------------------
st.set_page_config(page_title="RAG Chatbot Assignment", page_icon="💬", layout="wide")
load_dotenv()

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"
for d in (RAW_DIR, PROC_DIR, INDEX_DIR):
    d.mkdir(parents=True, exist_ok=True)

# -------------------- Sidebar --------------------
st.sidebar.title("Settings")
top_k = st.sidebar.slider("Top-K Documents", 2, 10, 4, 1)
score_threshold = st.sidebar.slider("Score Threshold (lower = stricter)", 0.0, 1.0, 0.40, 0.05)
section_filter = st.sidebar.text_input("Filter: section contains", value="")
year_min = st.sidebar.number_input("Filter: year ≥ (optional)", min_value=0, value=0, step=1)
st.sidebar.markdown("---")
st.sidebar.caption("Place PDFs/HTML into `data/raw/`, use the uploader, or click the fetch button below.")

# -------------------- Helpers --------------------
def _rebuild_index() -> Optional[IndexArtifacts]:
    docs = load_raw_documents(RAW_DIR)
    if not docs:
        return None
    chunks = clean_and_chunk(docs, save_dir=PROC_DIR)
    if not chunks:
        return None
    return ensure_faiss_index(chunks, index_dir=INDEX_DIR)

@st.cache_resource(show_spinner=True)
def bootstrap_index() -> Optional[IndexArtifacts]:
    return _rebuild_index()

def _clear_bootstrap_cache():
    try:
        bootstrap_index.clear()  # type: ignore[attr-defined]
    except Exception:
        pass

def _has_docs() -> bool:
    return any(RAW_DIR.rglob("*.pdf")) or any(RAW_DIR.rglob("*.htm*"))

# -------------------- Header --------------------
st.title("RAG Chatbot – Source-Grounded Answers")
st.caption("Local FAISS • Sentence Transformers • Streamlit. Inline citations, filters, metrics, and upload/fetch tools included.")

# -------------------- Cold-start self-heal --------------------
# If the server restarted and wiped runtime files, auto-populate so the app doesn't break.
if not _has_docs():
    with st.spinner("Cold start detected — fetching starter corpus…"):
        n_auto = fetch_starter_corpus(RAW_DIR)
    if n_auto > 0:
        st.success(f"Fetched {n_auto} starter files. Building index…")
        _clear_bootstrap_cache()
        artifacts = bootstrap_index()
    else:
        artifacts = None
        st.info("No documents fetched automatically. Use **Fetch starter corpus** or upload files, then click **Rebuild index**.")
else:
    artifacts = bootstrap_index()

# -------------------- Corpus tools --------------------
st.markdown("#### Add or update documents")

if st.button("🔽 Fetch starter corpus (Anika + federal AI)"):
    with st.spinner("Downloading starter corpus…"):
        n = fetch_starter_corpus(RAW_DIR)
    if n > 0:
        st.success(f"Fetched {n} files into `data/raw/`. Rebuilding index…")
        _clear_bootstrap_cache()
        artifacts = bootstrap_index()
    else:
        st.warning("No files were fetched. Try again, or use the uploader below.")

uploaded = st.file_uploader("Drop PDF/HTML files", type=["pdf", "html", "htm"], accept_multiple_files=True)
col_save, col_rebuild = st.columns([1, 1])
if uploaded and col_save.button("Save uploads"):
    saved = 0
    for f in uploaded:
        (RAW_DIR / f.name).write_bytes(f.getbuffer())
        saved += 1
    st.success(f"Saved {saved} file(s) to `data/raw/`. Rebuilding index…")
    _clear_bootstrap_cache()
    artifacts = bootstrap_index()

if col_rebuild.button("Rebuild index"):
    with st.spinner("Building embeddings and FAISS index…"):
        _clear_bootstrap_cache()
        artifacts = bootstrap_index()
    if artifacts:
        st.success("Index rebuilt.")

# -------------------- Index status --------------------
if artifacts is None:
    st.info("No index yet. Add/fetch documents and click **Rebuild index**.")
else:
    with st.expander("ℹ️ Data & Index Details", expanded=False):
        st.json({
            "num_chunks": len(artifacts.metadata),
            "embedding_model": artifacts.embedding_model_name,
            "index_path": str(INDEX_DIR.resolve()),
            "fields": ["text", "source", "title", "section", "year", "url"],
        })

# -------------------- Retriever --------------------
retriever = RagRetriever(artifacts, top_k_default=top_k) if artifacts else None

# -------------------- Chat UI (always on) --------------------
if "chat" not in st.session_state:
    st.session_state.chat = []

for turn in st.session_state.chat:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

placeholder = "Ask a question about your documents…" if retriever else "Fetch/upload docs and rebuild the index to enable Q&A…"
user_query = st.chat_input(placeholder)

if user_query:
    st.session_state.chat.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        if not retriever:
            msg = ("I’m ready once your index is built. "
                   "Use **Fetch starter corpus** or upload docs, then click **Rebuild index**.")
            st.markdown(msg)
            st.session_state.chat.append({"role": "assistant", "content": msg})
        else:
            t0 = time.time()
            token_counter = TokenCounter()

            results = retriever.search(
                query=user_query,
                top_k=top_k,
                score_threshold=score_threshold,
                section_contains=section_filter.strip() or None,
                year_min=int(year_min) if year_min else None,
            )

            if not results:
                msg = ("I couldn’t find a confident answer in the indexed documents. "
                       "Try rephrasing, lowering the score threshold to ~0.30, or clearing filters.")
                st.markdown(msg)
                st.session_state.chat.append({"role": "assistant", "content": msg})
            else:
                system_prompt, ctx = build_prompt(user_query, results)

                # Cost-free synthesis (swap for a real LLM if desired)
                answer_text = "**Answer (draft):**\n\nBased on the top sources:\n\n"
                for c in ctx:
                    answer_text += f"- {c['summary']} {c['inline_citation']}\n"
                answer_text += "\n_This draft is synthesized from retrieved chunks shown below._"

                answer = AnswerWithCitations(
                    text=answer_text,
                    citations=[c["inline_citation"] for c in ctx],
                    sources=ctx,
                )

                st.markdown(answer.text)
                with st.expander("🔎 Retrieved Chunks & Sources"):
                    for i, src in enumerate(answer.sources, start=1):
                        st.markdown(
                            f"**[{i}]** {src['title']} — {src['inline_citation']}\n\n"
                            f"`score={src['score']:.3f}` | `{src['source']}` | "
                            f"`{src.get('section','')}` | `{src.get('year','')}`\n\n"
                            f"> {src['preview']}\n\n"
                            f"{src.get('url','')}"
                        )

                elapsed_ms = int((time.time() - t0) * 1000)
                st.markdown("---")
                c1, c2, c3 = st.columns(3)
                c1.metric("Latency", fmt_ms(elapsed_ms))
                c2.metric("Prompt Tokens (est.)", token_counter.estimate_prompt_tokens(system_prompt))
                c3.metric("Chunks Retrieved", len(answer.sources))

                st.session_state.chat.append({"role": "assistant", "content": answer_text})

