import time
import pathlib
from typing import Optional

import streamlit as st
from dotenv import load_dotenv

from src.loaders import load_raw_documents
from src.clean_chunk import clean_and_chunk
from src.embed_index import ensure_faiss_index, IndexArtifacts
from src.retrieve import RagRetriever, AnswerWithCitations
from src.prompts import build_prompt
from src.metrics import TokenCounter, fmt_ms

# ---------------- App Config ----------------
st.set_page_config(page_title="RAG Chatbot Assignment", page_icon="💬", layout="wide")
load_dotenv()

DATA_DIR = pathlib.Path("data")
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"
for d in (RAW_DIR, PROC_DIR, INDEX_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------- Sidebar ----------------
st.sidebar.title("Settings")
top_k = st.sidebar.slider("Top-K Documents", min_value=2, max_value=10, value=4, step=1)
score_threshold = st.sidebar.slider("Score Threshold (lower = stricter)", 0.0, 1.0, 0.40, 0.05)
section_filter = st.sidebar.text_input("Filter: section contains", value="")
year_min = st.sidebar.number_input("Filter: year ≥ (optional)", min_value=0, value=0, step=1)
st.sidebar.markdown("---")
st.sidebar.caption("Place 30–50 PDFs/HTML into `data/raw/` or use the uploader below.")

# ---------------- Helpers ----------------
def _rebuild_index() -> Optional[IndexArtifacts]:
    """(Re)build the index if documents are present; return None if not."""
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

# cache handle so we can clear it when uploading
def _clear_bootstrap_cache():
    try:
        bootstrap_index.clear()  # type: ignore[attr-defined]
    except Exception:
        pass

artifacts = bootstrap_index()

# ---------------- Header ----------------
st.title("RAG Chatbot – Source-Grounded Answers")
st.caption("Local FAISS + Sentence Transformers + Streamlit. Inline citations, filters, and metrics included.")

# ---------------- Inline uploader (works even with no index) ----------------
st.markdown("#### Add or update documents")
uploaded = st.file_uploader("Drop PDF/HTML files", type=["pdf", "html", "htm"], accept_multiple_files=True)
col_upl, col_reb = st.columns([1, 1])

if uploaded:
    with col_upl:
        if st.button("Save uploads"):
            for f in uploaded:
                dest = RAW_DIR / f.name
                dest.write_bytes(f.getbuffer())
            st.success(f"Saved {len(uploaded)} file(s) to `data/raw/`.")
            _clear_bootstrap_cache()
            artifacts = bootstrap_index()

with col_reb:
    if st.button("Rebuild index"):
        _clear_bootstrap_cache()
        artifacts = bootstrap_index()
        if artifacts:
            st.success("Index rebuilt.")

# Always show index status (no st.stop)
if artifacts is None:
    st.info("No index yet. Add at least one real PDF or HTML and click **Save uploads** or **Rebuild index**.")
else:
    with st.expander("ℹ️ Data & Index Details", expanded=False):
        st.json({
            "num_chunks": len(artifacts.metadata),
            "embedding_model": artifacts.embedding_model_name,
            "index_path": str(INDEX_DIR.resolve()),
            "fields": ["text", "source", "title", "section", "year", "url"]
        })

# ---------------- Retriever (if available) ----------------
retriever = RagRetriever(artifacts, top_k_default=top_k) if artifacts else None

# ---------------- Chat UI (always rendered) ----------------
if "chat" not in st.session_state:
    st.session_state.chat = []

for turn in st.session_state.chat:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

prompt_placeholder = "Ask a question about your documents…" if retriever else "Upload documents and rebuild the index to enable Q&A…"
user_query = st.chat_input(prompt_placeholder)

if user_query:
    st.session_state.chat.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        if not retriever:
            msg = ("I’m ready once your index is built. "
                   "Use the uploader above, then click **Rebuild index**.")
            st.markdown(msg)
            st.session_state.chat.append({"role": "assistant", "content": msg})
        else:
            t0 = time.time()
            token_counter = TokenCounter()

            retrieved = retriever.search(
                query=user_query,
                top_k=top_k,
                score_threshold=score_threshold,
                section_contains=section_filter.strip() or None,
                year_min=int(year_min) if year_min else None,
            )

            if not retrieved:
                msg = (
                    "I couldn’t find a confident answer in the indexed documents. "
                    "Try rephrasing, lowering the score threshold, or removing filters."
                )
                st.markdown(msg)
                st.session_state.chat.append({"role": "assistant", "content": msg})
            else:
                system_prompt, final_context = build_prompt(user_query, retrieved)

                # Local draft synthesis (no external LLM call).
                synthesized_answer = "**Answer (draft):**\n\nBased on the top sources:\n\n"
                for ctx in final_context:
                    synthesized_answer += f"- {ctx['summary']} {ctx['inline_citation']}\n"
                synthesized_answer += "\n_This draft is synthesized from retrieved chunks shown below._"

                answer = AnswerWithCitations(
                    text=synthesized_answer,
                    citations=[c["inline_citation"] for c in final_context],
                    sources=final_context,
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

                st.session_state.chat.append({"role": "assistant", "content": synthesized_answer})
