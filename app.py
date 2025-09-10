import os
import time
import json
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

# ---------- App Config ----------
st.set_page_config(page_title="RAG Chatbot Assignment", page_icon="💬", layout="wide")
load_dotenv()

DATA_DIR = pathlib.Path("data")
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"
for d in (RAW_DIR, PROC_DIR, INDEX_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------- Sidebar Controls ----------
st.sidebar.title("Settings")
top_k = st.sidebar.slider("Top-K Documents", min_value=2, max_value=10, value=4, step=1)
score_threshold = st.sidebar.slider("Score Threshold (lower = stricter)", 0.0, 1.0, 0.40, 0.05)
section_filter = st.sidebar.text_input("Filter: section contains", value="")
year_min = st.sidebar.number_input("Filter: year >= (optional)", min_value=0, value=0, step=1)
st.sidebar.markdown("---")
st.sidebar.caption("Place 30–50 PDFs/HTML into `data/raw/` and reload the app.")

# ---------- Bootstrap ----------
@st.cache_resource(show_spinner=True)
def bootstrap_index() -> Optional[IndexArtifacts]:
    docs = load_raw_documents(RAW_DIR)
    if not docs:
        return None

    chunks = clean_and_chunk(docs, save_dir=PROC_DIR)
    if not chunks:
        return None

    artifacts = ensure_faiss_index(chunks, index_dir=INDEX_DIR)
    return artifacts

artifacts = bootstrap_index()

st.title("RAG Chatbot – Source-Grounded Answers")
st.caption("Local FAISS + Sentence Transformers + Streamlit. Inline citations, filters, and metrics included.")

if artifacts is None:
    st.warning(
        "No index yet. Add at least one real PDF or HTML to `data/raw/` and reload. "
        "I couldn't build any chunks to embed."
    )
    st.stop()

retriever = RagRetriever(artifacts, top_k_default=top_k)

with st.expander("ℹ️ Data & Index Details"):
    st.json({
        "num_chunks": len(artifacts.metadata),
        "embedding_model": artifacts.embedding_model_name,
        "index_path": str(INDEX_DIR.resolve()),
        "fields": ["text", "source", "title", "section", "year", "url"]
    })

# ---------- Chat UI ----------
if "chat" not in st.session_state:
    st.session_state.chat = []

for turn in st.session_state.chat:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

user_query = st.chat_input("Ask a question about your documents…")
if user_query:
    st.session_state.chat.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        t0 = time.time()
        token_counter = TokenCounter()

        retrieved = retriever.search(
            query=user_query,
            top_k=top_k,
            score_threshold=score_threshold,
            section_contains=section_filter.strip() or None,
            year_min=int(year_min) if year_min else None
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

            # Local draft synthesis (no paid LLM call). Swap in your LLM here if desired.
            synthesized_answer = "**Answer (draft):**\n\nBased on the top sources:\n\n"
            for ctx in final_context:
                synthesized_answer += f"- {ctx['summary']} {ctx['inline_citation']}\n"
            synthesized_answer += "\n_This draft is synthesized from retrieved chunks shown below._"

            answer = AnswerWithCitations(
                text=synthesized_answer,
                citations=[c["inline_citation"] for c in final_context],
                sources=final_context
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
            st.markdown("
