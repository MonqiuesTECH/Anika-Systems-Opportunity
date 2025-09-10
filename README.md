# RAG Chatbot Assignment – Anika Systems

This repository contains my implementation of the **Data Wrangler + RAG Chatbot** assignment for the AI Engineer role at Anika Systems.

The solution demonstrates end-to-end **data mastery and retrieval-augmented generation (RAG)** using a local Streamlit app, FAISS vector database, and an LLM endpoint.

---

## 📌 Project Overview

**Goal:** Acquire, clean, enrich, embed, and index text documents, then surface their value through a chatbot that returns **source-grounded answers**.

**Features Implemented:**
- Data acquisition and preprocessing (30–50 text-heavy documents)
- Intelligent text chunking (by headings and sentences)
- Embedding generation using `sentence-transformers`
- Local vector search with **FAISS**
- Streamlit chatbot interface with:
  - Inline source citations
  - Adjustable `top_k` parameter
  - Optional metadata filters (e.g., year, section keyword)
- Observability:
  - Latency and token count logging
  - Error handling for empty or failed retrievals
- Demonstration of **5 Q&A examples** that clearly depend on source docs
- Graceful fallback response when no relevant results are found

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Streamlit** – interactive UI
- **FAISS** – vector database for similarity search
- **SentenceTransformers** – embedding generation
- **LangChain** – retrieval pipeline + prompt assembly
- **tiktoken** – token counting
- **dotenv** – environment variable management
- **Optional:** OpenAI API, local LLM (Ollama) or Vertex AI endpoint

---

## 📂 Repo Structure

