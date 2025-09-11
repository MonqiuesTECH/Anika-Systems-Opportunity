Anika Systems RAG Chatbot

Retrieval-Augmented Generation Demo – Built with Streamlit, FAISS, and SentenceTransformers

Overview

This project demonstrates a production-ready retrieval-augmented generation (RAG) pipeline designed for Anika Systems’ AI Engineer assignment. The application ingests 30–50+ documents (PDF/HTML), chunks and embeds them using a transformer model, stores vectors in FAISS, and serves a black/blue-branded Streamlit chatbot that answers queries with concise, grounded responses.

Beyond the requirements, the implementation adds:

Corpus auto-fetch: Pulls 50 public or synthetic documents to guarantee scale.

Index progress visualization: Extraction and embedding tracked via Streamlit progress bar.

Metrics after each query: Top-1 score, average score, and chunk count to quantify retrieval quality.

Explainability previews: 100-character snippets of each cited chunk.

UX polish: Black/blue theme, sidebar controls, auto-scroll, and toggle for expander behavior.

Branding: Footer “Powered by Monique Bruce” for authorship clarity.

This repo shows how to deliver a lean but complete RAG product, not just a prototype.

Architecture

Components

Data ingestion: PDFs (via pypdf) and HTML (via BeautifulSoup), normalized into plain text.

Chunking: Sliding window (1200 chars with 200 overlap) for context preservation.

Embeddings: sentence-transformers/all-MiniLM-L6-v2 for efficient semantic vectors.

Indexing: FAISS flat index with cosine similarity (normalized dot-product).

Search: Top-K retrieval with adjustable score threshold.

Answer generation: Concise 2–3 sentence extractive summaries from top chunks.

UI/UX: Streamlit app with metrics, previews, expander toggle, and custom styling.

Setup
Prerequisites

Python 3.10+ (tested on 3.13)

pip or uv environment manager

Installation

git clone <repo-url>
cd anika-systems-rag
pip install -r requirements.txt


Run the app

streamlit run app.py

The app will launch at [localhost:8501.](https://anika-systems-opportunity-i4p93gcvm3ta5n7jfoy587.streamlit.app/)

Usage

Load documents

Place PDFs/HTML in data/raw/, or

Click “Fetch 50 docs” in the sidebar for automatic corpus creation.

Build the index

Click “Rebuild index”. Progress bar shows extraction → chunking → embedding.

Ask questions

Type a query into the chat box.

Get concise, grounded answers with cited sources and previews.

Adjust retrieval

Use sidebar to tune Top-K (1–10) and Score Threshold (0.0–1.0).

Features

Concise grounded answers
Ensures responses are limited to a few sentences drawn from retrieved chunks.

Retrieval quality metrics
Displays Top-1 score, average score, and total indexed chunks for transparency.

Explainability
Provides previews (first 100 chars) from each cited chunk to reinforce grounding.

Corpus auto-scale
Guarantees 50+ docs by combining fetch + synthetic fillers if needed.

Professional UX
Black/blue theme, branded footer, sidebar expander toggle, and auto-scroll on query.

Example Workflow

Fetch 50 documents (Anika Systems site + AI governance references).

Rebuild the index (extract, embed, store).

Query: “Summarize Anika Systems’ core capabilities.”

Response: concise 2–3 sentences with 2–3 cited sources and previews.

Metrics confirm retrieval quality (e.g., Top-1 score 0.81, Avg@4 score 0.67, 520 chunks indexed).

Tech Stack

Python: 3.13

Framework: Streamlit

Vector Store: FAISS

Embeddings: SentenceTransformers (MiniLM-L6-v2)

Parsing: PyPDF2, BeautifulSoup4

HTTP: Requests

Data handling: JSON, pathlib

Compliance & Extension

Security: No external APIs required; all embeddings run locally.

Scaling: Easily extendable to cloud deployment (AWS/GCP/Azure) with S3/GCS for raw docs.

Explainability: Designed with transparency (sources, previews, metrics) for enterprise trust.

Extensibility: Swap FAISS for Pinecone/Weaviate, or MiniLM for OpenAI/Anthropic embeddings.

Author

Monique Bruce – Software Engineer & AI Architect

Experienced in designing and implementing AI-driven systems that streamline operations and reduce manual effort across healthcare, finance, aerospace, and e-commerce.

Provides technical guidance to startups and growth-stage companies on building scalable AI infrastructure and preparing products for investor and enterprise adoption.

Over six years of hands-on experience in software engineering, cloud infrastructure, and automation, with a strong track record of delivering end-to-end production solutions.
