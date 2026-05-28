# Root Cause Analysis of Vehicle Failures Using Agentic AI and Knowledge Graphs
### Prototype 1 — 🚗 Automotive Document Q&A

---

## Overview

This prototype is a **PDF-based Question Answering system** built specifically for automotive engineering documents. It combines a **Hierarchical Knowledge Graph** (Neo4j) with **vector similarity search** (ChromaDB) to answer technical queries from uploaded PDF documents with page-level citations.

The system was developed and tested on a research paper focused on highway lane change criticality analysis for autonomous vehicles and ADAS systems.

---

## What It Does

- Accepts any automotive PDF document via a web interface
- Parses the document and builds a **3-level hierarchical knowledge graph**: `Document → Section (Page) → Chunk → Entity`
- Stores vector embeddings of text chunks in ChromaDB for semantic similarity search
- When a query is asked, it retrieves the most relevant chunks via vector search and then **climbs the knowledge graph** to enrich the context with section and entity information
- Sends the enriched context to **Groq's LLaMA3-70B** to generate a precise, cited answer
- Displays the answer along with source page numbers in a clean chat interface

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Groq LLaMA3-70B (llama-3.3-70b-versatile) |
| Knowledge Graph | Neo4j Desktop (local) |
| Vector Store | ChromaDB (persistent, local) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Entity Extraction | spaCy (en_core_web_sm) |
| Backend | FastAPI |
| Frontend | Streamlit |
| PDF Parsing | PyMuPDF (fitz) |

---

## Architecture

```
PDF Document
     ↓
PDF Parser (PyMuPDF)
     ↓
Text Chunker (RecursiveCharacterTextSplitter)
  chunk_size = 1000 | chunk_overlap = 100
     ↓
┌─────────────────────────────────┐
│     Neo4j Knowledge Graph       │
│  Document → Section → Chunk     │
│              ↓                  │
│           Entity                │
│    (spaCy named entities)       │
└─────────────────────────────────┘
     ↓
ChromaDB (vector embeddings per chunk)
     ↓
Query Pipeline:
  1. Embed question → vector search in ChromaDB
  2. Retrieve top-k chunks
  3. Traverse Neo4j graph upward for context
  4. Send enriched context to Groq LLaMA3
  5. Return answer with page citations
```

---

## Project Structure

```
automotive-kg-app/
├── .env               ← API keys and DB credentials (not in repo)
├── requirements.txt
├── ingest.py          ← PDF parsing, graph construction, embedding
├── query.py           ← vector search + graph traversal + LLM call
├── api.py             ← FastAPI backend
├── app.py             ← Streamlit frontend
├── data/              ← PDF files go here
└── chroma_db/         ← auto-created vector store
```

---

## How to Run Locally

**Prerequisites:** Python 3.10+, Neo4j Desktop installed and running

```bash
# 1. Activate virtual environment
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 2. Start Neo4j Desktop and click Start on your database

# 3. Run ingestion (builds the knowledge graph)
python ingest.py

# 4. Start the backend (new terminal)
uvicorn api:app --reload --port 8000

# 5. Start the frontend (new terminal)
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## Known Limitations

**1. Broad summarization queries fail**
Questions like *"What are the main conclusions?"* often return incorrect or incomplete answers. This is because the entry point for retrieval is vector similarity search — and conclusion text is semantically similar to the body of the paper, causing the retrieval to pick wrong chunks.

**2. Chunking causes context loss**
Splitting text into fixed-size chunks (even with overlap) can break sentences and separate related information across different chunks. Relationships between ideas are not explicitly modeled.

**3. Knowledge graph is partially used**
The graph stores entity relationships across the document, but the query engine only uses the graph for context enrichment after vector search — not as the primary retrieval mechanism. This means the full power of the graph is not utilized.

**4. Rate limits on large documents**
The Groq free tier has request limits. Large documents (500+ pages) may be slow to ingest.

---

## Sample Queries That Work Well

- *"What sensors does the JUPITER vehicle platform use?"*
- *"What is Time Headway (THW) and what is its critical threshold?"*
- *"Are left lane changes more critical than right lane changes?"*
- *"What is the AVEAS project?"*
- *"What data acquisition methods were used?"*

---

## Why This Approach

Standard RAG (Retrieval Augmented Generation) retrieves flat chunks of text without any structural awareness. The hierarchical knowledge graph adds:

- **Structural context** — every chunk knows which page and section it came from
- **Entity linking** — the same entity (e.g. "LiDAR") mentioned on different pages is connected as a single node
- **Richer answers** — the LLM receives not just text but also page numbers and related entities

This makes answers more accurate and traceable compared to plain vector search.

---

## Future Improvements (addressed in Prototype 2)

- Replace chunking with pure graph-based entity and relationship extraction
- Implement direct graph traversal queries for structural questions
- Add a dedicated summarization pipeline for conclusion-type queries
- Move toward the Microsoft GraphRAG approach (graph + chunks + full document summary)

