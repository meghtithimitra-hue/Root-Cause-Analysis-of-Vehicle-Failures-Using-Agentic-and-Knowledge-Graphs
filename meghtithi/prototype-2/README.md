# Root Cause Analysis of Vehicle Failures Using Agentic AI and Knowledge Graphs
### Prototype 2 — Automotive Failure Predictor

---

## Overview

This prototype is the next evolution of the automotive document Q&A system. It eliminates chunking entirely and replaces it with a **pure Knowledge Graph** approach — where Groq's LLaMA3-70B reads every paragraph and extracts structured **entities and relationships** which are stored directly in Neo4j.

Queries are answered through **direct graph traversal** — no vector search, no ChromaDB — making the system more semantically aware and capable of answering structural questions like conclusions and summaries that Prototype 1 struggled with.

---

## What Changed From Prototype 1

| | Prototype 1 | Prototype 2 |
|---|---|---|
| Text storage | Chunks (500-1000 chars) | Paragraphs → Entities + Relationships |
| Retrieval | ChromaDB vector search | Neo4j graph traversal |
| Entity linking | spaCy (surface level) | Groq LLM (semantic, relationship-aware) |
| Conclusion queries | ❌ Often fails | ✅ Falls back to document structure |
| Context loss | Some (chunking boundaries) | Minimal (relationships are explicit) |
| Ingestion speed | Fast (seconds) | Slower (LLM reads every paragraph) |
| Dependencies | ChromaDB + Neo4j | Neo4j only |

---

## What It Does

- Accepts any automotive PDF document via a web interface
- Parses the document into **paragraphs** (not fixed-size chunks)
- Sends every paragraph to **Groq LLaMA3-70B** which extracts:
  - Named entities (systems, metrics, organizations, values, people)
  - Semantic relationships between entities (e.g. `JUPITER --[USES_SENSOR]--> LiDAR`)
- Stores the full knowledge graph in **Neo4j** with the structure: `Document → Paragraph → Entity --[RELATIONSHIP]--> Entity`
- When a query is asked, extracts keywords from the question and **traverses the graph** to find matching entities, their relationships, and connected paragraphs
- Falls back to document structure (last pages) for broad summarization questions
- Sends the graph-enriched context to Groq LLaMA3 to generate a precise answer

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM (extraction + answering) | Groq LLaMA3-70B (llama-3.3-70b-versatile) |
| Knowledge Graph | Neo4j Desktop (local) |
| Vector Store | None (pure graph) |
| Entity/Relation Extraction | Groq LLaMA3-70B |
| Keyword Extraction | spaCy (en_core_web_sm) |
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
Paragraph Splitter
  (split by \n\n, min 50 chars)
     ↓
For each paragraph → Groq LLaMA3 extracts:
  {
    entities: [{name, type}, ...],
    relationships: [{from, relation, to}, ...]
  }
     ↓
┌──────────────────────────────────────────────┐
│           Neo4j Pure Knowledge Graph          │
│                                              │
│  Document                                   │
│     └──[HAS_PARAGRAPH]──> Paragraph          │
│                              └──[CONTAINS_ENTITY]──> Entity │
│                                                      ↕      │
│                              Entity ──[RELATION]──> Entity  │
└──────────────────────────────────────────────┘
     ↓
Query Pipeline:
  1. Extract keywords from question (spaCy)
  2. Match entities in Neo4j (fuzzy name match)
  3. Hop to related entities + source paragraphs
  4. If no match → fall back to last N paragraphs
  5. Send graph context to Groq LLaMA3
  6. Return answer with page citations
```

---

## Project Structure

```
automotive-kg-graph/
├── .env               ← API keys and DB credentials (not in repo)
├── requirements.txt
├── ingest.py          ← paragraph parsing + LLM-based graph extraction
├── query.py           ← pure graph traversal + LLM answering
├── api.py             ← FastAPI backend
├── app.py             ← Streamlit frontend
└── data/              ← PDF files go here
```

---

## How to Run Locally

**Prerequisites:** Python 3.10+, Neo4j Desktop installed and running

```bash
# 1. Activate virtual environment
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 2. Start Neo4j Desktop and click Start on your database

# 3. Run ingestion — NOTE: this is slower than Prototype 1
#    Groq reads every paragraph to extract the graph
python ingest.py

# 4. Start the backend (new terminal)
uvicorn api:app --reload --port 8000

# 5. Start the frontend (new terminal)
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## Important Note on Ingestion Speed

Because Groq's LLM reads every paragraph during ingestion, the process is slower than Prototype 1:

| Document size | Estimated ingestion time |
|---|---|
| 8-page research paper | 3 - 5 minutes |
| 50-page report | 20 - 30 minutes |
| 200+ page book | Not recommended on free tier (rate limits) |

The `time.sleep(2)` between paragraphs in `ingest.py` is intentional — it respects Groq's free tier limit of ~30 requests per minute.

---

## Sample Queries That Work Well

- *"What sensors does the JUPITER vehicle platform use?"*
- *"What is the relationship between THW and lane change criticality?"*
- *"What were the main conclusions of the paper?"*
- *"How are left and right lane changes different in terms of criticality?"*
- *"What is the AVEAS project and what does it aim to do?"*
- *"What is the cc1 parameter and how does it affect THW?"*

---

## Why Pure Graph Over Chunking

In Prototype 1, chunking caused two core problems:

**Problem 1 — Context loss at boundaries**
A sentence starting on chunk 5 and ending on chunk 6 would be split. The model would never see the complete thought.

**Problem 2 — Implicit relationships**
The fact that *"THW is critical below 0.9 seconds"* and *"left lane changes have high THW criticality"* are related was never captured. They were just two separate text pieces.

In Prototype 2, these are explicit graph edges:
```
THW --[CRITICAL_BELOW]--> 0.9 seconds
Left lane change --[HAS_HIGHER]--> THW criticality
```

The graph understands meaning, not just stores text.

---

## Known Limitations

**1. Groq rate limits during ingestion**
The free tier allows ~30 requests per minute. Large documents will be slow and may occasionally hit limits. The `time.sleep(2)` handles this for small documents.

**2. LLM extraction quality**
Entity and relationship extraction depends on how well the LLM reads each paragraph. Occasionally it may miss subtle relationships or misclassify entity types. Quality improves with more capable models.

**3. Keyword-based graph entry**
The query engine enters the graph by matching keywords from the question to entity names. If the user's phrasing doesn't match how the LLM named an entity during extraction, retrieval may miss relevant nodes.

**4. No vector fallback**
Unlike Prototype 1, there is no vector similarity fallback. If graph traversal finds nothing, the system falls back to the last few pages of the document — which works for conclusions but may not be ideal for all query types.

---

## Relationship to Microsoft GraphRAG

This prototype independently implements a core idea from **Microsoft's GraphRAG** paper (2024) — that LLM-extracted entity relationship graphs provide richer retrieval than pure vector search. The key difference is that GraphRAG uses three layers (graph + chunks + document summary) while this prototype uses graph only. A future version combining all three layers would represent the full GraphRAG architecture.

---

## Comparison with Prototype 1

| Capability | Prototype 1 | Prototype 2 |
|---|---|---|
| Specific fact queries | ✅ Excellent | ✅ Excellent |
| Technical metric queries | ✅ Good | ✅ Good |
| Cross-document entity linking | ⚠️ Partial | ✅ Full |
| Conclusion / summary queries | ❌ Fails | ✅ Improved |
| Ingestion speed | ✅ Fast | ⚠️ Slower |
| Large document support | ⚠️ Limited | ❌ Not recommended |
| Dependencies | More (ChromaDB) | Simpler (Neo4j only) |

---

## Future Improvements

- Combine graph traversal with vector search for hybrid retrieval (full GraphRAG)
- Add a dedicated summarization endpoint that reads full document structure
- Improve entity matching with embedding-based fuzzy search instead of string matching
- Add graph visualization in the frontend using Neo4j Bloom or D3.js
- Support multi-document graphs (connect entities across multiple PDFs)
