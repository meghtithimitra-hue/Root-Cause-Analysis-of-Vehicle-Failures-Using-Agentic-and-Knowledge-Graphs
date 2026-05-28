# Automotive Knowledge Graph Q&A App

## Overview
This project is an Automotive Document Question Answering application built using a Hierarchical Knowledge Graph + Retrieval-Augmented Generation (RAG) architecture.

The system parses automotive PDF documents, structures the extracted information into a 3-level hierarchical knowledge graph:

Document → Section → Chunk

Instead of relying only on flat vector similarity retrieval, the application leverages graph relationships to retrieve richer contextual information from parent nodes, improving response quality and relevance.

---

## Key Features

- PDF document ingestion
- Hierarchical Knowledge Graph construction
- Semantic vector search using embeddings
- Context-aware question answering
- Page-level citation support
- Interactive frontend with Streamlit
- FastAPI backend for API access
- Groq-powered LLM inference (LLaMA3-70B)

---

## Architecture

### Workflow

1. PDF document ingestion
2. Text extraction and chunking
3. Hierarchical graph construction:
   - Document node
   - Section nodes
   - Chunk nodes
4. Embedding generation and storage in ChromaDB
5. Query processing:
   - Semantic similarity retrieval
   - Graph context expansion
   - LLM answer generation
6. Response with citations

---

## Tech Stack

- Python
- Streamlit
- FastAPI
- Neo4j
- ChromaDB
- Groq API
- LLaMA3-70B
- PyMuPDF / PDF parsing libraries

---

## Project Structure

```bash
automotive-kg-app/
│
├── api.py              # FastAPI backend
├── app.py              # Streamlit frontend
├── ingest.py           # PDF ingestion + KG construction
├── query.py            # Retrieval + QA logic
├── requirements.txt
├── .gitignore
├── README.md
└── data/
```

---

## Installation

Clone the repository:

```bash
git clone <repo-url>
```

Move into the project folder:

```bash
cd automotive-kg-app
```

Create virtual environment:

### Windows
```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / Mac
```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
```

---

## Running the Project

### 1. Add your PDF
Place your automotive PDF inside:

```bash
data/
```

---

### 2. Run ingestion pipeline

```bash
python ingest.py
```

This will:
- parse the PDF
- create graph hierarchy
- generate embeddings

---

### 3. Start backend

```bash
uvicorn api:app --reload
```

---

### 4. Start frontend

```bash
streamlit run app.py
```

---

## Why Hierarchical KG instead of Flat RAG?

Traditional RAG retrieves isolated chunks based only on vector similarity.

This system improves retrieval by:
- linking chunks to parent sections
- preserving document hierarchy
- enabling context expansion through graph traversal

Result:
better contextual understanding and richer answers.

---

## Future Improvements

- multi-document support
- hybrid keyword + semantic retrieval
- agentic query planning
- cloud deployment
- user authentication
- evaluation benchmark pipeline

---

## Author

Meghtithi Mitra
