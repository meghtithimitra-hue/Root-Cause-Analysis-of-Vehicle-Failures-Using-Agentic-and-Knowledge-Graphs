# 🚗 Root Cause Analysis of Vehicle Failures Using Agentic AI and Knowledge Graphs

This project is an AI-powered vehicle diagnostic system built to understand automotive failures using Knowledge Graphs, GraphRAG, and LLM-based reasoning. Instead of relying only on keyword matching, the system tries to understand relationships between vehicle systems, components, symptoms, and failures to generate smarter diagnostics.

The idea behind this project was to convert automotive maintenance knowledge from technical PDFs into a structured graph that an AI model can reason over.

---

## ✨ What This Project Does

- Extracts information from automotive maintenance PDFs
- Identifies important vehicle entities and relationships using NLP
- Builds a hierarchical automotive knowledge graph
- Visualizes the graph interactively
- Uses GraphRAG retrieval to fetch relevant relationships
- Uses Llama3 through Ollama for AI-powered diagnostics
- Provides an interactive Streamlit web application

---

## 🧠 Technologies Used

- Python
- Streamlit
- spaCy
- NetworkX
- PyVis
- LangChain
- Ollama
- Llama3
- GraphRAG Architecture

---

## 📂 Project Structure

```bash
vehicle-graph-ai/
│
├── data/
│   └── automotive.pdf
│
├── graph/
│   ├── full_text.txt
│   ├── entities.json
│   ├── triples.json
│   └── hierarchical_graph.html
│
├── scripts/
│   ├── extract_text.py
│   ├── extract_entities.py
│   ├── build_graph.py
│   ├── reasoning.py
│   └── graph_rag.py
│
├── app.py
├── requirements.txt
└── README.md
```

---

## ⚙️ How the System Works

### 1. PDF Text Extraction
The automotive PDF is processed and converted into raw text.

### 2. NLP-Based Entity Extraction
spaCy is used to identify:
- vehicle systems
- components
- symptoms
- failures
- relationships

### 3. Triple Generation
Extracted knowledge is converted into graph triples:

```text
coolant leak → causes → engine overheating
```

### 4. Hierarchical Knowledge Graph Creation
The triples are transformed into an automotive knowledge graph showing relationships between systems and failures.

### 5. GraphRAG Retrieval
When the user enters a vehicle issue, the system retrieves relevant graph relationships.

### 6. LLM-Based Diagnostic Reasoning
The retrieved graph context is passed to Llama3 via Ollama to generate diagnostic reasoning and recommendations.

### 7. Streamlit Interface
All outputs are displayed through an interactive Streamlit application.

---

## 🚀 Installation

### Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/Root-Cause-Analysis-of-Vehicle-Failures-Using-Agentic-and-Knowledge-Graphs.git
```

### Create Virtual Environment

```bash
python -m venv .venv
```

### Activate Environment

#### Windows

```bash
.venv\Scripts\activate
```

#### Mac/Linux

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📦 Install spaCy Model

```bash
python -m spacy download en_core_web_sm
```

---

## 🤖 Install Ollama + Llama3

Download Ollama:

https://ollama.com/download

Pull Llama3:

```bash
ollama pull llama3
```

Run Llama3:

```bash
ollama run llama3
```

---

## 🏗️ Build the Knowledge Graph

### Extract PDF Text

```bash
python scripts/extract_text.py
```

### Extract Entities and Relationships

```bash
python scripts/extract_entities.py
```

### Build Hierarchical Graph

```bash
python scripts/build_graph.py
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

Open in browser:

```text
http://localhost:8501
```

---

## 💡 Example Queries

```text
2018 Honda Civic engine overheating and coolant leak
```

```text
Toyota Corolla brake vibration while stopping
```

```text
engine overheating after long drive
```

---

## 🔮 Future Improvements

- Semantic retrieval using embeddings
- ChromaDB integration
- Neo4j graph database support
- Multi-hop graph reasoning
- Confidence scoring
- Real-time graph highlighting
- Advanced Agentic AI workflows
