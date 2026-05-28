import os
import fitz  # PyMuPDF
import spacy
from dotenv import load_dotenv
from neo4j import GraphDatabase
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ── Connections ──────────────────────────────────────────────────
neo4j_driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
)

chroma_client = chromadb.PersistentClient(path="./chroma_db")
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
collection = chroma_client.get_or_create_collection(
    name="automotive_chunks",
    embedding_function=embed_fn
)

nlp = spacy.load("en_core_web_sm")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " "]
)

# ── Step 1: Parse PDF ────────────────────────────────────────────
def parse_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if text:
            pages.append({"page": i + 1, "text": text})
    print(f"✅ Parsed {len(pages)} pages")
    return pages

# ── Step 2: Build Neo4j Hierarchy ────────────────────────────────
def build_graph(pages, doc_title="Automotive PDF"):
    with neo4j_driver.session() as session:
        # Clear old data
        session.run("MATCH (n) DETACH DELETE n")

        # Document node
        session.run(
            "CREATE (d:Document {id: 'doc1', title: $title})",
            title=doc_title
        )

        all_chunks = []

        for page in pages:
            page_id = f"page_{page['page']}"

            # Section node (one per page)
            session.run("""
                CREATE (s:Section {id: $sid, page: $page})
                WITH s
                MATCH (d:Document {id: 'doc1'})
                CREATE (d)-[:HAS_SECTION]->(s)
            """, sid=page_id, page=page['page'])

            # Chunk nodes
            chunks = splitter.split_text(page['text'])
            for j, chunk_text in enumerate(chunks):
                chunk_id = f"{page_id}_chunk_{j}"
                session.run("""
                    CREATE (c:Chunk {id: $cid, text: $text, page: $page})
                    WITH c
                    MATCH (s:Section {id: $sid})
                    CREATE (s)-[:HAS_CHUNK]->(c)
                """, cid=chunk_id, text=chunk_text,
                     page=page['page'], sid=page_id)

                all_chunks.append({
                    "id": chunk_id,
                    "text": chunk_text,
                    "page": page['page']
                })

        print(f"✅ Graph built: {len(all_chunks)} chunks across {len(pages)} sections")
        return all_chunks

# ── Step 3: Extract Entities & Link ──────────────────────────────
def extract_entities(chunks):
    with neo4j_driver.session() as session:
        for chunk in chunks:
            doc = nlp(chunk['text'])
            for ent in doc.ents:
                if len(ent.text.strip()) < 2:
                    continue
                session.run("""
                    MERGE (e:Entity {name: $name, type: $type})
                    WITH e
                    MATCH (c:Chunk {id: $cid})
                    MERGE (c)-[:MENTIONS]->(e)
                """, name=ent.text.strip(),
                     type=ent.label_,
                     cid=chunk['id'])
    print("✅ Entities extracted and linked")

# ── Step 4: Store Embeddings in ChromaDB ─────────────────────────
def store_embeddings(chunks):
    ids = [c['id'] for c in chunks]
    texts = [c['text'] for c in chunks]
    metadatas = [{"page": c['page']} for c in chunks]

    collection.add(ids=ids, documents=texts, metadatas=metadatas)
    print(f"✅ Embeddings stored: {len(chunks)} vectors")

# ── Run Everything ────────────────────────────────────────────────
if __name__ == "__main__":
    PDF_PATH = "./data/book.pdf"  # ← change to your exact PDF filename

    print("\n🚀 Starting ingestion pipeline...\n")
    pages = parse_pdf(PDF_PATH)
    chunks = build_graph(pages)
    extract_entities(chunks)
    store_embeddings(chunks)
    print("\n🎉 Ingestion complete! Your knowledge graph is ready.\n")
