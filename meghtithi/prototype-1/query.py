import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq

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
collection = chroma_client.get_collection(
    name="automotive_chunks",
    embedding_function=embed_fn
)

# Groq client — drop-in replacement for OpenAI client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def answer_query(question: str, top_k: int = 8) -> dict:

    # Step 1 — Vector search: find most relevant chunks
    results = collection.query(
        query_texts=[question],
        n_results=top_k
    )

    chunk_ids = results['ids'][0]
    raw_chunks = results['documents'][0]
    metadatas  = results['metadatas'][0]

    # Step 2 — Graph traversal: climb up to Section + get entities
    enriched = []
    with neo4j_driver.session() as session:
        for i, chunk_id in enumerate(chunk_ids):
            record = session.run("""
                MATCH (s:Section)-[:HAS_CHUNK]->(c:Chunk {id: $cid})
                MATCH (d:Document)-[:HAS_SECTION]->(s)
                OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity)
                RETURN d.title   AS doc,
                       s.page    AS page,
                       c.text    AS chunk,
                       collect(e.name) AS entities
            """, cid=chunk_id).single()

            if record:
                entity_str = ", ".join(record['entities']) if record['entities'] else "none"
                enriched.append(
                    f"[Page {record['page']} | Key terms: {entity_str}]\n{record['chunk']}"
                )
            else:
                enriched.append(
                    f"[Page {metadatas[i].get('page', '?')}]\n{raw_chunks[i]}"
                )

    # Step 3 — Send enriched context to Groq (llama3-70b)
    context = "\n\n---\n\n".join(enriched)

    system_prompt = """You are an expert automotive engineer assistant.
Answer questions strictly based on the provided context from an automotive document.
Always mention which page the information comes from.
If the answer is not in the context, say: 'I could not find that information in the document.'
Be precise and technical when needed."""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",   # updated Groq model
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"Context:\n\n{context}\n\nQuestion: {question}"}
        ],
        temperature=0.2,
        max_tokens=800
    )

    answer = response.choices[0].message.content
    source_pages = sorted(set(m.get('page') for m in metadatas))

    return {
        "answer": answer,
        "source_pages": source_pages,
        "chunks_used": len(enriched)
    }
