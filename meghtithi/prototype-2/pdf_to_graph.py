import fitz  # PyMuPDF
import json
import os
from groq import Groq
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
)

# ── Extraction prompt ──────────────────────────────────────────────────────
EXTRACTION_PROMPT = """You are an automotive knowledge graph builder.

Read the ENTIRE text below from an automotive research paper.
Your job is to extract failure knowledge as structured triples.

CRITICAL RULES:
- Read the full text holistically before extracting — do NOT process sentence by sentence
- Preserve complete causal chains: what symptom → why it fails → which part → what to do
- If a paragraph says "cyclical loading causes crack in side plate which leads to clutch failure 
  diagnosed by slipping", extract the FULL chain — do not split it
- Infer the observable symptom even if not stated explicitly 
  (e.g. "clutch disc crack" → symptom is "clutch slipping")
- confidence: your certainty this is a real causal relationship (0.50 to 0.95)
- Return ONLY a valid JSON array. No explanation, no markdown, no backticks.

Output format:
[
  {
    "symptom": "observable symptom a driver or mechanic would notice",
    "failure_mode": "root technical cause of failure",
    "component": "specific part that fails",
    "repair_action": "what should be done to fix it",
    "confidence": 0.80,
    "causal_chain": "one sentence explaining the full cause-effect relationship"
  }
]
"""

# ── Page-aware extraction (no chunking) ───────────────────────────────────
def extract_pages(pdf_path: str) -> list[dict]:
    """
    Returns list of {page_num, text} dicts.
    Pages are the natural semantic unit in academic papers —
    each page is a complete thought, not an arbitrary chunk.
    """
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if len(text) > 100:  # skip blank/header-only pages
            pages.append({"page_num": i + 1, "text": text})
    doc.close()
    print(f"  📄 Extracted {len(pages)} content pages")
    return pages

def extract_full_text(pdf_path: str) -> str:
    """Full document text — used for short PDFs that fit in context."""
    doc = fitz.open(pdf_path)
    text = "\n\n".join(page.get_text() for page in doc)
    doc.close()
    return text

# ── LLM triple extraction ─────────────────────────────────────────────────
# Groq llama-3.3-70b context window = ~128k tokens
# A dense 7-page paper ≈ 6,000–8,000 tokens → fits in ONE call
# Threshold: if full text < 80,000 chars (~20k tokens), send whole doc at once
SINGLE_CALL_CHAR_LIMIT = 80_000

def call_llm_for_triples(text: str, label: str) -> list:
    """Send text to LLM and return extracted triples."""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": EXTRACTION_PROMPT},
                {"role": "user",   "content": f"Extract all failure triples from this text:\n\n{text}"}
            ],
            temperature=0.1,   # deterministic extraction
            max_tokens=4000
        )

        raw = response.choices[0].message.content.strip()

        # Clean up any accidental markdown wrapping
        raw = raw.replace("```json", "").replace("```", "").strip()
        if not raw.startswith("["):
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start != -1:
                raw = raw[start:end]

        triples = json.loads(raw)
        print(f"    ✅ {label}: {len(triples)} triples extracted")
        return triples

    except json.JSONDecodeError as e:
        print(f"    ⚠️  JSON parse error on {label}: {e}")
        print(f"    Raw output preview: {raw[:300]}")
        return []
    except Exception as e:
        print(f"    ⚠️  LLM error on {label}: {e}")
        return []


def extract_triples(pdf_path: str) -> list:
    """
    Strategy:
    - Short PDF  → one single LLM call with full document text
                   (best semantic preservation, full context)
    - Long PDF   → one LLM call per PAGE (page = natural semantic unit,
                   never an arbitrary character-count chunk)

    We NEVER split mid-sentence or mid-paragraph.
    """
    full_text = extract_full_text(pdf_path)
    total_chars = len(full_text)
    print(f"  📝 Total document size: {total_chars:,} characters")

    if total_chars <= SINGLE_CALL_CHAR_LIMIT:
        # ── Strategy A: whole document in one call ─────────────────────
        print("  🚀 Strategy: Single call (full document — best semantic fidelity)")
        triples = call_llm_for_triples(full_text, "full document")

    else:
        # ── Strategy B: page-by-page (natural semantic units) ──────────
        print("  📑 Strategy: Page-by-page (document too large for single call)")
        print("     Pages are natural semantic units — no mid-sentence splits")
        pages = extract_pages(pdf_path)
        all_triples = []

        for page in pages:
            label = f"page {page['page_num']}"
            page_triples = call_llm_for_triples(page["text"], label)
            all_triples.extend(page_triples)

        # Deduplication: same symptom+failure_mode = same triple
        seen = set()
        triples = []
        for t in all_triples:
            key = (t.get("symptom","").lower(), t.get("failure_mode","").lower())
            if key not in seen:
                seen.add(key)
                triples.append(t)
        print(f"  🔁 After dedup: {len(triples)} unique triples")

    return triples


# ── Neo4j insertion ───────────────────────────────────────────────────────
def insert_triples(triples: list, source_name: str):
    required_keys = {"symptom", "failure_mode", "component", "repair_action"}

    def _insert(tx, t, source):
        tx.run("""
            MERGE (s:Symptom {name: $symptom})
            MERGE (f:FailureMode {name: $failure_mode})
            MERGE (c:Component {name: $component})
            MERGE (r:RepairAction {name: $repair_action})
            SET f.causal_chain = $causal_chain
            MERGE (s)-[i:INDICATES]->(f)
              ON CREATE SET i.confidence = $confidence, i.source = $source
              ON MATCH  SET i.confidence = CASE
                              WHEN i.confidence < $confidence
                              THEN $confidence ELSE i.confidence END
            MERGE (f)-[:AFFECTS]->(c)
            MERGE (c)-[:REQUIRES]->(r)
        """,
            symptom=t["symptom"],
            failure_mode=t["failure_mode"],
            component=t["component"],
            repair_action=t["repair_action"],
            confidence=float(t.get("confidence", 0.70)),
            causal_chain=t.get("causal_chain", ""),
            source=source
        )

    inserted, skipped = 0, 0
    with driver.session() as session:
        for t in triples:
            if not required_keys.issubset(t.keys()):
                skipped += 1
                continue
            try:
                session.execute_write(_insert, t, source_name)
                inserted += 1
            except Exception as e:
                print(f"  ⚠️  Insert error: {e}")
                skipped += 1

    print(f"  ✅ Inserted: {inserted} | Skipped (incomplete): {skipped}")


# ── Main ──────────────────────────────────────────────────────────────────
def process_pdf(pdf_path: str):
    print(f"\n📄 Processing: {os.path.basename(pdf_path)}")
    print("  Architecture: NO chunking — pages as semantic units\n")

    triples = extract_triples(pdf_path)

    if not triples:
        print("  ⚠️  No triples extracted. Check that the PDF has readable text.")
        return

    print(f"\n  📊 Sample triples extracted:")
    for t in triples[:4]:
        print(f"    [{t.get('confidence', '?')}] {t.get('symptom')} "
              f"→ {t.get('failure_mode')} "
              f"→ {t.get('component')}")
        if t.get("causal_chain"):
            print(f"         chain: {t['causal_chain'][:80]}...")

    source_name = os.path.basename(pdf_path)
    print(f"\n  💾 Inserting into Neo4j...")
    insert_triples(triples, source_name)

    # Graph stats
    with driver.session() as session:
        result = session.run("""
            MATCH (n) RETURN labels(n)[0] as type, count(n) as count
            ORDER BY count DESC
        """)
        print("\n  📈 Graph node counts after ingestion:")
        for r in result:
            print(f"    {r['type']}: {r['count']}")


    print(f"\n✅ Done. Run visualize_graph.py to inspect the updated graph.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pdf_to_graph.py yourfile.pdf")
    else:
        process_pdf(sys.argv[1])
