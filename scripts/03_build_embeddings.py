import json
import os
from collections import defaultdict

import chromadb
from sentence_transformers import SentenceTransformer

HIERARCHICAL_PATH = "data/processed/hierarchical_graph.json"
RAW_JSON_PATH = "automotive_faults_aktc_obike_et_al.json"
CHROMA_DIR = "data/chroma_db"
COLLECTION_NAME = "automotive_kg"
MODEL_NAME = "all-MiniLM-L6-v2"


def load_hierarchical_graph(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_raw_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_step_results(raw):
    """Build lookup: step_text -> (result_a, result_b)."""
    mapping = {}
    for r in raw:
        for ds in r.get("diagnosis_steps", []):
            step = ds["step"]
            res = ds.get("result", ["", ""])
            mapping[step] = (res[0] if len(res) > 0 else "",
                             res[1] if len(res) > 1 else "")
    return mapping


def build_parent_map(graph_data):
    """Build node_id -> (category_label, subcategory_label) for children."""
    # First build node lookup
    nodes = {n["id"]: n for n in graph_data["nodes"]}

    # Build child->parent chain using edges
    # cat -> subcat -> symptom/step
    subcat_to_cat = {}  # subcat_id -> category_label
    leaf_to_subcat = {}  # leaf_id -> subcat_id

    for e in graph_data["edges"]:
        src, tgt = e["source"], e["target"]
        rel = e["relation"]
        if rel == "HAS_SUBCATEGORY":
            cat_label = nodes[src]["label"]
            subcat_to_cat[tgt] = cat_label
        elif rel == "HAS_SYMPTOM":
            leaf_to_subcat[tgt] = src
        elif rel == "HAS_DIAGNOSIS_STEP":
            leaf_to_subcat[tgt] = src

    return subcat_to_cat, leaf_to_subcat, nodes


def main():
    # ── 1. Load data ──────────────────────────────────────────────────
    graph_data = load_hierarchical_graph(HIERARCHICAL_PATH)
    raw = load_raw_json(RAW_JSON_PATH)
    step_results = build_step_results(raw)
    subcat_to_cat, leaf_to_subcat, nodes = build_parent_map(graph_data)

    # ── 2. Build embedding entries ────────────────────────────────────
    entries = []  # each: (node_id, text, metadata)

    for n in graph_data["nodes"]:
        nid = n["id"]
        ntype = n["node_type"]
        label = n["label"]

        if ntype == "Subcategory":
            cat = subcat_to_cat.get(nid, "")
            text = f"{cat} > {label}"
            meta = {
                "node_id": nid,
                "node_type": ntype,
                "category": cat,
                "subcategory": label,
            }
            entries.append((nid, text, meta))

        elif ntype == "Symptom":
            scid = leaf_to_subcat.get(nid, "")
            sc_label = nodes[scid]["label"] if scid in nodes else ""
            cat = subcat_to_cat.get(scid, "")
            text = f"{cat} > {sc_label} | Symptom: {label}"
            meta = {
                "node_id": nid,
                "node_type": ntype,
                "category": cat,
                "subcategory": sc_label,
            }
            entries.append((nid, text, meta))

        elif ntype == "DiagnosisStep":
            scid = leaf_to_subcat.get(nid, "")
            sc_label = nodes[scid]["label"] if scid in nodes else ""
            cat = subcat_to_cat.get(scid, "")
            r_a, r_b = step_results.get(label, ("", ""))
            text = (f"{cat} > {sc_label} | Diagnosis step: {label}"
                    f" | Possible results: {r_a}, {r_b}")
            meta = {
                "node_id": nid,
                "node_type": ntype,
                "category": cat,
                "subcategory": sc_label,
            }
            entries.append((nid, text, meta))

    print(f"Entries to embed: {len(entries)}")
    # Sample a few texts
    for _, text, meta in entries[:3]:
        print(f"  [{meta['node_type']}] {text}")

    # ── 3. Embed ──────────────────────────────────────────────────────
    print(f"\nLoading model '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)
    texts = [e[1] for e in entries]
    print(f"Embedding {len(texts)} texts...")
    embeddings = model.encode(texts, show_progress_bar=True)
    print(f"Embedding dim: {embeddings.shape[1]}")

    # ── 4. Store in ChromaDB ──────────────────────────────────────────
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Delete existing collection if present for a clean rebuild
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [e[0] for e in entries]
    metadatas = [e[2] for e in entries]

    # Chroma add in batches to avoid memory issues
    batch_size = 64
    for i in range(0, len(ids), batch_size):
        end = min(i + batch_size, len(ids))
        collection.add(
            ids=ids[i:end],
            embeddings=embeddings[i:end].tolist(),
            metadatas=metadatas[i:end],
            documents=texts[i:end],
        )

    print(f"Stored {collection.count()} vectors in '{COLLECTION_NAME}' at {CHROMA_DIR}")

    # ── 5. Self-test query ────────────────────────────────────────────
    print("\n=== Self-test: 'brake pedal feels spongy' ===")
    query = "brake pedal feels spongy"
    q_emb = model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=5,
        include=["metadatas", "documents", "distances"],
    )

    for rank, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]),
            start=1):
        print(f"\n  {rank}. [{meta['node_type']}] (dist={dist:.4f})")
        print(f"     {doc[:120]}")

    print("\nDone.")


if __name__ == "__main__":
    main()
