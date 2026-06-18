"""
Vector retrieval layer using SentenceTransformers + numpy.

Provides semantic search over automotive graph node names before graph traversal.
Embeddings are cached locally to avoid rebuilding on every run.
"""

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

_EMBEDDINGS_DIR = Path(__file__).resolve().parent / "embeddings"
_NODES_FILE = _EMBEDDINGS_DIR / "all_nodes.json"
_EMBEDDINGS_FILE = _EMBEDDINGS_DIR / "node_embeddings.pkl"

_MODEL = None


def _get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL


def build_embeddings() -> int:
    """Fetch all relevant node names from Neo4j, generate embeddings, and cache them."""
    from graph_retriever import get_driver

    driver = get_driver()
    query = """
    MATCH (n)
    WHERE n:Symptom OR n:Component OR n:DiagnosticTest OR n:Result OR n:RepairAction
    RETURN n.name AS name, labels(n) AS labels
    """
    try:
        with driver.session(database="neo4j") as session:
            rows = session.run(query).data()
    finally:
        driver.close()

    if not rows:
        raise ValueError("No nodes found in Neo4j")

    _EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    names = [r["name"] for r in rows]
    metadata = [{"name": r["name"], "labels": [l for l in r["labels"] if l != "Community"]} for r in rows]

    with open(_NODES_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    model = _get_model()
    embeddings = model.encode(names, show_progress_bar=True)

    with open(_EMBEDDINGS_FILE, "wb") as f:
        pickle.dump({"names": names, "embeddings": embeddings, "metadata": metadata}, f)

    return len(names)


def _load_embeddings() -> dict:
    """Load cached embeddings; build them if missing."""
    if not _EMBEDDINGS_FILE.exists():
        build_embeddings()
    with open(_EMBEDDINGS_FILE, "rb") as f:
        return pickle.load(f)


def search_similar(query: str, top_k: int = 5) -> list[dict]:
    """
    Return up to ``top_k`` semantically similar node names + scores.

    Each result dict has keys: ``name``, ``labels``, ``score``.
    """
    if not query or not query.strip():
        return []

    data = _load_embeddings()
    names: list[str] = data["names"]
    embeddings: np.ndarray = data["embeddings"]
    metadata: list[dict] = data.get("metadata", [{"labels": []} for _ in names])

    model = _get_model()
    query_emb = model.encode([query.strip()])

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norm_embeddings = embeddings / np.where(norms > 0, norms, 1)
    query_norm = query_emb / np.linalg.norm(query_emb)

    scores = np.dot(norm_embeddings, query_norm.T).flatten()
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    seen_names: set[str] = set()
    for idx in top_indices:
        name = names[idx]
        if name in seen_names:
            continue
        seen_names.add(name)
        results.append({
            "name": name,
            "labels": metadata[idx].get("labels", []) if isinstance(metadata[idx], dict) else [],
            "score": round(float(scores[idx]), 4),
        })
    return results


def main():
    """CLI entry point: build embeddings or search."""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python vector_retriever.py build            # build / rebuild embeddings")
        print("  python vector_retriever.py search <query>   # search similar nodes")
        return

    command = sys.argv[1].lower()
    if command == "build":
        count = build_embeddings()
        print(f"Embeddings built for {count} nodes.")
    elif command == "search":
        if len(sys.argv) < 3:
            print("Provide a search query.")
            return
        query = " ".join(sys.argv[2:])
        results = search_similar(query)
        print(f"Top matches for '{query}':")
        for r in results:
            labels = ", ".join(r["labels"]) if r["labels"] else "?"
            print(f"  {r['score']:.4f}  {r['name']}  [{labels}]")
    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
