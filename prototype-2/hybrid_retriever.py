"""
Hybrid retrieval pipeline: Vector Search → Community Expansion → Graph Traversal → Evidence Scoring.

Orchestrates the full automotive diagnosis flow by combining semantic vector search
with the existing Neo4j graph retrieval and Leiden community layer.
"""

from graph_retriever import (
    get_driver,
    retrieve_symptom,
    _flatten_community_context,
    _compute_path_strength,
    _compute_support_count,
    _compute_community_agreement,
    _compute_test_agreement,
)
from vector_retriever import search_similar, build_embeddings


# ── New evidence weights ──────────────────────────────────────────────────
_EVIDENCE_WEIGHTS = {
    "vector_similarity": 0.30,
    "path_strength": 0.25,
    "community_agreement": 0.20,
    "support_count": 0.15,
    "test_agreement": 0.10,
}


def _compute_hybrid_evidence(
    vector_similarity: float,
    path_strength: float,
    support_count: float,
    community_agreement: float,
    test_agreement: float,
) -> float:
    """Compute evidence score using the hybrid formula (0–100)."""
    return round(100 * (
        _EVIDENCE_WEIGHTS["vector_similarity"] * vector_similarity +
        _EVIDENCE_WEIGHTS["path_strength"] * path_strength +
        _EVIDENCE_WEIGHTS["community_agreement"] * community_agreement +
        _EVIDENCE_WEIGHTS["support_count"] * support_count +
        _EVIDENCE_WEIGHTS["test_agreement"] * test_agreement
    ), 1)


# ── Main pipeline ─────────────────────────────────────────────────────────

def hybrid_retrieve(query: str) -> dict:
    """
    Full hybrid retrieval pipeline:

    1. Vector search — semantically match user query to graph node names
    2. Pick best Symptom match
    3. Standard graph retrieval (Symptom → Component → Test → Result → Repair)
    4. Community expansion via Leiden communities
    5. Hybrid evidence scoring with vector similarity weight
    """
    # ── Step 1: Vector search ────────────────────────────────────────────
    try:
        matches = search_similar(query, top_k=5)
    except Exception:
        matches = []

    symptom_matches = [m for m in matches if "Symptom" in m.get("labels", [])]
    best_symptom_match = symptom_matches[0] if symptom_matches else None

    if not best_symptom_match:
        matched_symptom = query
        vector_similarity = 0.0
    else:
        matched_symptom = best_symptom_match["name"]
        vector_similarity = best_symptom_match["score"]

    # ── Step 2: Graph retrieval ──────────────────────────────────────────
    try:
        result = retrieve_symptom(matched_symptom)
    except Exception:
        result = _empty_hybrid_result(query)

    path_strength = result.get("path_strength", 0)
    support_count = result.get("support_count", 0)
    community_agreement = result.get("community_agreement", 0)
    test_agreement = result.get("test_agreement", 0)

    # ── Step 3: Community members ────────────────────────────────────────
    ctx = result.get("community_context", {})
    community_members = _flatten_community_context(ctx)

    # ── Step 4: Evidence score ────────────────────────────────────────────
    evidence_score = _compute_hybrid_evidence(
        vector_similarity, path_strength, support_count,
        community_agreement, test_agreement,
    )

    result["vector_similarity"] = round(vector_similarity, 4)
    result["vector_matches"] = matches
    result["matched_symptom"] = matched_symptom
    result["community_members"] = community_members
    result["evidence_score"] = evidence_score
    result["raw_query"] = query

    return result


def _empty_hybrid_result(query: str) -> dict:
    return {
        "symptom": query,
        "matched_symptom": query,
        "vector_similarity": 0.0,
        "vector_matches": [],
        "diagnostic_tests": [],
        "results": [],
        "repair_actions": [],
        "reasoning_paths": [],
        "top_paths": [],
        "community": None,
        "community_context": {},
        "community_members": [],
        "evidence_score": 0,
        "path_strength": 0,
        "support_count": 0,
        "community_agreement": 0,
        "test_agreement": 0,
    }


def main():
    """CLI entry point for hybrid retrieval."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python hybrid_retriever.py <query>")
        print("       python hybrid_retriever.py build  (build embeddings)")
        return

    arg = sys.argv[1].lower()
    if arg == "build":
        count = build_embeddings()
        print(f"Embeddings built for {count} nodes.")
        return

    query = " ".join(sys.argv[1:])
    result = hybrid_retrieve(query)
    print(f"Query:             {result.get('raw_query', query)}")
    print(f"Matched Symptom:   {result.get('matched_symptom', 'N/A')}")
    print(f"Vector Similarity: {result.get('vector_similarity', 0):.4f}")
    if result.get("community"):
        c = result["community"]
        print(f"Community:         {c.get('name', 'N/A')}  (id={c.get('id', '?')})")
    members = result.get("community_members", [])
    if members:
        print(f"Community members: {', '.join(members[:5])}")
    print(f"Evidence Score:    {result.get('evidence_score', 0)}%")
    print()
    print("Vector matches:")
    for m in result.get("vector_matches", []):
        print(f"  {m['score']:.4f}  {m['name']}")
    print()
    print(f"Diagnostic Tests: {', '.join(result.get('diagnostic_tests', [])) or '(none)'}")
    print(f"Results:          {', '.join(result.get('results', [])) or '(none)'}")
    print(f"Repair Actions:   {', '.join(result.get('repair_actions', [])) or '(none)'}")
    print()
    print("Top 3 Paths:")
    for path in result.get("top_paths", []):
        print(f"  {' -> '.join(path)}")


if __name__ == "__main__":
    main()
