import json
import os
from collections import deque

import chromadb
import networkx as nx
from sentence_transformers import SentenceTransformer

CHROMA_DIR = "data/chroma_db"
COLLECTION_NAME = "automotive_kg"
MODEL_NAME = "all-MiniLM-L6-v2"
GRAPH_PATH = "data/processed/hierarchical_graph.json"


# ---------------------------------------------------------------------------
# Lazy-loaded singletons
# ---------------------------------------------------------------------------
_model = None
_chroma_collection = None
_graph_nx = None
_graph_nodes = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _get_collection():
    global _chroma_collection
    if _chroma_collection is None:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _chroma_collection = client.get_collection(COLLECTION_NAME)
    return _chroma_collection


def _load_graph(path=GRAPH_PATH):
    global _graph_nx, _graph_nodes
    if _graph_nx is not None:
        return _graph_nx, _graph_nodes
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    _graph_nx = nx.node_link_graph(data, edges="edges", multigraph=False)
    _graph_nodes = {n["id"]: n for n in data["nodes"]}
    return _graph_nx, _graph_nodes


# ---------------------------------------------------------------------------
# 1. Embedding
# ---------------------------------------------------------------------------
def embed_query(query: str) -> list[float]:
    model = _get_model()
    return model.encode(query).tolist()


# ---------------------------------------------------------------------------
# 2. Vector search (ChromaDB)
# ---------------------------------------------------------------------------
def vector_search(
    query_embedding: list[float],
    top_k: int = 10,
) -> list[dict]:
    collection = _get_collection()
    raw = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["metadatas", "distances"],
    )
    results = []
    for meta, dist in zip(raw["metadatas"][0], raw["distances"][0]):
        results.append({
            "node_id": meta["node_id"],
            "node_type": meta["node_type"],
            "label": _extract_label(meta),
            "category": meta.get("category", ""),
            "subcategory": meta.get("subcategory", ""),
            "distance": dist,
        })
    return results


def _extract_label(meta: dict) -> str:
    """Derive a human-readable label from metadata."""
    nid = meta["node_id"]
    ntype = meta["node_type"]
    # Pull the label from the canonical graph node if we can
    _, graph_nodes = _load_graph()
    if nid in graph_nodes:
        return graph_nodes[nid]["label"]
    # Fallback
    parts = nid.split(":", 1)
    return parts[1] if len(parts) > 1 else nid


# ---------------------------------------------------------------------------
# 3. Graph search (substring match + BFS)
# ---------------------------------------------------------------------------
def graph_search(
    query: str,
    graph_path: str = GRAPH_PATH,
    max_hops: int = 2,
    top_k: int = 10,
) -> list[dict]:
    G, graph_nodes = _load_graph(graph_path)
    ql = query.lower()
    query_words = [w for w in ql.split() if len(w) > 2]

    # – Find seed nodes whose label contains any query word (case-insensitive) -
    seed_ids = set()
    for nid, ndata in graph_nodes.items():
        label = ndata.get("label", "").lower()
        if any(w in label for w in query_words):
            seed_ids.add(nid)

    if not seed_ids:
        return []

    # – BFS from seeds ------------------------------------------------------
    visited = {}
    queue = deque()
    for sid in seed_ids:
        visited[sid] = 0
        queue.append((sid, 0))

    while queue:
        cur, dist = queue.popleft()
        if dist >= max_hops:
            continue
        for neighbor in G.neighbors(cur):
            if neighbor not in visited:
                visited[neighbor] = dist + 1
                queue.append((neighbor, dist + 1))

    # – Build result list ---------------------------------------------------
    results = []
    for nid, hop in visited.items():
        ndata = graph_nodes.get(nid, {})
        ntype = ndata.get("node_type", "Unknown")
        label = ndata.get("label", nid)
        # Derive category / subcategory from graph positions
        cat, subcat = _derive_hierarchy(nid, G, graph_nodes)
        results.append({
            "node_id": nid,
            "node_type": ntype,
            "label": label,
            "category": cat,
            "subcategory": subcat,
            "hop_distance": hop,
        })

    # Sort: lower hop first; within same hop prefer nodes whose label
    # directly contains a query-word substring over those that merely
    # neighbor a match (e.g. symptom before its parent subcategory).
    query_words_set = set(query_words)
    def _sort_key(r):
        label_words = set(r["label"].lower().split())
        overlap = len(query_words_set & label_words)
        return (r["hop_distance"], -overlap, r["label"])
    results.sort(key=_sort_key)
    return results[:top_k]


def _derive_hierarchy(
    nid: str, G: nx.Graph, graph_nodes: dict
) -> tuple[str, str]:
    """Walk predecessor edges to find the true category / subcategory chain.

    Edge directions:
        Category --HAS_SUBCATEGORY--> Subcategory
        Subcategory --HAS_SYMPTOM/HAS_DIAGNOSIS_STEP--> Symptom / DiagnosisStep
    So a node's parent is its predecessor in the directed graph.
    """
    _REL = {"HAS_SUBCATEGORY", "HAS_SYMPTOM", "HAS_DIAGNOSIS_STEP"}

    def _parent(n, allowed_rels=None):
        """Return the first predecessor connected via an allowed relation."""
        for pred in G.predecessors(n):
            edge_data = G.get_edge_data(pred, n)
            if edge_data is None:
                continue
            rel = edge_data.get("relation", "")
            if allowed_rels is None or rel in allowed_rels:
                return pred
        return None

    def _label(nid):
        nd = graph_nodes.get(nid)
        return nd["label"] if nd else nid

    # – Category node: no parent ---------------------------------------------
    ndata = graph_nodes.get(nid)
    ntype = ndata.get("node_type", "") if ndata else ""
    if ntype == "Category":
        return (_label(nid), "")

    # – Subcategory node: walk up one hop via HAS_SUBCATEGORY -----------------
    if ntype == "Subcategory":
        cat_id = _parent(nid, allowed_rels={"HAS_SUBCATEGORY"})
        cat_label = _label(cat_id) if cat_id else ""
        return (cat_label, "")

    # – Symptom / DiagnosisStep: walk up two hops -----------------------------
    subcat_id = _parent(nid)
    if subcat_id is None:
        return ("", "")
    subcat_label = _label(subcat_id)
    cat_id = _parent(subcat_id, allowed_rels={"HAS_SUBCATEGORY"})
    cat_label = _label(cat_id) if cat_id else ""
    return (cat_label, subcat_label)


# ---------------------------------------------------------------------------
# 4. Hybrid retrieval
# ---------------------------------------------------------------------------
def hybrid_retrieve(query: str, top_k: int = 10) -> dict:
    q_emb = embed_query(query)

    vec_results = vector_search(q_emb, top_k=top_k)
    # Request more graph candidates so BFS-discovered nodes (categories,
    # subcategories) aren't silently truncated before the merge step.
    gph_results = graph_search(query, max_hops=2, top_k=top_k * 3)

    # Merge by node_id
    merged: dict[str, dict] = {}
    for r in vec_results:
        nid = r["node_id"]
        merged[nid] = {
            "node_id": nid,
            "node_type": r["node_type"],
            "label": r["label"],
            "category": r["category"],
            "subcategory": r["subcategory"],
            "score": 1.0 - r["distance"],
            "source": "vector",
        }

    for r in gph_results:
        nid = r["node_id"]
        if nid in merged:
            merged[nid]["source"] = "both"
            merged[nid]["score"] += 0.5  # boost
        else:
            merged[nid] = {
                "node_id": nid,
                "node_type": r["node_type"],
                "label": r["label"],
                "category": r["category"],
                "subcategory": r["subcategory"],
                "score": 0.3,
                "source": "graph",
            }

    candidates = sorted(merged.values(), key=lambda x: -x["score"])
    return {"query": query, "candidates": candidates[:top_k]}


# ---------------------------------------------------------------------------
# __main__ test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import pprint

    query = "brake pedal feels spongy when I press it"
    result = hybrid_retrieve(query, top_k=10)

    print(f"Query: {result['query']}\n")
    print(f"{'Rank':<5} {'Score':<6} {'Source':<7} {'Type':<16} {'Category':<22} {'Subcategory':<24} {'Label'}")
    print("-" * 140)
    for i, c in enumerate(result["candidates"], 1):
        print(
            f"{i:<5} {c['score']:<6.3f} {c['source']:<7} "
            f"{c['node_type']:<16} {c['category'][:22]:<22} {c['subcategory'][:24]:<24} "
            f"{c['label'][:55]}"
        )
