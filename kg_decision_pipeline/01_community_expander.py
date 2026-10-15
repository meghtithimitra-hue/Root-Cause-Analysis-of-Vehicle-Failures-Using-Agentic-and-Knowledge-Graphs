"""
01_community_expander.py

Community-aware query expansion using Leiden community structure from
graphify-out/graph.json.

Takes the output of hybrid_retrieve() and expands the candidate list by
adding other nodes that belong to the same Leiden communities as the
original candidates. This surfaces related failure modes, symptoms, and
diagnosis steps that the vector + graph search may have missed but that
cluster alongside the matched results in the knowledge graph.

Usage:
    from kg_decision_pipeline.01_community_expander import expand_candidates
    expanded = expand_candidates(hybrid_retrieve("brake pedal spongy"))
"""

import json
from collections import defaultdict

GRAPHIFY_GRAPH_PATH = "graphify-out/graph.json"
COMMUNITY_WEIGHT = 0.15
MAX_COMMUNITY_MEMBERS = 10

# ---------------------------------------------------------------------------
# Lazy-loaded singletons
# ---------------------------------------------------------------------------
_node_to_comm = None
_comm_members = None
_label_to_comm = None


def _load_community_data(path=GRAPHIFY_GRAPH_PATH):
    """Load graphify-out/graph.json and build community indexes."""
    global _node_to_comm, _comm_members, _label_to_comm
    if _node_to_comm is not None:
        return _node_to_comm, _comm_members, _label_to_comm

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    node_to_comm = {}
    comm_members: dict[int, list[dict]] = defaultdict(list)
    label_to_comm = {}

    for node in data["nodes"]:
        nid = node["id"]
        comm = node.get("community")
        if comm is None:
            continue

        node_to_comm[nid] = comm
        entry = {
            "node_id": nid,
            "label": node.get("label", nid),
            "norm_label": node.get("norm_label", ""),
            "file_type": node.get("file_type", ""),
        }
        comm_members[comm].append(entry)

        label = node.get("label", "").lower().strip()
        norm = node.get("norm_label", "").lower().strip()
        if label:
            label_to_comm[label] = comm
        if norm and norm != label:
            label_to_comm[norm] = comm

    _node_to_comm = node_to_comm
    _comm_members = {k: v for k, v in comm_members.items()}
    _label_to_comm = label_to_comm
    return _node_to_comm, _comm_members, _label_to_comm


def _find_community(
    candidate: dict,
    node_to_comm: dict,
    label_to_comm: dict,
) -> int | None:
    """Resolve the community ID for a candidate via node_id or label."""
    nid = candidate.get("node_id", "")
    if nid in node_to_comm:
        return node_to_comm[nid]

    label = candidate.get("label", "").lower().strip()
    if label and label in label_to_comm:
        return label_to_comm[label]

    return None


def expand_candidates(
    hybrid_result: dict,
    max_community_members: int = MAX_COMMUNITY_MEMBERS,
    community_weight: float = COMMUNITY_WEIGHT,
) -> dict:
    """Expand hybrid retrieval candidates using Leiden community membership.

    Parameters
    ----------
    hybrid_result : dict
        The output of *hybrid_retrieve()* — must have keys *query* (str) and
        *candidates* (list of dicts, each containing at least *node_id*,
        *label*, *node_type*, *score*, *source*, *category*, *subcategory*).
    max_community_members : int
        Maximum number of additional nodes to inject per community (sorted
        alphabetically for determinism).
    community_weight : float
        Score assigned to each community-expanded node.

    Returns
    -------
    dict
        Dictionary matching the *hybrid_result* shape, with *candidates*
        extended by community-expanded entries (sorted by score descending).
    """
    node_to_comm, comm_members, label_to_comm = _load_community_data()

    query = hybrid_result["query"]
    candidates = hybrid_result["candidates"]
    seen_ids = {c["node_id"] for c in candidates}

    # -- Collect the unique communities touched by initial candidates ---------
    candidate_communities: set[int] = set()
    for c in candidates:
        comm = _find_community(c, node_to_comm, label_to_comm)
        if comm is not None:
            candidate_communities.add(comm)

    # -- Build expanded list --------------------------------------------------
    expanded = list(candidates)

    for comm_id in sorted(candidate_communities):
        members = sorted(
            comm_members.get(comm_id, []),
            key=lambda m: (m["file_type"] != "concept", m["label"]),
        )
        added = 0
        for member in members:
            if added >= max_community_members:
                break
            mid = member["node_id"]
            if mid in seen_ids:
                continue
            seen_ids.add(mid)
            expanded.append({
                "node_id": mid,
                "node_type": "community_expanded",
                "label": member["label"],
                "category": "",
                "subcategory": "",
                "score": community_weight,
                "source": "community",
                "community_id": comm_id,
            })
            added += 1

    expanded.sort(key=lambda x: -x["score"])
    return {"query": query, "candidates": expanded}


# ---------------------------------------------------------------------------
# __main__ smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.pipeline.hybrid_retrieval import hybrid_retrieve
    import pprint

    query = "brake warning light is on and pedal feels soft"
    print(f"Query: {query}\n")

    raw = hybrid_retrieve(query, top_k=10)
    print("--- Before community expansion ---")
    print(f"{'Rank':<5} {'Score':<6} {'Source':<10} {'Type':<20} {'Label'}")
    print("-" * 100)
    for i, c in enumerate(raw["candidates"], 1):
        print(
            f"{i:<5} {c['score']:<6.3f} {c['source']:<10} "
            f"{c.get('node_type',''):<20} {c['label'][:55]}"
        )

    expanded = expand_candidates(raw)
    print("\n--- After community expansion ---")
    print(f"{'Rank':<5} {'Score':<6} {'Source':<10} {'Type':<20} {'CommID':<8} {'Label'}")
    print("-" * 120)
    for i, c in enumerate(expanded["candidates"], 1):
        print(
            f"{i:<5} {c['score']:<6.3f} {c['source']:<10} "
            f"{c.get('node_type',''):<20} "
            f"{c.get('community_id',''):<8} {c['label'][:55]}"
        )
