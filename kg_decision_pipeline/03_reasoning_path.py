"""
03_reasoning_path.py

Builds a structured reasoning path from scored candidates (EXTRACTED /
INFERRED only).  Traverses the hierarchical knowledge graph at
data/processed/hierarchical_graph.json to walk from each matched node up
to its Subcategory and Category parents, collects sibling Symptoms and
DiagnosisSteps, and produces a human-readable chain of reasoning.

Usage:
    from kg_decision_pipeline.03_reasoning_path import build_reasoning_path
    path = build_reasoning_path(scored_result)
"""

import json
from collections import defaultdict
from typing import Any

HIERARCHICAL_GRAPH_PATH = "data/processed/hierarchical_graph.json"

# ---------------------------------------------------------------------------
# Lazy-loaded indexes
# ---------------------------------------------------------------------------
_nodes: dict[str, dict] | None = None
_parents: dict[str, list[str]] | None = None
_children_by_rel: dict[str, dict[str, list[str]]] | None = None


def _load_hierarchical_graph(path=HIERARCHICAL_GRAPH_PATH):
    global _nodes, _parents, _children_by_rel
    if _nodes is not None:
        return _nodes, _parents, _children_by_rel

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    nodes = {}
    parents: dict[str, list[str]] = defaultdict(list)
    children_by_rel: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for node in data["nodes"]:
        nodes[node["id"]] = node

    for edge in data.get("edges", []):
        src = edge["source"]
        tgt = edge["target"]
        rel = edge["relation"]
        parents[tgt].append(src)
        children_by_rel[src][rel].append(tgt)

    _nodes = nodes
    _parents = dict(parents)
    _children_by_rel = dict(children_by_rel)
    return _nodes, _parents, _children_by_rel


def _walk_up(
    node_id: str,
    nodes: dict,
    parents: dict,
) -> tuple[dict | None, dict | None, list[str]]:
    """Walk from a node up to its Subcategory then Category parent.

    Returns (subcategory_node, category_node, chain_steps)
    where chain_steps are human-readable strings.
    """
    current = nodes.get(node_id)
    if current is None:
        return None, None, []

    nt = current.get("node_type", "")
    chain = []

    # -- Subcategory or above ------------------------------------------------
    if nt == "Subcategory":
        subcat = current
    elif nt == "Category":
        return None, current, [f"Category '{current['label']}' is the root."]
    elif nt in ("Symptom", "DiagnosisStep"):
        p_ids = parents.get(node_id, [])
        subcat_node = None
        for pid in p_ids:
            pn = nodes.get(pid)
            if pn and pn.get("node_type") == "Subcategory":
                subcat_node = pn
                chain.append(
                    f"{nt} '{current['label']}' matched — "
                    f"under Subcategory '{pn['label']}'."
                )
                subcat = pn
                break
        else:
            chain.append(
                f"{nt} '{current['label']}' matched — no Subcategory parent found."
            )
            return None, None, chain
    else:
        return None, None, [f"Unknown node type '{nt}' for '{current.get('label','?')}'."]

    # -- Walk up to Category -------------------------------------------------
    cat_node = None
    p_ids = parents.get(subcat["id"], [])
    for pid in p_ids:
        pn = nodes.get(pid)
        if pn and pn.get("node_type") == "Category":
            cat_node = pn
            chain.append(
                f"Subcategory '{subcat['label']}' belongs to "
                f"Category '{pn['label']}'."
            )
            break
    else:
        chain.append(
            f"Subcategory '{subcat['label']}' has no Category parent."
        )

    return subcat, cat_node, chain


def build_reasoning_path(
    scored_result: dict,
    max_candidates: int = 5,
) -> dict[str, Any]:
    """Build a reasoning path from the top EXTRACTED / INFERRED candidates.

    Parameters
    ----------
    scored_result : dict
        Output of *score_candidates()* — must have keys *mode*, *query*, and
        *candidates_scored* (each candidate dict with at least *node_id*,
        *label*, *tag*, *node_type*).
    max_candidates : int
        Max top candidates to include in the reasoning path.

    Returns
    -------
    dict with keys:
        query              str
        mode               str
        top_subcategory    str | None
        top_category       str | None
        matched_symptoms   list[str]
        diagnosis_steps    list[str]
        reasoning_chain    list[str]
    """
    nodes, parents, children_by_rel = _load_hierarchical_graph()

    query = scored_result.get("query", "")
    mode = scored_result.get("mode", "AMBIGUOUS")
    candidates = scored_result.get("candidates_scored", [])

    # -- Filter to EXTRACTED / INFERRED only ---------------------------------
    high_conf = [c for c in candidates if c.get("tag") in ("EXTRACTED", "INFERRED")]
    high_conf = high_conf[:max_candidates]

    matched_symptoms: list[str] = []
    diagnosis_steps: list[str] = []
    reasoning_chain: list[str] = []
    top_subcategory = None
    top_category = None

    seen_symptoms: set[str] = set()
    seen_steps: set[str] = set()

    for cand in high_conf:
        nid = cand.get("node_id", "")
        label = cand.get("label", "")
        tag = cand.get("tag", "")
        nt = cand.get("node_type", "")

        subcat, cat, chain_steps = _walk_up(nid, nodes, parents)

        # Capture the first Subcategory / Category as "top"
        if subcat and top_subcategory is None:
            top_subcategory = subcat["label"]
        if cat and top_category is None:
            top_category = cat["label"]

        reasoning_chain.extend(chain_steps)

        # -- Collect siblings at the Subcategory level -----------------------
        if subcat is None:
            continue

        sid = subcat["id"]
        sub_children = children_by_rel.get(sid, {})

        for sym_id in sub_children.get("HAS_SYMPTOM", []):
            sym_node = nodes.get(sym_id)
            if sym_node and sym_id not in seen_symptoms:
                seen_symptoms.add(sym_id)
                matched_symptoms.append(sym_node["label"])

        for step_id in sub_children.get("HAS_DIAGNOSIS_STEP", []):
            step_node = nodes.get(step_id)
            if step_node and step_id not in seen_steps:
                seen_steps.add(step_id)
                diagnosis_steps.append(step_node["label"])

        reasoning_chain.append(
            f"Related diagnosis steps under '{subcat['label']}': "
            f"{', '.join(diagnosis_steps[-len(sub_children.get('HAS_DIAGNOSIS_STEP',[])):]) or 'none'}."
        )

    # -- Summarize the reasoning path ---------------------------------------
    if not high_conf:
        reasoning_chain.append(
            "No EXTRACTED or INFERRED candidates — insufficient confidence "
            "to build a reasoning path."
        )

    return {
        "query": query,
        "mode": mode,
        "top_subcategory": top_subcategory,
        "top_category": top_category,
        "matched_symptoms": matched_symptoms,
        "diagnosis_steps": diagnosis_steps,
        "reasoning_chain": reasoning_chain,
    }


# ---------------------------------------------------------------------------
# __main__ smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import importlib
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from scripts.pipeline.hybrid_retrieval import hybrid_retrieve
    expander = importlib.import_module(
        "kg_decision_pipeline.01_community_expander"
    )
    scorer = importlib.import_module(
        "kg_decision_pipeline.02_confidence_scorer"
    )

    query = "brake warning light is on and pedal feels soft"
    raw = hybrid_retrieve(query, top_k=10)
    expanded = expander.expand_candidates(raw)
    scored = scorer.score_candidates(expanded)
    path = build_reasoning_path(scored)

    print(f"Query: {path['query']}")
    print(f"Mode: {path['mode']}")
    print(f"Top Category: {path['top_category']}")
    print(f"Top Subcategory: {path['top_subcategory']}")
    print(f"\nMatched Symptoms ({len(path['matched_symptoms'])}):")
    for s in path["matched_symptoms"]:
        print(f"  - {s}")
    print(f"\nDiagnosis Steps ({len(path['diagnosis_steps'])}):")
    for s in path["diagnosis_steps"]:
        print(f"  - {s}")
    print(f"\nReasoning Chain ({len(path['reasoning_chain'])} steps):")
    for i, step in enumerate(path["reasoning_chain"], 1):
        print(f"  {i}. {step}")
