import json
import hashlib
import os
import pickle
import networkx as nx
from collections import defaultdict

RAW_JSON_PATH = "automotive_faults_aktc_obike_et_al.json"
GRAPHIFY_PATH = "graphify-out/graph.json"
OUTPUT_DIR = "data/processed"
GPICKLE_PATH = os.path.join(OUTPUT_DIR, "hierarchical_graph.gpickle")
JSON_PATH = os.path.join(OUTPUT_DIR, "hierarchical_graph.json")


def _slug(name):
    import re
    s = name.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def _hash_id(prefix, text):
    h = hashlib.md5(text.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}:{h}"


def main():
    # ── 1. Load original source-of-truth JSON ──────────────────────────
    with open(RAW_JSON_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)

    # ── 2. Load Graphify graph for extra cross-links ──────────────────
    graphify_data = {}
    try:
        with open(GRAPHIFY_PATH, "r", encoding="utf-8") as f:
            g = json.load(f)
        # Build quick lookups: node label -> id, and all is_symptom_of edges
        graphify_nodes = {}
        for n in g.get("nodes", []):
            graphify_nodes[n["id"]] = {
                "label": n.get("label", ""),
                "norm_label": n.get("norm_label", ""),
            }
        graphify_edges = g.get("links", [])
        graphify_data = {"nodes": graphify_nodes, "links": graphify_edges}
        print(f"Loaded Graphify graph: {len(graphify_nodes)} nodes, {len(graphify_edges)} edges")
    except FileNotFoundError:
        print("Graphify graph not found – proceeding without cross-links")

    # ── 3. Build canonical id maps ────────────────────────────────────
    # Category slugs
    all_categories = sorted(set(r["category"] for r in records))

    # Subcategory -> category mapping (one subcategory may appear in JSON
    # under multiple records if it has multiple symptom/step groups, but
    # we treat it as one node)
    subcat_to_cat = {}
    for r in records:
        subcat_to_cat[r["subcategory"]] = r["category"]

    # ── 4. Create MultiDiGraph ────────────────────────────────────────
    G = nx.MultiDiGraph()

    # Add Category nodes
    for cat in all_categories:
        cid = f"cat:{_slug(cat)}"
        G.add_node(cid, node_type="Category", label=cat)

    # Process records to add subcategory, symptom, step, result nodes
    for r in records:
        cat = r["category"]
        subcat = r["subcategory"]
        cid = f"cat:{_slug(cat)}"
        scid = f"subcat:{_slug(subcat)}"

        # Add subcategory node if not already present
        if not G.has_node(scid):
            G.add_node(scid, node_type="Subcategory", label=subcat)

        # Edge: Category -> Subcategory
        if not G.has_edge(cid, scid, key="HAS_SUBCATEGORY"):
            G.add_edge(cid, scid, key="HAS_SUBCATEGORY",
                       relation="HAS_SUBCATEGORY")

        # Symptoms
        for s in r.get("symptoms", []):
            sid = _hash_id("sym", s)
            if not G.has_node(sid):
                G.add_node(sid, node_type="Symptom", label=s)
            G.add_edge(scid, sid, key="HAS_SYMPTOM",
                       relation="HAS_SYMPTOM")

        # Diagnosis Steps
        for ds in r.get("diagnosis_steps", []):
            step_text = ds["step"]
            did = _hash_id("step", step_text)
            if not G.has_node(did):
                G.add_node(did, node_type="DiagnosisStep", label=step_text)
            G.add_edge(scid, did, key="HAS_DIAGNOSIS_STEP",
                       relation="HAS_DIAGNOSIS_STEP")

    # ── 5. Extract extra cross-links from Graphify's output ──────────
    # Look for symptoms that appear as sources of is_symptom_of edges
    # targeting multiple different subcategories — those are overlaps
    # Graphify may have found that the raw JSON doesn't explicitly capture.
    # We project them as SIMILAR_SYMPTOM_TO edges.

    extra_edges = 0
    if graphify_data.get("links"):
        # Group symptom node ids by their label (norm_label) to find shared text
        sym_label_to_gid = defaultdict(list)
        for nid, ndata in graphify_data["nodes"].items():
            # Only consider nodes that look like symptoms in graphify's index
            # (they'll have an is_symptom_of outgoing edge)
            sym_label_to_gid[ndata["norm_label"]].append(nid)

        # For each Graphify symptom, find which subcat it connects to
        g_sym_to_subcats = defaultdict(set)
        for link in graphify_data["links"]:
            if link["relation"] == "is_symptom_of":
                g_sym_to_subcats[link["source"]].add(link["target"])

        # Map subcat graphify ids -> canonical subcat labels
        g_subcat_labels = {}
        for nid, ndata in graphify_data["nodes"].items():
            g_subcat_labels[nid] = ndata["label"]

        # Now find pairs of subcategories that share the same symptom label
        # in Graphify's output (meaning the same symptom text appears under
        # different subcategories)
        label_to_subcats = defaultdict(set)
        for g_sym_id, g_subcat_ids in g_sym_to_subcats.items():
            if g_sym_id in graphify_data["nodes"]:
                sym_label = graphify_data["nodes"][g_sym_id]["norm_label"]
                for g_sc_id in g_subcat_ids:
                    if g_sc_id in g_subcat_labels:
                        label_to_subcats[sym_label].add(g_subcat_labels[g_sc_id])

        # Generate SIMILAR_SYMPTOM_TO edges
        for sym_label, subcat_labels in label_to_subcats.items():
            subcat_labels_list = sorted(subcat_labels)
            if len(subcat_labels_list) < 2:
                continue
            # For each pair of subcategories sharing this symptom text,
            # add a SIMILAR_SYMPTOM_TO edge between them
            for i in range(len(subcat_labels_list)):
                for j in range(i + 1, len(subcat_labels_list)):
                    sc_a = f"subcat:{_slug(subcat_labels_list[i])}"
                    sc_b = f"subcat:{_slug(subcat_labels_list[j])}"
                    if G.has_node(sc_a) and G.has_node(sc_b):
                        G.add_edge(sc_a, sc_b, key="SIMILAR_SYMPTOM_TO",
                                   relation="SIMILAR_SYMPTOM_TO",
                                   symptom_label=sym_label)
                        extra_edges += 1

    print(f"Extra SIMILAR_SYMPTOM_TO edges from Graphify cross-links: {extra_edges}")

    # ── 6. Save outputs ───────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(GPICKLE_PATH, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved gpickle: {GPICKLE_PATH}")

    # Convert to node-link JSON (NetworkX's node-link format)
    node_link_data = nx.node_link_data(G, edges="edges")
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(node_link_data, f, indent=2, ensure_ascii=False)
    print(f"Saved JSON:    {JSON_PATH}")

    # ── 7. Print sanity counts ────────────────────────────────────────
    node_types = defaultdict(int)
    for _, ndata in G.nodes(data=True):
        node_types[ndata.get("node_type", "Unknown")] += 1

    edge_types = defaultdict(int)
    for _, _, _, edata in G.edges(keys=True, data=True):
        edge_types[edata.get("relation", "Unknown")] += 1

    print("\n=== Node counts by type ===")
    for nt in ["Category", "Subcategory", "Symptom", "DiagnosisStep"]:
        print(f"  {nt}: {node_types.get(nt, 0)}")
    print(f"  Total nodes: {G.number_of_nodes()}")

    print("\n=== Edge counts by type ===")
    for et in ["HAS_SUBCATEGORY", "HAS_SYMPTOM", "HAS_DIAGNOSIS_STEP",
               "SIMILAR_SYMPTOM_TO"]:
        print(f"  {et}: {edge_types.get(et, 0)}")
    print(f"  Total edges: {G.number_of_edges()}")


if __name__ == "__main__":
    main()
