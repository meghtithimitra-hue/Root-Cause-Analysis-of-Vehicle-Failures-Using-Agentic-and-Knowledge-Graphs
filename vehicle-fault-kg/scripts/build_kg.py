import json
import re
import hashlib
import pickle
from pathlib import Path
from collections import defaultdict
import networkx as nx

RAW_JSON = "data/automotive_faults_aktc_obike_et_al.json"
OUT_DIR = Path("graphify-out")
OUT_DIR.mkdir(exist_ok=True)
Path("data/processed").mkdir(exist_ok=True)

# ── Load raw data ──────────────────────────────────────────────────────────────
with open(RAW_JSON, encoding="utf-8-sig") as f:
    records = json.load(f)

print(f"Loaded {len(records)} records")

# ── Helpers ────────────────────────────────────────────────────────────────────
def slug(text):
    return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')

def md5(text):
    return hashlib.md5(text.encode()).hexdigest()[:8]

def node_id(prefix, text):
    return f"{prefix}_{slug(text)[:40]}_{md5(text)}"

# ── STEP 1: Build extraction dicts ─────────────────────────────────────────────
print("\n=== STEP 1: Building extraction JSON ===")

nodes = {}   # id -> node dict
edges = []   # list of edge dicts

# Track first occurrence of each symptom/step text for source_file assignment
sym_first = {}   # symptom_text -> source_file
step_first = {}  # step_text    -> source_file

# Track which categories each symptom/step appears in
sym_cats  = defaultdict(set)
step_cats = defaultdict(set)

# First pass — collect shared symptom/step info
for r in records:
    cat    = r['category']
    subcat = r['subcategory']
    sf_sub = f"data/corpus/{slug(cat)}/{slug(subcat)}.md"
    for sym in r['symptoms']:
        sym_cats[sym].add(cat)
        if sym not in sym_first:
            sym_first[sym] = sf_sub
    for d in r['diagnosis_steps']:
        step_text = d['step']
        step_cats[step_text].add(cat)
        if step_text not in step_first:
            step_first[step_text] = sf_sub

# Second pass — build nodes and edges
for r in records:
    cat    = r['category']
    subcat = r['subcategory']
    sf_cat = f"data/corpus/{slug(cat)}.md"
    sf_sub = f"data/corpus/{slug(cat)}/{slug(subcat)}.md"

    # Category node
    cat_id = node_id("cat", cat)
    if cat_id not in nodes:
        nodes[cat_id] = {
            "id": cat_id, "label": cat,
            "file_type": "document", "source_file": sf_cat
        }

    # Subcategory node
    sub_id = node_id("subcat", subcat)
    if sub_id not in nodes:
        nodes[sub_id] = {
            "id": sub_id, "label": subcat,
            "file_type": "document", "source_file": sf_sub
        }
    edges.append({
        "source": cat_id, "target": sub_id,
        "relation": "HAS_SUBCATEGORY",
        "confidence": "EXTRACTED",
        "source_file": sf_cat, "weight": 1.0
    })

    # Symptom nodes
    for sym in r['symptoms']:
        sym_id = node_id("sym", sym)
        if sym_id not in nodes:
            nodes[sym_id] = {
                "id": sym_id, "label": sym,
                "file_type": "document",
                "source_file": sym_first[sym]
            }
        edges.append({
            "source": sub_id, "target": sym_id,
            "relation": "HAS_SYMPTOM",
            "confidence": "EXTRACTED",
            "source_file": sf_sub, "weight": 1.0
        })

    # DiagnosisStep nodes
    for d in r['diagnosis_steps']:
        step_text = d['step']
        results   = d.get('result', ['', ''])
        step_id   = node_id("step", step_text)
        if step_id not in nodes:
            nodes[step_id] = {
                "id": step_id, "label": step_text,
                "file_type": "document",
                "source_file": step_first[step_text],
                "result_a": results[0] if len(results) > 0 else "",
                "result_b": results[1] if len(results) > 1 else ""
            }
        edges.append({
            "source": sub_id, "target": step_id,
            "relation": "HAS_DIAGNOSIS_STEP",
            "confidence": "EXTRACTED",
            "source_file": sf_sub, "weight": 1.0
        })

# Cross-category SIMILAR edges
cross_sym  = 0
cross_step = 0
seen_sim   = set()

for sym, cats in sym_cats.items():
    if len(cats) > 1:
        sym_id = node_id("sym", sym)
        for d in r['diagnosis_steps']:  # just to iterate records
            pass
        # Find all records with this symptom across different categories
        subcat_nodes = []
        for r2 in records:
            if sym in r2['symptoms']:
                subcat_nodes.append((r2['category'], node_id("subcat", r2['subcategory'])))
        unique_cats = list({c for c, _ in subcat_nodes})
        if len(unique_cats) > 1:
            # Add SIMILAR_SYMPTOM_TO between the first-occurrence node
            # and all other subcategory symptom nodes in different categories
            for i in range(len(subcat_nodes)):
                for j in range(i+1, len(subcat_nodes)):
                    c1, s1 = subcat_nodes[i]
                    c2, s2 = subcat_nodes[j]
                    if c1 != c2:
                        key = tuple(sorted([s1, s2]))
                        if key not in seen_sim:
                            seen_sim.add(key)
                            edges.append({
                                "source": s1, "target": s2,
                                "relation": "SIMILAR_SYMPTOM_TO",
                                "confidence": "INFERRED",
                                "source_file": sym_first[sym],
                                "weight": 0.8,
                                "label": sym
                            })
                            cross_sym += 1

for step_text, cats in step_cats.items():
    if len(cats) > 1:
        subcat_nodes = []
        for r2 in records:
            for d in r2['diagnosis_steps']:
                if d['step'] == step_text:
                    subcat_nodes.append((r2['category'], node_id("subcat", r2['subcategory'])))
        unique_cats = list({c for c, _ in subcat_nodes})
        if len(unique_cats) > 1:
            for i in range(len(subcat_nodes)):
                for j in range(i+1, len(subcat_nodes)):
                    c1, s1 = subcat_nodes[i]
                    c2, s2 = subcat_nodes[j]
                    if c1 != c2:
                        key = tuple(sorted([s1, s2]))
                        if key not in seen_sim:
                            seen_sim.add(key)
                            edges.append({
                                "source": s1, "target": s2,
                                "relation": "SIMILAR_STEP_TO",
                                "confidence": "INFERRED",
                                "source_file": step_first[step_text],
                                "weight": 0.8,
                                "label": step_text
                            })
                            cross_step += 1

node_list = list(nodes.values())
extraction = {
    "nodes": node_list,
    "edges": edges,
    "token_input": 0,
    "token_output": 0
}

with open("data/processed/extraction.json", "w") as f:
    json.dump(extraction, f, indent=2)

# Count by type
type_counts = defaultdict(int)
for n in node_list:
    prefix = n['id'].split('_')[0]
    type_counts[prefix] += 1

rel_counts = defaultdict(int)
for e in edges:
    rel_counts[e['relation']] += 1

print(f"Nodes: {dict(type_counts)}")
print(f"Edges: {dict(rel_counts)}")
print(f"Cross-category SIMILAR_SYMPTOM_TO: {cross_sym}")
print(f"Cross-category SIMILAR_STEP_TO: {cross_step}")

# ── STEP 2: Build graph using graphify.build ───────────────────────────────────
print("\n=== STEP 2: Building NetworkX graph via graphify.build ===")
try:
    from graphify.build import build_from_json
    G = build_from_json(extraction, directed=True)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
except Exception as e:
    print(f"graphify.build failed: {e}, falling back to networkx directly")
    G = nx.DiGraph()
    for n in node_list:
        G.add_node(n['id'], **n)
    for e in edges:
        G.add_edge(e['source'], e['target'], **e)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# ── STEP 3: Deduplicate using graphify.dedup ───────────────────────────────────
print("\n=== STEP 3: Deduplication via graphify.dedup ===")
try:
    # Prevent scipy/numpy C extension hang on Windows
    import os
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    from graphify.dedup import deduplicate_entities

    nodes_in = [{"id": n, **d} for n, d in G.nodes(data=True)]
    edges_in = [{"source": u, "target": v, **d}
                for u, v, d in G.edges(data=True)]

    result = deduplicate_entities(nodes_in, edges_in)

    if isinstance(result, tuple) and len(result) == 3:
        nodes_d, edges_d, merge_map = result
    elif isinstance(result, tuple) and len(result) == 2:
        nodes_d, edges_d = result
        merge_map = {}
    else:
        nodes_d, edges_d = nodes_in, edges_in
        merge_map = {}

    print(f"Before dedup: {len(nodes_in)} nodes")
    print(f"After dedup:  {len(nodes_d)} nodes")
    print(f"Merged pairs: {len(merge_map)}")

    # Rebuild G from deduped nodes/edges
    G2 = nx.DiGraph()
    for n in nodes_d:
        n = dict(n)
        nid = n.pop("id")
        G2.add_node(nid, **n)
    for e in edges_d:
        e = dict(e)
        src = e.pop("source")
        tgt = e.pop("target")
        if G2.has_node(src) and G2.has_node(tgt):
            G2.add_edge(src, tgt, **e)
    G = G2
    print(f"Graph after dedup: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges")

except Exception as e:
    print(f"graphify.dedup failed: {e}")
    print("Continuing with undeduped graph")

# ── STEP 4: Leiden clustering via graphify.cluster ────────────────────────────
print("\n=== STEP 4: Leiden clustering via graphify.cluster ===")
try:
    from graphify.cluster import cluster
    G, community_map_raw = cluster(G)
    print(f"Communities detected: {len(set(nx.get_node_attributes(G, 'community').values()))}")
except Exception as e:
    print(f"graphify.cluster failed ({e}), running networkx louvain fallback")
    try:
        import networkx.algorithms.community as nx_comm
        UG = G.to_undirected()
        partition = nx_comm.louvain_communities(UG, seed=42)
        for i, comm in enumerate(partition):
            for nid in comm:
                if G.has_node(nid):
                    G.nodes[nid]['community'] = i
        print(f"Louvain communities: {len(partition)}")
    except Exception as e2:
        print(f"Louvain also failed: {e2}")

# ── STEP 5: Analyze via graphify.analyze functions directly ───────────────────
print("\n=== STEP 5: Graph analysis via graphify.analyze ===")
analysis = {"god_nodes": [], "surprising_connections": [], "suggested_questions": []}
try:
    from graphify.analyze import god_nodes, surprising_connections, suggest_questions

    # God nodes — highest degree hubs
    gn_list = god_nodes(G)
    print("Top 5 God Nodes:")
    for gn in gn_list[:5]:
        label = gn.get('label', gn.get('id', ''))
        degree = gn.get('degree', '?')
        print(f"  {label} (degree={degree})")
    analysis["god_nodes"] = gn_list

    # Surprising connections — cross-community bridge edges
    sc_list = surprising_connections(G)
    print("Top 3 Surprising Connections:")
    for sc in sc_list[:3]:
        src = sc.get('source_label', sc.get('source', '?'))
        tgt = sc.get('target_label', sc.get('target', '?'))
        score = sc.get('surprise_score', '?')
        print(f"  {src} -> {tgt} (score={score})")
    analysis["surprising_connections"] = sc_list

    # Suggested questions
    try:
        sq_list = suggest_questions(G)
        analysis["suggested_questions"] = sq_list
        print(f"Suggested questions: {len(sq_list)} generated")
    except Exception as e:
        print(f"suggest_questions skipped: {e}")

except Exception as e:
    print(f"graphify.analyze failed ({e}), computing manually")
    degree = dict(G.degree())
    top5 = sorted(degree.items(), key=lambda x: -x[1])[:5]
    print("Top 5 God Nodes by degree:")
    for nid, deg in top5:
        print(f"  {G.nodes[nid].get('label', nid)} deg={deg}")
    analysis["god_nodes"] = [
        {"id": nid, "label": G.nodes[nid].get('label', nid), "degree": deg}
        for nid, deg in top5
    ]
# ── STEP 6: Save outputs ───────────────────────────────────────────────────────
print("\n=== STEP 6: Saving outputs ===")

# community map
community_nodes = defaultdict(list)
for nid, ndata in G.nodes(data=True):
    cid = ndata.get('community', -1)
    community_nodes[cid].append(nid)

def get_cat(nid):
    sf = G.nodes[nid].get('source_file', '')
    parts = Path(sf).parts
    return parts[2] if len(parts) >= 3 else (parts[1] if len(parts) >= 2 else '')

community_map = {}
for cid, nids in community_nodes.items():
    cats = set(get_cat(n) for n in nids) - {''}
    community_map[str(cid)] = {
        "node_ids": nids,
        "categories": list(cats),
        "is_multi_category": len(cats) > 1,
        "size": len(nids)
    }

multi = sum(1 for v in community_map.values() if v['is_multi_category'])
print(f"Communities: {len(community_map)} total, {multi} multi-category")

# graph json
node_link = nx.node_link_data(G, edges='edges')
with open(OUT_DIR / "graph.json", "w") as f:
    json.dump(node_link, f, indent=2)

with open(OUT_DIR / "community_map.json", "w") as f:
    json.dump(community_map, f, indent=2)

with open(OUT_DIR / "analysis.json", "w") as f:
    json.dump(analysis, f, indent=2, default=str)

# Also save for pipeline
with open("data/processed/hierarchical_graph.json", "w") as f:
    json.dump(node_link, f, indent=2)

with open("data/processed/community_map.json", "w") as f:
    json.dump(community_map, f, indent=2)

# GRAPH_REPORT.md
report_lines = [
    "# Automotive Fault Knowledge Graph — Community Report\n",
    f"- Total nodes: {G.number_of_nodes()}",
    f"- Total edges: {G.number_of_edges()}",
    f"- Communities: {len(community_map)}\n"
]
for cid, v in sorted(community_map.items(), key=lambda x: -x[1]['size'])[:20]:
    cats = ', '.join(v['categories']) or 'Unknown'
    report_lines.append(f"## Community {cid}: {cats}")
    report_lines.append(f"- Node count: {v['size']}")
    report_lines.append(f"- Multi-category: {v['is_multi_category']}\n")

with open(OUT_DIR / "GRAPH_REPORT.md", "w") as f:
    f.write('\n'.join(report_lines))

print("Saved: graphify-out/graph.json")
print("Saved: graphify-out/community_map.json")
print("Saved: graphify-out/analysis.json")
print("Saved: graphify-out/GRAPH_REPORT.md")
print("Saved: data/processed/hierarchical_graph.json")
print("Saved: data/processed/community_map.json")
print("\n=== DONE ===")