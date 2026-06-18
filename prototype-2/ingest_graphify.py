"""
Graphify ingestion pipeline for automotive fault triples.

Reads triples.csv, builds a NetworkX graph, runs Leiden/Louvain community
detection, and exports graph.json, GRAPH_REPORT.md, and graph.html in the
standard Graphify output format.
"""

import csv
import json
import os
import sys
from pathlib import Path

import networkx as nx
from networkx.readwrite import json_graph

from graphify.cluster import cluster, score_all
from graphify.export import to_json, to_html
from graphify.report import generate as generate_report
from graphify.analyze import god_nodes, surprising_connections, suggest_questions

INPUT_CSV = "triples.csv"
OUT_DIR = Path("graphify-out")

# Label mapping for node types
LABEL_TYPE = {
    "System": "system",
    "Component": "component",
    "Symptom": "symptom",
    "DiagnosticTest": "diagnostic_test",
    "Result": "result",
    "RepairAction": "repair_action",
}

# Predicate → (subject_label, object_label)
LABEL_MAP = {
    "HAS_COMPONENT": ("System", "Component"),
    "SHOWS_SYMPTOM": ("Component", "Symptom"),
    "DIAGNOSED_BY": ("Component", "DiagnosticTest"),
    "HAS_RESULT": ("DiagnosticTest", "Result"),
    "REQUIRES_FIX": ("Component", "RepairAction"),
}

# ---------------------------------------------------------------------------
# 1. Read triples
# ---------------------------------------------------------------------------
triples: list[tuple[str, str, str]] = []
with open(INPUT_CSV, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        triples.append((row["subject"], row["predicate"], row["object"]))

print(f"Loaded {len(triples)} triples from {INPUT_CSV}")

# ---------------------------------------------------------------------------
# 2. Build node metadata (label → type)
# ---------------------------------------------------------------------------
node_labels: dict[str, str] = {}

for subj, pred, obj in triples:
    s_label, o_label = LABEL_MAP[pred]
    node_labels.setdefault(subj, s_label)
    node_labels.setdefault(obj, o_label)

# ---------------------------------------------------------------------------
# 3. Build NetworkX multi-digraph (preserve edge direction)
# ---------------------------------------------------------------------------
G = nx.MultiDiGraph()

for node_id, label in node_labels.items():
    G.add_node(node_id, label=node_id, file_type=LABEL_TYPE[label],
               source_file="automotive_faults_aktc_obike_et_al.json")

for subj, pred, obj in triples:
    G.add_edge(subj, obj, key=pred, relation=pred, confidence="EXTRACTED",
               _src=subj, _tgt=obj)

print(f"Built graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# ---------------------------------------------------------------------------
# 4. Community detection (Leiden via graphify, falls back to Louvain)
# ---------------------------------------------------------------------------
communities = cluster(G)
cohesion = score_all(G, communities)

# Label communities by their most common system-derived type
COMMUNITY_LABEL_SOURCES = {
    cid: next((n for n in members if node_labels.get(n) == "System"), None)
    for cid, members in communities.items()
}
community_labels: dict[int, str] = {
    cid: (src if src else f"Community {cid}")
    for cid, src in COMMUNITY_LABEL_SOURCES.items()
}

# Derive descriptive names for system-less communities from member types
_node_type = lambda nid: node_labels.get(nid, "unknown")
for cid, members in communities.items():
    if community_labels[cid].startswith("Community"):
        type_counts: dict[str, int] = {}
        for m in members:
            t = _node_type(m)
            type_counts[t] = type_counts.get(t, 0) + 1
        dominant = max(type_counts, key=type_counts.get)
        community_labels[cid] = f"{dominant.capitalize()} Cluster ({cid})"

print(f"Detected {len(communities)} communities")
for cid in sorted(communities, key=lambda c: -len(communities[c])):
    print(f"  Community {cid}: '{community_labels[cid]}' — {len(communities[cid])} nodes, cohesion {cohesion.get(cid, 0):.3f}")

# ---------------------------------------------------------------------------
# 5. Analysis
# ---------------------------------------------------------------------------
god_node_list = god_nodes(G, top_n=15)
print(f"\nTop god nodes:")
for n in god_node_list:
    print(f"  {n['label']} — {n['degree']} edges")

surprise_list = surprising_connections(G, communities, top_n=5)
print(f"\nSurprising connections: {len(surprise_list)}")

questions = suggest_questions(G, communities, community_labels, top_n=7)
print(f"Suggested questions: {len(questions)}")

# ---------------------------------------------------------------------------
# 6. Export
# ---------------------------------------------------------------------------
os.makedirs(OUT_DIR, exist_ok=True)

# 6a. graph.json (node-link format with community assignment)
json_path = str(OUT_DIR / "graph.json")
to_json(G, communities, json_path, force=True)
print(f"\nExported {json_path}")

# 6b. GRAPH_REPORT.md
detection_result = {
    "total_files": 1,
    "total_words": sum(len(s) + len(o) for _, _, (s, o) in [(t, t, (t[0], t[2])) for t in triples]),
}
report_md = generate_report(
    G=G,
    communities=communities,
    cohesion_scores=cohesion,
    community_labels=community_labels,
    god_node_list=god_node_list,
    surprise_list=surprise_list,
    detection_result=detection_result,
    token_cost={"input": 0, "output": 0},
    root="auto-vehi-ai",
    suggested_questions=questions,
    min_community_size=2,
)
report_path = OUT_DIR / "GRAPH_REPORT.md"
report_path.write_text(report_md, encoding="utf-8")
print(f"Exported {report_path}")

# 6c. graph.html (interactive vis.js visualization)
html_path = str(OUT_DIR / "graph.html")
node_counts = {cid: len(members) for cid, members in communities.items()}
to_html(G, communities, html_path, community_labels=community_labels,
        member_counts=node_counts)
print(f"Exported {html_path}")

# 6d. Export communities JSON for external consumption
comm_data = {}
for cid, members in communities.items():
    comm_data[str(cid)] = {
        "label": community_labels.get(cid, f"Community {cid}"),
        "cohesion": cohesion.get(cid, 0.0),
        "size": len(members),
        "members": sorted(members),
    }
comm_path = OUT_DIR / "communities.json"
comm_path.write_text(json.dumps(comm_data, indent=2), encoding="utf-8")
print(f"Exported {comm_path}")

# Summary
print(f"\n{'='*50}")
print(f"Ingestion complete. Output in {OUT_DIR}/")
print(f"  graph.json         — queryable knowledge graph")
print(f"  GRAPH_REPORT.md    — audit report")
print(f"  graph.html         — interactive visualization")
print(f"  communities.json   — community listings")
