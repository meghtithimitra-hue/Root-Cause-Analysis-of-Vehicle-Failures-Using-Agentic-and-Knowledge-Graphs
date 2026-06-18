"""
Leiden community detection for the automotive fault Neo4j graph.

Reads the existing graph from data, runs Leiden clustering (via graphify.cluster),
and outputs Cypher to create :Community nodes wired via :BELONGS_TO and :RELATES_TO.

Usage:
  python leiden_communities.py          # writes leiden_communities.cypher
  python leiden_communities.py --run    # also executes against Neo4j via py2neo
"""

import csv
import json
import os
import sys
from pathlib import Path

import networkx as nx
from graphify.cluster import cluster, score_all

INPUT_CSV = "triples.csv"
OUTPUT_CYPHER = "leiden_communities.cypher"
OUTPUT_JSON = Path("graphify-out") / "communities.json"

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "neo4j@123"

# --- Node label mapping (same as ingest scripts) ---
LABEL_MAP = {
    "HAS_COMPONENT": ("System", "Component"),
    "SHOWS_SYMPTOM": ("Component", "Symptom"),
    "DIAGNOSED_BY": ("Component", "DiagnosticTest"),
    "HAS_RESULT": ("DiagnosticTest", "Result"),
    "REQUIRES_FIX": ("Component", "RepairAction"),
}

# ── 1. Load triples ──────────────────────────────────────────────────────────
triples: list[tuple[str, str, str]] = []
with open(INPUT_CSV, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        triples.append((row["subject"], row["predicate"], row["object"]))

print(f"Loaded {len(triples)} triples")

# ── 2. Build undirected graph ────────────────────────────────────────────────
G = nx.Graph()
for subj, pred, obj in triples:
    G.add_node(subj)
    G.add_node(obj)
    G.add_edge(subj, obj, relation=pred)

print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# ── 3. Leiden clustering ─────────────────────────────────────────────────────
communities = cluster(G)
cohesion = score_all(G, communities)

# Label communities by their most connected System-like member
node_types: dict[str, str] = {}
for s, p, o in triples:
    sl, ol = LABEL_MAP[p]
    node_types.setdefault(s, sl)
    node_types.setdefault(o, ol)

system_nodes = {n for n, t in node_types.items() if t == "System"}

community_labels: dict[int, str] = {}
for cid, members in communities.items():
    # Find a System node in the community, if any
    sys_hit = next((m for m in members if m in system_nodes), None)
    if sys_hit:
        community_labels[cid] = f"{sys_hit}"
    else:
        # Derive from dominant node type
        counts: dict[str, int] = {}
        for m in members:
            t = node_types.get(m, "Unknown")
            counts[t] = counts.get(t, 0) + 1
        dom = max(counts, key=counts.get)
        community_labels[cid] = f"{dom} Cluster"

print(f"\nLeiden detected {len(communities)} communities")
for cid in sorted(communities, key=lambda c: -len(communities[c])):
    print(f"  [{cid:2d}] {community_labels[cid]:40s}  {len(communities[cid]):3d} nodes  cohesion={cohesion.get(cid,0):.3f}")

# Save communities JSON (mirrors graphify-out format)
os.makedirs(OUTPUT_JSON.parent, exist_ok=True)
comm_data = {}
for cid, members in communities.items():
    comm_data[str(cid)] = {
        "label": community_labels.get(cid, f"Community {cid}"),
        "cohesion": round(cohesion.get(cid, 0.0), 4),
        "size": len(members),
        "members": sorted(members),
    }
OUTPUT_JSON.write_text(json.dumps(comm_data, indent=2), encoding="utf-8")
print(f"\nCommunities saved -> {OUTPUT_JSON}")

# ── 4. Build inter-community edge counts ─────────────────────────────────────
node_to_community: dict[str, int] = {}
for cid, members in communities.items():
    for m in members:
        node_to_community[m] = cid

inter_comm: dict[tuple[int, int], int] = {}
for u, v in G.edges():
    cu = node_to_community.get(u)
    cv = node_to_community.get(v)
    if cu is not None and cv is not None and cu != cv:
        a, b = (cu, cv) if cu < cv else (cv, cu)
        inter_comm[(a, b)] = inter_comm.get((a, b), 0) + 1

# ── 5. Generate Cypher ───────────────────────────────────────────────────────
lines = [
    "// ─────────────────────────────────────────────────────────────────────",
    "// Leiden communities — automotive fault knowledge graph",
    f"// Generated: {len(communities)} communities, {G.number_of_nodes()} nodes",
    "// ─────────────────────────────────────────────────────────────────────",
    "",
    "// Constraints",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Community) REQUIRE c.community_id IS UNIQUE;",
    "",
    "// Clear previous community run (optional — comment out to keep)",
    "MATCH (c:Community) DETACH DELETE c;",
    "",
    "// ── Community nodes ───────────────────────────────────────────────────",
]

# Community nodes (sorted by size descending)
sorted_cids = sorted(communities.keys(), key=lambda c: -len(communities[c]))
for cid in sorted_cids:
    name = community_labels.get(cid, f"Community {cid}")
    esc_name = name.replace("'", "\\'")
    sz = len(communities[cid])
    coh = cohesion.get(cid, 0.0)
    lines.append(
        f"MERGE (c{cid}:Community {{community_id: {cid}}}) "
        f"SET c{cid}.name = '{esc_name}', "
        f"c{cid}.size = {sz}, "
        f"c{cid}.cohesion = {coh:.4f};"
    )

lines += [
    "",
    "// ── BELONGS_TO relationships ──────────────────────────────────────────",
]

# BELONGS_TO from each node to its community
for cid, members in communities.items():
    for node_id in sorted(members):
        esc_node = node_id.replace("'", "\\'")
        lines.append(
            f"MATCH (n {{name: '{esc_node}'}}), "
            f"(c:Community {{community_id: {cid}}}) "
            f"WHERE NOT n:Community "
            f"MERGE (n)-[:BELONGS_TO]->(c);"
        )

lines += [
    "",
    "// ── RELATES_TO (cross-community) relationships ───────────────────────",
]

for (a, b), weight in sorted(inter_comm.items(), key=lambda x: -x[1]):
    lines.append(
        f"MATCH (ca:Community {{community_id: {a}}}), "
        f"(cb:Community {{community_id: {b}}}) "
        f"MERGE (ca)-[:RELATES_TO {{weight: {weight}}}]->(cb);"
    )

lines += [
    "",
    "// ── Indexes for fast lookups ─────────────────────────────────────────",
    "CREATE INDEX IF NOT EXISTS FOR (n:Community) ON (n.community_id);",
    "",
    "// Done.",
]

cypher = "\n".join(lines)
Path(OUTPUT_CYPHER).write_text(cypher, encoding="utf-8")
print(f"Cypher written -> {OUTPUT_CYPHER} ({len(lines)} lines)")

# ── 6. Optionally execute against Neo4j ──────────────────────────────────────
if "--run" in sys.argv:
    try:
        from py2neo import Graph as NeoGraph
        ng = NeoGraph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
        # Execute in batches
        batch: list[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("//"):
                continue
            batch.append(line)
            if stripped.endswith(";"):
                try:
                    ng.run("\n".join(batch))
                except Exception as e:
                    print(f"  Error on statement: {e}", file=sys.stderr)
                batch = []
        print("Executed against Neo4j successfully")
    except Exception as e:
        print(f"Could not connect to Neo4j: {e}", file=sys.stderr)
        print("Cypher file was written — run manually against Neo4j")
