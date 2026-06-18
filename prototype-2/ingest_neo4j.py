import csv
from py2neo import Graph, Node, Relationship

INPUT = "triples.csv"
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "neo4j@123"

graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# ---------------------------------------------------------------------------
# 1. Constraints & indexes
# ---------------------------------------------------------------------------
for label in ("System", "Component", "Symptom", "DiagnosticTest", "Result", "RepairAction"):
    graph.run(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.name IS UNIQUE")
    graph.run(f"CREATE INDEX IF NOT EXISTS FOR (n:{label}) ON (n.name)")

# ---------------------------------------------------------------------------
# 2. Read triples
# ---------------------------------------------------------------------------
LABEL_MAP = {
    "HAS_COMPONENT": ("System", "Component"),
    "SHOWS_SYMPTOM": ("Component", "Symptom"),
    "DIAGNOSED_BY": ("Component", "DiagnosticTest"),
    "HAS_RESULT": ("DiagnosticTest", "Result"),
    "REQUIRES_FIX": ("Component", "RepairAction"),
}

triples = []
with open(INPUT, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        triples.append((row["subject"], row["predicate"], row["object"]))

# ---------------------------------------------------------------------------
# 3. Merge nodes (idempotent)
# ---------------------------------------------------------------------------
node_cache = {}

def get_node(name, label):
    key = (name, label)
    if key not in node_cache:
        n = Node(label, name=name)
        graph.merge(n, label, "name")
        node_cache[key] = n
    return node_cache[key]

for subj, pred, obj in triples:
    s_label, o_label = LABEL_MAP[pred]
    get_node(subj, s_label)
    get_node(obj, o_label)

# ---------------------------------------------------------------------------
# 4. Create relationships
# ---------------------------------------------------------------------------
tx = graph.begin()
for subj, pred, obj in triples:
    s_label, o_label = LABEL_MAP[pred]
    s_node = node_cache[(subj, s_label)]
    o_node = node_cache[(obj, o_label)]
    tx.create(Relationship(s_node, pred, o_node))
tx.commit()

print(f"Ingested {len(triples)} triples into Neo4j ({NEO4J_URI})")
