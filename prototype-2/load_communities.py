"""Load leiden_communities.cypher into Neo4j."""

import sys
from neo4j import GraphDatabase

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "neo4j@123"
CYPHER_FILE = "leiden_communities.cypher"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

# Verify
with driver.session(database="neo4j") as session:
    session.run("RETURN 1")
print(f"Connected to Neo4j at {NEO4J_URI}")

# Read and parse statements
lines = open(CYPHER_FILE, encoding="utf-8").readlines()
statements = []
current = []
for line in lines:
    s = line.strip()
    if not s or s.startswith("//"):
        continue
    current.append(line)
    if s.endswith(";"):
        statements.append("".join(current))
        current.clear()
print(f"Parsed {len(statements)} statements")

# Execute
counts = {"community": 0, "belongs_to": 0, "relates_to": 0}
with driver.session(database="neo4j") as session:
    for i, stmt in enumerate(statements):
        try:
            session.run(stmt)
            if "CREATE (c" in stmt and ":Community" in stmt:
                counts["community"] += 1
            elif "BELONGS_TO" in stmt:
                counts["belongs_to"] += 1
            elif "RELATES_TO" in stmt:
                counts["relates_to"] += 1
        except Exception as e:
            print(f"  Error [{i}]: {e}", file=sys.stderr)

        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{len(statements)}", end="\r")

print(f"\nDone: {counts['community']} communities, "
      f"{counts['belongs_to']} BELONGS_TO, "
      f"{counts['relates_to']} RELATES_TO")

# Verify
with driver.session(database="neo4j") as session:
    row = session.run(
        "MATCH (c:Community) RETURN count(c) AS communities, "
        "count { MATCH ()-[r:BELONGS_TO]->() } AS belongs_to, "
        "count { MATCH ()-[r:RELATES_TO]->() } AS relates_to"
    ).single()
    print(f"\nVerification:")
    print(f"  (:Community)              {row['communities']}")
    print(f"  (:Node)-[:BELONGS_TO]->() {row['belongs_to']}")
    print(f"  ()-[:RELATES_TO]->()      {row['relates_to']}")

driver.close()
