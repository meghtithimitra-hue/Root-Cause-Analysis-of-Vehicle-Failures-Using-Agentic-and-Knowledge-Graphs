from py2neo import Graph

g = Graph("bolt://localhost:7687", auth=("neo4j", "neo4j@123"))

print("=== Node counts ===")
result = g.run("MATCH (n) RETURN labels(n)[0] AS label, count(*) AS cnt ORDER BY cnt DESC")
for r in result:
    print(f"  {r['label']:20s} {r['cnt']}")

print("\n=== Relationship counts ===")
result = g.run("MATCH ()-[r]->() RETURN type(r) AS rel, count(*) AS cnt ORDER BY cnt DESC")
for r in result:
    print(f"  {r['rel']:20s} {r['cnt']}")
