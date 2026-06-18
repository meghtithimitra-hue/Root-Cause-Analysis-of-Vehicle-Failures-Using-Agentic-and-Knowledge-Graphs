from neo4j import GraphDatabase
d = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "neo4j@123"))
with d.session(database="neo4j") as s:
    r = s.run("MATCH (c:Community) RETURN count(c)")
    print(f"Communities: {r.single()[0]}")
    r = s.run("MATCH ()-[r:BELONGS_TO]->() RETURN count(r)")
    print(f"BELONGS_TO: {r.single()[0]}")
    r = s.run("MATCH ()-[r:RELATES_TO]->() RETURN count(r)")
    print(f"RELATES_TO: {r.single()[0]}")
    print()
    print("Top 5 communities:")
    r = s.run("MATCH (c:Community) RETURN c.name, c.size ORDER BY c.size DESC LIMIT 5")
    for row in r:
        print(f"  {row['c.name']}: {row['c.size']} nodes")
    r = s.run("MATCH (n)-[r:BELONGS_TO]->() WITH n, count(r) AS cnt WHERE cnt > 1 RETURN n.name, cnt LIMIT 5")
    dupes = list(r)
    if dupes:
        print()
        print("Nodes with multiple BELONGS_TO:")
        for row in dupes:
            print(f"  {row['n.name']}: {row['cnt']}")
    else:
        print()
        print("No nodes with multiple BELONGS_TO (clean)")
d.close()
