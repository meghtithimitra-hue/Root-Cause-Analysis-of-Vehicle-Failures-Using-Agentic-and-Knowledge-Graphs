from neo4j import GraphDatabase
d = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "neo4j@123"))
with d.session(database="neo4j") as s:
    r = s.run("MATCH (n)-[:BELONGS_TO]->(c:Community) RETURN c.name, count(n) AS sz ORDER BY sz DESC LIMIT 5")
    for row in r:
        print(f"  {row['c.name']}: {row['sz']} nodes")
    r2 = s.run("MATCH (n)-[:BELONGS_TO]->() WHERE NOT n:Community RETURN count(n)")
    print(f"Non-Community with BELONGS_TO: {r2.single()[0]}")
    r3 = s.run("MATCH (n:Community)-[:BELONGS_TO]->() RETURN count(n)")
    print(f"Community with outgoing BELONGS_TO: {r3.single()[0]}")
    r4 = s.run("MATCH p = ()-[:BELONGS_TO]->() RETURN count(p)")
    print(f"Total BELONGS_TO relationships: {r4.single()[0]}")
d.close()
