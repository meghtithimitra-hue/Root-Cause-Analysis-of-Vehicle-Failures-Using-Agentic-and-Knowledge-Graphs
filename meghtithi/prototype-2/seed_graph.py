from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

load_dotenv()

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
)

AUTOMOTIVE_DATA = [
    # (symptom, failure_mode, component, repair_action, confidence)
    ("engine won't start", "dead battery",        "battery",         "jump start or replace battery",     0.85),
    ("engine won't start", "failed starter motor","starter motor",   "replace starter motor",             0.75),
    ("engine won't start", "fuel starvation",     "fuel pump",       "replace fuel pump",                 0.70),
    ("engine won't start", "clogged fuel filter", "fuel filter",     "replace fuel filter",               0.60),
    ("engine won't start", "faulty crankshaft sensor","crankshaft position sensor","replace CKP sensor", 0.65),
    ("engine won't start", "flooded engine",      "carburetor",      "clear flood, crank without throttle",0.50),
    ("slow crank",         "weak battery",        "battery",         "recharge or replace battery",       0.80),
    ("slow crank",         "corroded terminals",  "battery terminals","clean or replace terminals",       0.70),
    ("no crank",           "failed starter relay","starter relay",   "replace starter relay",             0.80),
    ("no crank",           "blown fuse",          "fuse box",        "inspect and replace fuses",         0.75),
    ("no crank",           "dead battery",        "battery",         "jump start or replace battery",     0.90),
    ("engine misfires",    "worn spark plugs",    "spark plugs",     "replace spark plugs",               0.85),
    ("engine misfires",    "faulty ignition coil","ignition coil",   "replace ignition coil",             0.75),
    ("engine misfires",    "vacuum leak",         "intake manifold", "inspect and seal vacuum leaks",     0.60),
    ("engine overheats",   "coolant leak",        "radiator hose",   "replace hose, refill coolant",      0.80),
    ("engine overheats",   "failed water pump",   "water pump",      "replace water pump",                0.75),
    ("engine overheats",   "stuck thermostat",    "thermostat",      "replace thermostat",                0.70),
    ("check engine light", "O2 sensor fault",     "oxygen sensor",   "replace O2 sensor",                 0.70),
    ("check engine light", "loose gas cap",       "gas cap",         "tighten or replace gas cap",        0.60),
    ("battery drains overnight","parasitic draw", "alternator",      "test and replace alternator",       0.75),
    ("battery drains overnight","faulty relay",   "relay module",    "identify and replace faulty relay", 0.65),
]

def seed(tx):
    tx.run("MATCH (n) DETACH DELETE n")  # clear first

    for symptom, failure, component, repair, conf in AUTOMOTIVE_DATA:
        tx.run("""
            MERGE (s:Symptom {name: $symptom})
            MERGE (f:FailureMode {name: $failure})
            MERGE (c:Component {name: $component})
            MERGE (r:RepairAction {name: $repair})
            MERGE (s)-[:INDICATES {confidence: $conf}]->(f)
            MERGE (f)-[:AFFECTS]->(c)
            MERGE (c)-[:REQUIRES]->(r)
        """, symptom=symptom, failure=failure,
             component=component, repair=repair, conf=conf)

with driver.session() as session:
    session.execute_write(seed)
    print("✅ Knowledge graph seeded successfully")
    result = session.run("MATCH (n) RETURN labels(n)[0] as type, count(n) as count")
    for r in result:
        print(f"   {r['type']}: {r['count']} nodes")

driver.close()