import json
import networkx as nx
from pyvis.network import Network

# Load triples
with open("graph/triples.json", "r", encoding="utf-8") as f:
    triples = json.load(f)

# Create graph
G = nx.DiGraph()

# Main root
ROOT = "Automotive System"
G.add_node(ROOT, level=0, color="gold", size=40)

# Hierarchical categories
systems = {
    "Engine System": [
        "engine", "piston", "cylinder", "spark", "combustion",
        "oil", "fuel", "injector", "valve", "coolant"
    ],

    "Cooling System": [
        "radiator", "cooling", "coolant", "fan", "thermostat",
        "overheating", "water pump"
    ],

    "Electrical System": [
        "battery", "alternator", "voltage", "wiring",
        "sensor", "electrical", "starter"
    ],

    "Brake System": [
        "brake", "disc", "pad", "abs", "hydraulic"
    ],

    "Transmission System": [
        "gear", "gearbox", "clutch", "transmission"
    ]
}

# Add system nodes
for system in systems:
    G.add_node(system,
               level=1,
               color="orange",
               size=30)

    G.add_edge(ROOT, system)

# Add entity relationships
for triple in triples:

    subject = triple["subject"].lower()
    relation = triple["relation"].lower()
    obj = triple["object"].lower()

    # Skip noisy entities
    if len(subject) < 3 or len(obj) < 3:
        continue

    if subject == obj:
        continue

    # Assign subject to category
    subject_system = None

    for system, keywords in systems.items():
        if any(k in subject for k in keywords):
            subject_system = system
            break

    if subject_system is None:
        subject_system = "General"

        if not G.has_node("General"):
            G.add_node("General",
                       level=1,
                       color="orange",
                       size=30)

            G.add_edge(ROOT, "General")

    # Add subject node
    G.add_node(subject,
               level=2,
               color="skyblue",
               size=18)

    # Connect subject to system
    G.add_edge(subject_system, subject)

    # Add object node
    G.add_node(obj,
               level=3,
               color="lightgreen",
               size=15)

    # Add relationship
    G.add_edge(subject, obj, title=relation)

# Create interactive visualization
net = Network(
    height="900px",
    width="100%",
    bgcolor="#111111",
    font_color="white",
    directed=True
)

# Hierarchical layout
net.set_options("""
var options = {
  "layout": {
    "hierarchical": {
      "enabled": true,
      "direction": "UD",
      "sortMethod": "directed",
      "nodeSpacing": 180,
      "treeSpacing": 220,
      "levelSeparation": 180
    }
  },
  "physics": {
    "enabled": false
  },
  "edges": {
    "color": {
      "color": "#aaaaaa"
    },
    "smooth": true,
    "arrows": {
      "to": {
        "enabled": true
      }
    }
  }
}
""")

net.from_nx(G)

# Save graph
net.save_graph("graph/hierarchical_graph.html")

print("Hierarchical knowledge graph created!")
print("Open graph/hierarchical_graph.html")