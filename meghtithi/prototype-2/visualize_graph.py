from neo4j import GraphDatabase
from pyvis.network import Network
from dotenv import load_dotenv
import os
import webbrowser

load_dotenv()

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
)

# Color and size per node type
NODE_STYLES = {
    "Symptom":      {"color": "#378ADD", "size": 28, "shape": "dot"},
    "FailureMode":  {"color": "#D85A30", "size": 22, "shape": "dot"},
    "Component":    {"color": "#EF9F27", "size": 18, "shape": "dot"},
    "RepairAction": {"color": "#1D9E75", "size": 16, "shape": "dot"},
}

EDGE_COLORS = {
    "INDICATES": "#378ADD",
    "AFFECTS":   "#D85A30",
    "REQUIRES":  "#1D9E75",
}

def build_graph_html(output_file="graph_viz.html"):
    net = Network(
        height="750px",
        width="100%",
        bgcolor="#1a1a2e",
        font_color="white",
        directed=True
    )

    net.set_options("""
    {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -60,
          "centralGravity": 0.005,
          "springLength": 120,
          "springConstant": 0.08
        },
        "solver": "forceAtlas2Based",
        "stabilization": { "iterations": 150 }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100,
        "navigationButtons": true,
        "keyboard": true
      },
      "edges": {
        "smooth": { "type": "curvedCW", "roundness": 0.2 },
        "arrows": { "to": { "enabled": true, "scaleFactor": 0.6 } },
        "font": { "size": 10, "color": "#aaaaaa" }
      },
      "nodes": {
        "font": { "size": 13, "face": "arial" },
        "borderWidth": 1.5
      }
    }
    """)

    added_nodes = set()

    with driver.session() as session:
        # Fetch all paths
        result = session.run("""
            MATCH (s:Symptom)-[i:INDICATES]->(f:FailureMode)
                  -[a:AFFECTS]->(c:Component)
                  -[r:REQUIRES]->(rep:RepairAction)
            RETURN s, f, c, rep,
                   i.confidence AS conf,
                   i.source AS source
        """)

        for record in result:
            nodes = {
                "Symptom":      record["s"],
                "FailureMode":  record["f"],
                "Component":    record["c"],
                "RepairAction": record["rep"],
            }

            prev_id = None
            edge_types = ["INDICATES", "AFFECTS", "REQUIRES"]
            node_keys = ["Symptom", "FailureMode", "Component", "RepairAction"]

            for idx, (label, node) in enumerate(nodes.items()):
                node_id = f"{label}:{node['name']}"
                style = NODE_STYLES[label]

                tooltip = f"<b>{label}</b><br>{node['name']}"
                if label == "Symptom" and record["source"]:
                    tooltip += f"<br><i>Source: {record['source']}</i>"
                if label == "FailureMode":
                    tooltip += f"<br>Confidence: {record['conf']:.0%}"

                if node_id not in added_nodes:
                    net.add_node(
                        node_id,
                        label=node["name"],
                        color=style["color"],
                        size=style["size"],
                        shape=style["shape"],
                        title=tooltip,
                        borderWidth=2,
                        borderWidthSelected=4,
                    )
                    added_nodes.add(node_id)

                if prev_id:
                    edge_label = edge_types[idx - 1]
                    edge_title = edge_label
                    if edge_label == "INDICATES":
                        edge_title = f"INDICATES ({record['conf']:.0%})"
                    net.add_edge(
                        prev_id, node_id,
                        label=edge_label,
                        color=EDGE_COLORS.get(edge_label, "#888888"),
                        title=edge_title,
                        width=2 if edge_label == "INDICATES" else 1,
                    )

                prev_id = node_id


    net.save_graph(output_file)
    print(f"✅ Graph visualization saved to: {output_file}")

    # Auto-open in browser
    abs_path = os.path.abspath(output_file)
    webbrowser.open(f"file://{abs_path}")
    print("🌐 Opening in browser...")

    driver.close()

if __name__ == "__main__":
    build_graph_html()
