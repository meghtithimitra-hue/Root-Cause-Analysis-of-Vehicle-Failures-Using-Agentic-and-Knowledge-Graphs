import json
import networkx as nx
from pathlib import Path

with open('graphify-out/graph.json') as f:
    data = json.load(f)

G = nx.node_link_graph(data, edges='edges', multigraph=True)

# Save graph.json path for tree_html (it takes file paths, not graph objects)
graph_path = Path('graphify-out/graph.json')
tree_out = Path('graphify-out/graph_tree.html')
callflow_out = Path('graphify-out/graph_callflow.html')

# Tree view — takes graph_path and output_path as Path objects
try:
    from graphify.tree_html import write_tree_html
    write_tree_html(
        graph_path=graph_path,
        output_path=tree_out,
        root=None,
        max_children=200,
        project_label="Automotive Fault Knowledge Graph",
        top_k_edges=0
    )
    print(f"Saved: {tree_out}")
except Exception as e:
    print(f"tree_html failed: {e}")

# Callflow/community view — takes file paths as strings
try:
    from graphify.callflow_html import write_callflow_html
    write_callflow_html(
        graphify_out='graphify-out',
        graph='graphify-out/graph.json',
        output=str(callflow_out),
        lang='auto',
        max_sections=15,
        max_diagram_nodes=18,
        max_diagram_edges=24,
        verbose=True
    )
    print(f"Saved: {callflow_out}")
except Exception as e:
    print(f"callflow_html failed: {e}")