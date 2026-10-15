import json

with open('graphify-out/graph.json') as f:
    g = json.load(f)

nodes = g['nodes']

print("=== SAMPLE NODES BY TYPE ===\n")

for prefix, label in [('cat_', 'CATEGORY'), ('subcat_', 'SUBCATEGORY'),
                       ('sym_', 'SYMPTOM'), ('step_', 'DIAGNOSISSTEP')]:
    node = next((n for n in nodes if n.get('id','').startswith(prefix)), None)
    if node:
        print(f"--- {label} ---")
        print(json.dumps(node, indent=2))
        print()

print("=== FIELD COVERAGE ACROSS ALL NODES ===")
from collections import defaultdict
field_counts = defaultdict(int)
for n in nodes:
    for k in n.keys():
        field_counts[k] += 1

for field, count in sorted(field_counts.items()):
    pct = count/len(nodes)*100
    print(f"  {field}: {count}/{len(nodes)} ({pct:.0f}%)")