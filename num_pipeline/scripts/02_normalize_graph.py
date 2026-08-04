import json
import re
from pathlib import Path
from collections import defaultdict

print("=== NORMALIZING GRAPH ===")

with open('graphify-out/graph.json') as f:
    g = json.load(f)
with open('data/automotive_faults_aktc_obike_et_al.json',
          encoding='utf-8-sig') as f:
    raw = json.load(f)

# Build lookup: label -> (category, subcategory) from raw JSON
label_to_meta = {}
for r in raw:
    cat = r['category']
    subcat = r['subcategory']
    label_to_meta[subcat.lower()] = (cat, subcat)
    for sym in r['symptoms']:
        label_to_meta[sym.lower()] = (cat, subcat)
    for d in r['diagnosis_steps']:
        label_to_meta[d['step'].lower()] = (cat, subcat)

def get_node_type(nid):
    if nid.startswith('cat_'):    return 'Category'
    if nid.startswith('subcat_'): return 'Subcategory'
    if nid.startswith('sym_'):    return 'Symptom'
    if nid.startswith('step_'):   return 'DiagnosisStep'
    return 'Unknown'

def get_cat_from_source(source_file):
    parts = Path(source_file).parts
    if len(parts) >= 3:
        return parts[2].replace('_', ' ').title()
    return ''

def get_subcat_from_source(source_file):
    parts = Path(source_file).parts
    if len(parts) >= 4:
        return parts[3].replace('_', ' ').title()
    return ''

# Add missing fields to every node
nodes = g['nodes']
for node in nodes:
    nid = node.get('id', '')
    label = node.get('label', '')
    sf = node.get('source_file', '')

    # Add node_type
    node['node_type'] = get_node_type(nid)

    # Add category and subcategory
    meta = label_to_meta.get(label.lower())
    if meta:
        node['category'] = meta[0]
        node['subcategory'] = meta[1] if node['node_type'] != 'Category' else ''
    else:
        node['category'] = get_cat_from_source(sf)
        node['subcategory'] = get_subcat_from_source(sf) \
            if node['node_type'] in ('Symptom', 'DiagnosisStep') else ''

# Verify
type_counts = defaultdict(int)
has_cat = 0
has_subcat = 0
for n in nodes:
    type_counts[n['node_type']] += 1
    if n.get('category'): has_cat += 1
    if n.get('subcategory'): has_subcat += 1

print("Node types added:")
for t, c in sorted(type_counts.items()):
    print(f"  {t}: {c}")
print(f"Nodes with category: {has_cat}/{len(nodes)}")
print(f"Nodes with subcategory: {has_subcat}/{len(nodes)}")

# Save normalized graph
with open('data/processed/hierarchical_graph.json', 'w') as f:
    json.dump(g, f, indent=2)

print("\nSaved: data/processed/hierarchical_graph.json")
print("Done.")