import json
from pathlib import Path
from collections import defaultdict

print("=" * 60)
print("AUTOMOTIVE FAULT GRAPH — QUALITY ANALYSIS REPORT")
print("=" * 60)

# Load graph
with open('graphify-out/graph.json') as f:
    g = json.load(f)
with open('graphify-out/community_map.json') as f:
    cm = json.load(f)
with open('data/raw/automotive_faults_aktc_obike_et_al.json',
          encoding='utf-8-sig') as f:
    raw = json.load(f)

nodes = {n['id']: n for n in g.get('nodes', [])}
edges = g.get('links', g.get('edges', []))

# ── 1. COVERAGE ───────────────────────────────────────────────
print("\n── 1. COVERAGE ──────────────────────────────────────")
raw_cats    = set(r['category'] for r in raw)
raw_subcats = set(r['subcategory'] for r in raw)
raw_syms    = set(s for r in raw for s in r['symptoms'])
raw_steps   = set(d['step'] for r in raw
                  for d in r['diagnosis_steps'])

graph_labels = set(n.get('label','') for n in nodes.values())
found_cats    = raw_cats    & graph_labels
found_subcats = raw_subcats & graph_labels
found_syms    = raw_syms    & graph_labels
found_steps   = raw_steps   & graph_labels

print(f"Categories:      {len(found_cats)}/{len(raw_cats)} "
      f"({len(found_cats)/len(raw_cats)*100:.0f}%)")
print(f"Subcategories:   {len(found_subcats)}/{len(raw_subcats)} "
      f"({len(found_subcats)/len(raw_subcats)*100:.0f}%)")
print(f"Symptoms:        {len(found_syms)}/{len(raw_syms)} "
      f"({len(found_syms)/len(raw_syms)*100:.0f}%)")
print(f"Diagnosis Steps: {len(found_steps)}/{len(raw_steps)} "
      f"({len(found_steps)/len(raw_steps)*100:.0f}%)")

# ── 2. HIERARCHY INTEGRITY ────────────────────────────────────
print("\n── 2. HIERARCHY INTEGRITY ───────────────────────────")
rel_counts = defaultdict(int)
for e in edges:
    rel_counts[e.get('relation', 'unknown')] += 1

for rel, count in sorted(rel_counts.items()):
    print(f"  {rel}: {count}")

cats_with_subs  = set(e['source'] for e in edges
                      if e.get('relation') == 'HAS_SUBCATEGORY')
subs_with_syms  = set(e['source'] for e in edges
                      if e.get('relation') == 'HAS_SYMPTOM')
subs_with_steps = set(e['source'] for e in edges
                      if e.get('relation') == 'HAS_DIAGNOSIS_STEP')

print(f"\nCategories with subcategories: "
      f"{len(cats_with_subs)}/13")
print(f"Subcategories with symptoms:   "
      f"{len(subs_with_syms)}/98")
print(f"Subcategories with steps:      "
      f"{len(subs_with_steps)}/98")

# ── 3. COMMUNITY QUALITY ──────────────────────────────────────
print("\n── 3. COMMUNITY QUALITY ─────────────────────────────")
total_coms = len(cm)
multi_coms = sum(1 for v in cm.values()
                 if v.get('is_multi_category'))
sizes = [v['size'] for v in cm.values()]

print(f"Total communities: {total_coms}")
print(f"Multi-category:    {multi_coms}/{total_coms} "
      f"({multi_coms/total_coms*100:.1f}%)")
print(f"Avg community size: {sum(sizes)/len(sizes):.1f} nodes")
print(f"Largest community:  {max(sizes)} nodes")
print(f"Smallest community: {min(sizes)} nodes")
print(f"Anonymous (no name): "
      f"{sum(1 for v in cm.values() if not v.get('categories'))}")

# ── 4. CROSS-CATEGORY SIGNAL ──────────────────────────────────
print("\n── 4. CROSS-CATEGORY SIGNAL ─────────────────────────")
def get_cat(node):
    sf = node.get('source_file', '')
    parts = Path(sf).parts
    return parts[2] if len(parts) >= 3 else ''

cross = [(e, get_cat(nodes.get(e.get('source',''),{})),
             get_cat(nodes.get(e.get('target',''),{})))
         for e in edges]
cross_cat = [(e, s, t) for e, s, t in cross
             if s and t and s != t]

print(f"Cross-category edges: {len(cross_cat)}")
print(f"Edge-to-node ratio:   {len(edges)/len(nodes):.2f}")

# Top cross-category pairs
pair_counts = defaultdict(int)
for _, s, t in cross_cat:
    pair = tuple(sorted([s, t]))
    pair_counts[pair] += 1

print("\nTop 5 cross-category connections:")
for (s, t), count in sorted(pair_counts.items(),
                              key=lambda x: -x[1])[:5]:
    print(f"  {s} <-> {t}: {count} edges")

# ── 5. GOD NODES ──────────────────────────────────────────────
print("\n── 5. GOD NODES (highest connectivity) ──────────────")
degree = defaultdict(int)
for e in edges:
    degree[e.get('source','')] += 1
    degree[e.get('target','')] += 1

top10 = sorted(degree.items(), key=lambda x: -x[1])[:10]
for nid, deg in top10:
    nd = nodes.get(nid, {})
    label = nd.get('label', nid)
    cat = get_cat(nd)
    com = nd.get('community', '?')
    print(f"  deg={deg:3d} | com={com:2} | "
          f"{cat:25} | {label[:40]}")

# ── 6. DUPLICATE CHECK ────────────────────────────────────────
print("\n── 6. DUPLICATE NODE CHECK ──────────────────────────")
label_counts = defaultdict(int)
for n in nodes.values():
    label_counts[n.get('label','').lower().strip()] += 1
dups = {k: v for k, v in label_counts.items() if v > 1}
print(f"Duplicate labels: {len(dups)}")
if dups:
    print("Examples:")
    for label, count in list(dups.items())[:5]:
        print(f"  '{label}' appears {count} times")
else:
    print("No duplicate labels found — graph is clean")

# ── 7. PASS/FAIL SCORECARD ────────────────────────────────────
print("\n── 7. PASS/FAIL SCORECARD ───────────────────────────")
checks = {
    "100% category coverage":
        len(found_cats) == len(raw_cats),
    "100% subcategory coverage":
        len(found_subcats) == len(raw_subcats),
    ">80% symptom coverage":
        len(found_syms)/len(raw_syms) > 0.8,
    ">80% diagnosis step coverage":
        len(found_steps)/len(raw_steps) > 0.8,
    "All 13 categories have subcategories":
        len(cats_with_subs) == 13,
    "All 98 subcategories have symptoms":
        len(subs_with_syms) == 98,
    "All 98 subcategories have steps":
        len(subs_with_steps) == 98,
    ">10 communities":
        total_coms > 10,
    ">50% multi-category communities":
        multi_coms/total_coms > 0.5,
    ">100 cross-category edges":
        len(cross_cat) > 100,
    "No duplicate nodes":
        len(dups) == 0,
}

passed = sum(checks.values())
for check, result in checks.items():
    status = "✓ PASS" if result else "✗ FAIL"
    print(f"  {status} — {check}")

print(f"\nOVERALL: {passed}/{len(checks)} checks passed")
if passed == len(checks):
    print("★ GRAPH IS PRODUCTION READY")
elif passed >= 9:
    print("◆ GRAPH IS GOOD — minor issues only")
elif passed >= 7:
    print("◇ GRAPH IS ACCEPTABLE — review failures")
else:
    print("✗ GRAPH NEEDS WORK")