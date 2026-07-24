import sys, os, types, json
sys.path.insert(0, 'scripts/pipeline')
os.environ['GEMINI_API_KEY'] = os.environ.get('GEMINI_API_KEY', '')
pkg = types.ModuleType('pipeline')
pkg.__path__ = [os.path.join(os.getcwd(), 'scripts', 'pipeline')]
sys.modules['pipeline'] = pkg
sv_pkg = types.ModuleType('sensor_validation')
sv_pkg.__path__ = [os.path.join(os.getcwd(), 'scripts', 'sensor_validation')]
sys.modules['sensor_validation'] = sv_pkg
from pipeline.hybrid_retrieval import _load_graph, _load_community_map, _stem
_, graph_nodes = _load_graph()
cm = _load_community_map()

# Find communities for target nodes
targets = ['Poor fuel economy', 'Rough idle', 'Engine hesitates on acceleration']
for t in targets:
    for nid, nd in graph_nodes.items():
        if nd.get('label', '') == t:
            cid = nd.get('community', -1)
            print('LABEL:', t)
            print('  node_id:', nid, 'community:', cid)
            for cid2, data in cm.items():
                if nid in data.get('node_ids', []):
                    print('  In community', cid2, 'cats:', data.get('categories', []))
                    other = [graph_nodes[n2]['label'] for n2 in data.get('node_ids', []) if n2 in graph_nodes]
                    print('  All', len(other), 'nodes:', other[:6])
            print()

# Test stemmer for hesitation variants
print('=== Stemmer test ===')
for w in ['hesitates', 'hesitation', 'hesitating', 'hesitant', 'hesit']:
    print(f'  {w} -> {_stem(w)}')
