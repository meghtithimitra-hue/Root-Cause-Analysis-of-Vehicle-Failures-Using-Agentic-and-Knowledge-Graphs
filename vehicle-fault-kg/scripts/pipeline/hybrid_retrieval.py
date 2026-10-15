import json
from collections import defaultdict, deque
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
import networkx as nx

# ── Paths ──────────────────────────────────────────────────────
CHROMA_DIR = "data/chroma_db"
COLLECTION_NAME = "automotive_kg"
MODEL_NAME = "all-MiniLM-L6-v2"
GRAPH_PATH = "data/processed/hierarchical_graph.json"
COMMUNITY_MAP_PATH = "graphify-out/community_map.json"

# ── Lazy-loaded singletons ─────────────────────────────────────
_model = None
_collection = None
_graph_nx = None
_graph_nodes = None
_community_map = None

def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model

def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection

def _load_graph():
    global _graph_nx, _graph_nodes
    if _graph_nx is None:
        with open(GRAPH_PATH) as f:
            data = json.load(f)
        _graph_nx = nx.node_link_graph(
            data, edges='edges', multigraph=True)
        _graph_nodes = {n['id']: n for n in data['nodes']}
    return _graph_nx, _graph_nodes

def _load_community_map():
    global _community_map
    if _community_map is None:
        with open(COMMUNITY_MAP_PATH) as f:
            _community_map = json.load(f)
    return _community_map

# ── Query Embedding ────────────────────────────────────────────
def embed_query(query: str) -> list:
    """Convert query text into embedding vector."""
    return _get_model().encode(query).tolist()

# ── PATH 1: Vector Search ──────────────────────────────────────
def vector_search(query_embedding: list, top_k: int = 10) -> list:
    """
    Semantic similarity search via ChromaDB.
    Finds nodes whose embedding is closest in meaning to the query.
    Returns nodes ranked by cosine similarity.
    """
    raw = _get_collection().query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=['metadatas', 'distances', 'documents']
    )
    results = []
    for meta, dist, doc in zip(
        raw['metadatas'][0],
        raw['distances'][0],
        raw['documents'][0]
    ):
        results.append({
            'node_id':          meta['node_id'],
            'node_type':        meta['node_type'],
            'label':            meta.get('label', ''),
            'category':         meta.get('category', ''),
            'subcategory':      meta.get('subcategory', ''),
            'community_id':     meta.get('community_id', '-1'),
            'is_multi_category': meta.get('is_multi_category', 'False'),
            'embedding_text':   doc,
            'distance':         dist,
            'raw_score':        round(1.0 - dist, 4)
        })
    return results

# ── PATH 2: Graph Search ───────────────────────────────────────
def graph_search(query: str, max_hops: int = 2,
                 top_k: int = 30) -> list:
    """
    Structural graph traversal via BFS.
    Finds seed nodes by substring match on labels,
    then expands outward up to max_hops via graph edges.
    Captures hierarchy context (parent categories,
    sibling subcategories, related symptoms).
    """
    G, graph_nodes = _load_graph()
    words = [w for w in query.lower().split() if len(w) > 2]
    if not words:
        return []

    # Find seed nodes
    seeds = set()
    for nid, nd in graph_nodes.items():
        label = nd.get('label', '').lower()
        if any(w in label for w in words):
            seeds.add(nid)

    if not seeds:
        return []

    # BFS expansion
    visited = {s: 0 for s in seeds}
    queue = deque((s, 0) for s in seeds)
    while queue:
        cur, dist = queue.popleft()
        if dist >= max_hops:
            continue
        for nb in G.neighbors(cur):
            if nb not in visited:
                visited[nb] = dist + 1
                queue.append((nb, dist + 1))

    # Build results
    words_set = set(words)
    results = []
    for nid, hop in visited.items():
        nd = graph_nodes.get(nid, {})
        label = nd.get('label', nid)
        label_words = set(label.lower().split())
        overlap = len(words_set & label_words)
        results.append({
            'node_id':          nid,
            'node_type':        nd.get('node_type', 'Unknown'),
            'label':            label,
            'category':         nd.get('category', ''),
            'subcategory':      nd.get('subcategory', ''),
            'community_id':     str(nd.get('community', '-1')),
            'is_multi_category': 'False',
            'hop_distance':     hop,
            'word_overlap':     overlap,
            'raw_score':        round(max(0, 1.0 - (hop * 0.3)
                                         + (overlap * 0.1)), 4)
        })

    # Sort: direct matches first, then by hop distance
    results.sort(key=lambda r: (
        r['hop_distance'], -r['word_overlap'], r['label']
    ))
    return results[:top_k]

# ── PATH 3: Community Search ───────────────────────────────────
def community_search(top_vector_results: list,
                     top_k: int = 10) -> list:
    """
    Community-aware retrieval using Leiden/Louvain clusters.
    Takes the top vector results, finds their dominant community,
    then returns ALL nodes in that community as candidates.
    Multi-category communities (spanning 2+ vehicle systems)
    get a higher base score — these represent genuine
    cross-system fault relationships.
    """
    if not top_vector_results:
        return []

    cm = _load_community_map()
    _, graph_nodes = _load_graph()

    # Vote on best community from top 3 vector hits
    top3_cids = [r['community_id']
                 for r in top_vector_results[:3]]
    freq = defaultdict(int)
    for c in top3_cids:
        freq[c] += 1
    best_cid = max(freq, key=freq.get)

    community_data = cm.get(str(best_cid), {})
    is_multi = community_data.get('is_multi_category', False)

    # Multi-category communities get higher base score
    # because they represent cross-system fault patterns
    base_score = 0.6 if is_multi else 0.4
    cats = community_data.get('categories', [])

    results = []
    for nid in community_data.get('node_ids', []):
        nd = graph_nodes.get(nid, {})
        ntype = nd.get('node_type', '')
        if not ntype or ntype == 'Unknown':
            continue
        results.append({
            'node_id':          nid,
            'node_type':        ntype,
            'label':            nd.get('label', nid),
            'category':         nd.get('category', ''),
            'subcategory':      nd.get('subcategory', ''),
            'community_id':     str(best_cid),
            'is_multi_category': str(is_multi),
            'community_categories': cats,
            'raw_score':        base_score
        })

    return results[:top_k]

# ── HYBRID MERGE ───────────────────────────────────────────────
def hybrid_retrieve(query: str, top_k: int = 10) -> dict:
    """
    Main retrieval function combining all 3 paths.

    Scoring logic:
    - vector only:           score = 1.0 - cosine_distance
    - graph only:            score = 0.3
    - community only:        score = 0.4 (0.6 if multi-category)
    - vector + graph:        score += 0.5 boost
    - vector + community:    score += 0.3 boost
    - graph + community:     score += 0.2 boost
    - all three (best):      score += 0.8 boost

    Nodes found by multiple paths are ranked higher —
    agreement between independent retrieval methods
    is a strong signal of genuine relevance.
    """
    # Embed query
    q_emb = embed_query(query)

    # Run all 3 paths independently
    vec_results = vector_search(q_emb, top_k=top_k)
    gph_results = graph_search(query, max_hops=2,
                               top_k=top_k * 3)
    com_results = community_search(vec_results, top_k=top_k)

    # Merge by node_id
    merged = {}

    # Vector results — base
    for r in vec_results:
        nid = r['node_id']
        merged[nid] = {
            **r,
            'score': r['raw_score'],
            'source': 'vector',
            'found_by': ['vector']
        }

    # Graph results — add or boost
    for r in gph_results:
        nid = r['node_id']
        if nid in merged:
            merged[nid]['found_by'].append('graph')
            merged[nid]['source'] = 'vector+graph'
            merged[nid]['score'] += 0.5
            merged[nid]['hop_distance'] = r.get('hop_distance', 0)
        else:
            merged[nid] = {
                **r,
                'score': 0.3,
                'source': 'graph',
                'found_by': ['graph']
            }

    # Community results — add or boost
    for r in com_results:
        nid = r['node_id']
        base = r['raw_score']
        if nid in merged:
            found = merged[nid]['found_by']
            merged[nid]['found_by'].append('community')
            if 'vector' in found and 'graph' in found:
                merged[nid]['source'] = 'all'
                merged[nid]['score'] += 0.8
            elif 'vector' in found:
                merged[nid]['source'] = 'vector+community'
                merged[nid]['score'] += 0.3
            else:
                merged[nid]['source'] = 'graph+community'
                merged[nid]['score'] += 0.2
        else:
            merged[nid] = {
                **r,
                'score': base,
                'source': 'community',
                'found_by': ['community']
            }

    # Sort by score descending
    candidates = sorted(
        merged.values(), key=lambda x: -x['score']
    )[:top_k]

    # Source breakdown summary
    sources = defaultdict(int)
    for c in candidates:
        sources[c['source']] += 1

    return {
        'query': query,
        'candidates': candidates,
        'source_breakdown': dict(sources),
        'retrieval_stats': {
            'vector_candidates': len(vec_results),
            'graph_candidates':  len(gph_results),
            'community_candidates': len(com_results),
            'total_before_merge': len(merged) + top_k
        }
    }

# ── Test ───────────────────────────────────────────────────────
if __name__ == '__main__':
    test_queries = [
        "brake pedal feels spongy when I press it",
        "car engine overheating",
        "check engine light on dashboard",
        "transmission slipping between gears"
    ]

    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"Query: {query}")
        print('='*70)

        result = hybrid_retrieve(query, top_k=10)

        print(f"Source breakdown: {result['source_breakdown']}")
        stats = result['retrieval_stats']
        print(f"Candidates: vector={stats['vector_candidates']} "
              f"graph={stats['graph_candidates']} "
              f"community={stats['community_candidates']}")
        print()
        print(f"{'Rank':<5} {'Score':<7} {'Source':<20} "
              f"{'Type':<16} {'Category':<22} {'Label'}")
        print("-" * 100)
        for i, c in enumerate(result['candidates'], 1):
            print(
                f"{i:<5} {c['score']:<7.3f} "
                f"{c['source']:<20} "
                f"{c['node_type']:<16} "
                f"{c.get('category','')[:20]:<22} "
                f"{c.get('label','')[:35]}"
            )