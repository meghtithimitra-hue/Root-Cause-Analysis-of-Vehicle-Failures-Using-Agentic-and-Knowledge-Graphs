import json
from collections import defaultdict, deque
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
import networkx as nx
from .query_preprocessor import preprocess_query
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

# ── Graph seed filtering ────────────────────────────────────────
# Words too generic to create graph seeds alone
_STOPWORDS = frozenset({
    'the', 'and', 'for', 'with', 'from', 'that', 'this',
    'when', 'have', 'been', 'were', 'they', 'their', 'them',
    'are', 'was', 'not', 'but', 'can', 'will', 'also',
    'how', 'what', 'which', 'where', 'does', 'into',
})
_GENERIC_SEED_WORDS = frozenset({
    'engine', 'vehicle', 'car', 'system', 'problem',
    'issue', 'fault', 'light', 'leak', 'noise',
})

# ── Stemmer (consistent with query_preprocessor._stem) ─────────
def _stem(word):
    """Minimal suffix-stripping stemmer matching query_preprocessor."""
    w = word.lower().strip()
    if len(w) <= 3:
        return w
    for suf in ('ation', 'tion', 'sion', 'ment', 'ness', 'ible', 'able',
                'ful', 'less', 'ous', 'ive', 'ing', 'ity', 'ent',
                'ant', 'ism', 'ist', 'ize', 'ise', 'ify', 'ate',
                'ence', 'ance', 'ics', 'ies', 'ers'):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[:-len(suf)]
    if w.endswith('es') and len(w) > 4:
        w2 = w[:-2]
        if w2[-1] == w2[-2] and w2[-1] in 'bcdfghjklmnpqrstvwxyz':
            return w2
        return w2
    if w.endswith('ed') and len(w) > 4:
        w2 = w[:-2]
        if w2[-1] == w2[-2] and w2[-1] in 'bcdfghjklmnpqrstvwxyz':
            return w2
        return w2
    if w.endswith('ly') and len(w) > 4:
        return w[:-2]
    if w.endswith('ss') and len(w) > 4:
        return w[:-1]
    if w.endswith('s') and not w.endswith('ss') and len(w) > 4:
        return w[:-1]
    return w

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

        with open(GRAPH_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        _graph_nx = nx.node_link_graph(
            data,
            edges="edges",
            multigraph=True
        )

        _graph_nodes = {
            node["id"]: node
            for node in data["nodes"]
        }

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

# ── Multi-query vector search ──────────────────────────────────
def _multi_query_vector_search(
    processed_query, expanded_queries, top_k=10
):
    """
    Embed all expanded queries, run vector search for each,
    merge with max-pooling of raw_score per node_id.
    """
    all_queries = list(dict.fromkeys(
        [processed_query] + list(expanded_queries)
    ))
    model = _get_model()
    embeddings = model.encode(all_queries).tolist()

    merged_vec = {}
    for emb in embeddings:
        raw = _get_collection().query(
            query_embeddings=[emb],
            n_results=top_k,
            include=['metadatas', 'distances', 'documents']
        )
        for meta, dist, doc in zip(
            raw['metadatas'][0],
            raw['distances'][0],
            raw['documents'][0]
        ):
            nid = meta['node_id']
            rs = round(1.0 - dist, 4)
            if nid not in merged_vec or rs > merged_vec[nid]['raw_score']:
                merged_vec[nid] = {
                    'node_id':          nid,
                    'node_type':        meta['node_type'],
                    'label':            meta.get('label', ''),
                    'category':         meta.get('category', ''),
                    'subcategory':      meta.get('subcategory', ''),
                    'community_id':     meta.get('community_id', '-1'),
                    'is_multi_category': meta.get(
                        'is_multi_category', 'False'
                    ),
                    'embedding_text':   doc,
                    'distance':         dist,
                    'raw_score':        rs
                }
    return sorted(
        merged_vec.values(), key=lambda r: -r['raw_score']
    )[:top_k * 2]

# ── PATH 2: Graph Search ───────────────────────────────────────
def graph_search(
    query,
    communities=None,
    categories=None,
    max_hops=1,
    top_k=30
) -> list:
    """
    Structural graph traversal via BFS.
    Finds seed nodes by stemmed match on labels
    (filtering stopwords and generic seed words),
    then expands outward up to max_hops via graph edges.
    """
    G, graph_nodes = _load_graph()

    query_words = query.lower().split()
    query_stems = frozenset(
        _stem(w) for w in query_words if len(w) > 2
    )
    seed_words = frozenset(
        w for w in query_words
        if len(w) > 2
        and w not in _STOPWORDS
        and w not in _GENERIC_SEED_WORDS
    )

    if not query_stems:
        return []

    seeds = set()
    for nid, nd in graph_nodes.items():

        if communities:

            if str(nd.get("community")) not in map(str, communities):

                continue

        label = nd.get("label", "").lower()
        label_stems = frozenset(
            _stem(w) for w in label.split() if len(w) > 2
        )

        stem_overlap = query_stems & label_stems
        sub_match = False
        if seed_words:
            for sw in seed_words:
                sw_stem = _stem(sw)
                for lw in label.split():
                    if sw in lw or lw.startswith(sw_stem[:4]):
                        sub_match = True
                        break
                if sub_match:
                    break
        if not stem_overlap and not sub_match:
            continue

        if seed_words and not sub_match:
            if not (seed_words & set(label.split())):
                continue

        seeds.add(nid)

    if not seeds:
        for nid, nd in graph_nodes.items():
            if communities:
                if str(nd.get("community")) not in map(str, communities):
                    continue
            label = nd.get("label", "").lower()
            label_stems = frozenset(
                _stem(w) for w in label.split() if len(w) > 2
            )
            specific = query_stems - _GENERIC_SEED_WORDS
            if specific and (specific & label_stems):
                seeds.add(nid)
            elif not specific and (query_stems & label_stems):
                seeds.add(nid)
        if not seeds:
            return []

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

    results = []
    for nid, hop in visited.items():
        nd = graph_nodes.get(nid, {})
        label = nd.get('label', nid)
        label_stems = frozenset(
            _stem(w) for w in label.lower().split() if len(w) > 2
        )
        overlap = len(query_stems & label_stems)
        is_seed = hop == 0
        if is_seed:
            raw = round(max(0, 0.35 * min(overlap, 3) / 3), 4)
        else:
            decay = 0.5 ** hop
            raw = round(max(0, 0.20 * decay * min(overlap, 3) / 3), 4)
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
            'raw_score':        raw
        })

    results.sort(key=lambda r: (
        r['hop_distance'], -r['word_overlap'], r['label']
    ))
    return results[:top_k]

# ── PATH 3: Community Search ───────────────────────────────────
def community_search(
    preprocessor_communities,
    query,
    top_k_per_community=3
) -> list:
    """
    Community-aware retrieval using predicted communities
    from query_preprocessor. Within each community, rank
    nodes by stem overlap with the query and return top K.
    """
    if not preprocessor_communities:
        return []

    cm = _load_community_map()
    _, graph_nodes = _load_graph()

    query_stems = frozenset(
        _stem(w) for w in query.lower().split() if len(w) > 2
    )
    query_specific = query_stems - _GENERIC_SEED_WORDS

    results = []
    for cid in preprocessor_communities:
        community_data = cm.get(str(cid), {})
        if not community_data:
            continue
        is_multi = community_data.get('is_multi_category', False)
        base_score = 0.45 if is_multi else 0.30
        cats = community_data.get('categories', [])

        scored_nodes = []
        for nid in community_data.get('node_ids', []):
            nd = graph_nodes.get(nid, {})
            ntype = nd.get('node_type', '')
            if not ntype or ntype == 'Unknown':
                continue
            label = nd.get('label', nid)
            label_stems = frozenset(
                _stem(w) for w in label.lower().split() if len(w) > 2
            )
            specific_overlap = len(query_specific & label_stems)
            total_overlap = len(query_stems & label_stems)
            if query_specific and specific_overlap == 0:
                continue
            if not query_specific and total_overlap == 0:
                continue
            scored_nodes.append(
                (specific_overlap, total_overlap, nid, nd, ntype)
            )

        scored_nodes.sort(key=lambda x: (-x[0], -x[1]))

        for so, to, nid, nd, ntype in scored_nodes[:top_k_per_community]:
            label = nd.get('label', nid)
            rs = round(base_score + so * 0.15 + to * 0.05, 4)
            results.append({
                'node_id':          nid,
                'node_type':        ntype,
                'label':            label,
                'category':         nd.get('category', ''),
                'subcategory':      nd.get('subcategory', ''),
                'community_id':     str(cid),
                'is_multi_category': str(is_multi),
                'community_categories': cats,
                'raw_score':        rs
            })

    return results

# ── Semantic dedup ─────────────────────────────────────────────
def _semantic_dedup(candidates, max_final=15):
    """
    Collapse candidates whose labels share the same stem set.
    Keep the highest-scoring representative per stem cluster.
    """
    stem_to_best = {}
    for c in candidates:
        label_stems = frozenset(
            _stem(w) for w in c['label'].lower().split() if len(w) > 2
        )
        key = label_stems
        if key not in stem_to_best or c['score'] > stem_to_best[key]['score']:
            stem_to_best[key] = c
    deduped = sorted(
        stem_to_best.values(), key=lambda x: -x['score']
    )
    return deduped[:max_final]

# ── HYBRID MERGE ───────────────────────────────────────────────
def hybrid_retrieve(query: str, top_k: int = 10) -> dict:
    """
    Main retrieval function combining all 3 paths.

    Uses multi-query vector search, stemmed graph traversal,
    preprocessor-guided community retrieval, independent score
    normalization, weighted fusion, and semantic deduplication.
    """
    # ----------------------------------------
    # Preprocess Query
    # ----------------------------------------

    preprocessed = preprocess_query(query)

    processed_query = preprocessed["processed"]

    expanded_queries = preprocessed["expanded_queries"]

    retrieval_hints = preprocessed["retrieval_hints"]

    communities = retrieval_hints["communities"]

    categories = retrieval_hints["categories"]

    expected_sensors = preprocessed["expected_sensors"]

    entities = preprocessed["entities"]

    # ----------------------------------------
    # PATH 1: Multi-query vector search
    # ----------------------------------------

    vec_results = _multi_query_vector_search(
        processed_query, expanded_queries, top_k=top_k
    )

    # ----------------------------------------
    # PATH 2: Stemmed graph search
    # ----------------------------------------

    entity_labels = [e['label'] for e in entities]

    gph_results = graph_search(

        processed_query,

        communities=communities,

        categories=categories,

        max_hops=1,

        top_k=top_k * 3

    )

    # ----------------------------------------
    # PATH 3: Preprocessor-guided community search
    # ----------------------------------------

    com_results = community_search(
        communities, processed_query, top_k_per_community=3
    )

    # ----------------------------------------
    # Normalize scores independently to [0,1]
    # ----------------------------------------

    def _norm(results, key='raw_score'):
        if not results:
            return results
        vals = [r[key] for r in results]
        lo, hi = min(vals), max(vals)
        span = hi - lo if hi > lo else 1.0
        for r in results:
            r['norm_score'] = round((r[key] - lo) / span, 4)
        return results

    _norm(vec_results)
    _norm(gph_results)
    _norm(com_results)

    # ----------------------------------------
    # Merge by node_id
    # ----------------------------------------

    merged = {}

    # Vector results — base
    for r in vec_results:
        nid = r['node_id']
        merged[nid] = {
            **r,
            'score': r['norm_score'],
            'source': 'vector',
            'found_by': ['vector']
        }

    # Graph results — add or boost
    for r in gph_results:

        nid = r['node_id']

        if nid in merged:

            merged[nid]['found_by'].append('graph')
            merged[nid]['source'] = 'vector+graph'
            merged[nid]['graph_norm'] = r['norm_score']
            merged[nid]['hop_distance'] = r.get('hop_distance', 0)

        else:

            merged[nid] = {
                **r,
                'score': r['norm_score'],
                'graph_norm': r['norm_score'],
                'source': 'graph',
                'found_by': ['graph']
            }

    # Community results — add or boost
    for r in com_results:
        nid = r['node_id']
        if nid in merged:
            merged[nid]['found_by'].append('community')
            merged[nid]['community_norm'] = r['norm_score']
            found = merged[nid]['found_by']
            if 'vector' in found and 'graph' in found:
                merged[nid]['source'] = 'all'
            elif 'vector' in found:
                merged[nid]['source'] = 'vector+community'
            else:
                merged[nid]['source'] = 'graph+community'
        else:
            merged[nid] = {
                **r,
                'score': r['norm_score'],
                'community_norm': r['norm_score'],
                'source': 'community',
                'found_by': ['community']
            }

    # ----------------------------------------
    # Weighted fusion
    # ----------------------------------------

    W_VEC, W_GPH, W_COM = 0.55, 0.25, 0.20

    for item in merged.values():
        v = item.get('score', 0)
        g = item.get('graph_norm', 0)
        c = item.get('community_norm', 0)
        item['score'] = round(
            W_VEC * v + W_GPH * g + W_COM * c, 4
        )

    # ----------------------------------------
    # Path agreement bonus
    # ----------------------------------------

    _AGREEMENT = {
        frozenset(['vector']):             1.0,
        frozenset(['graph']):              1.0,
        frozenset(['community']):          1.0,
        frozenset(['vector', 'graph']):    1.15,
        frozenset(['vector', 'community']):1.10,
        frozenset(['graph', 'community']): 1.05,
        frozenset(['vector', 'graph', 'community']): 1.25,
    }

    for item in merged.values():
        fb = frozenset(item['found_by'])
        item['score'] = round(
            item['score'] * _AGREEMENT.get(fb, 1.0), 4
        )

    # ----------------------------------------
    # Category hint boost
    # ----------------------------------------

    if categories:
        cat_set = set(c.lower() for c in categories)
        for item in merged.values():
            nc = item.get('category', '').lower()
            if nc in cat_set:
                item['score'] = round(item['score'] * 1.08, 4)

    # ----------------------------------------
    # Node type ranking penalty
    # ----------------------------------------

    NODE_TYPE_PENALTY = {
        "DiagnosisStep": 0.80,
        "Subcategory":   0.85,
        "Category":      0.75,
        "Symptom":       1.00,
        "Fault":         1.00,
        "Sensor":        0.95,
        "Cause":         1.00,
        "Action":        0.90,
    }

    for item in merged.values():
        nt = item.get("node_type")
        item["score"] = round(
            item["score"] * NODE_TYPE_PENALTY.get(nt, 1.0), 4
        )

    # ----------------------------------------
    # Sort, dedup, truncate
    # ----------------------------------------

    candidates = sorted(
        merged.values(), key=lambda x: -x['score']
    )

    candidates = _semantic_dedup(candidates, max_final=top_k)

    # Source breakdown summary
    sources = defaultdict(int)
    for c in candidates:
        sources[c['source']] += 1

    return {

        "query": query,

        "processed_query": processed_query,

        "expected_sensors": expected_sensors,

        "retrieval_hints": retrieval_hints,

        "entities": entities,

        "candidates": candidates,

        "source_breakdown": dict(sources),

        "retrieval_stats": {

            "vector_candidates": len(vec_results),

            "graph_candidates": len(gph_results),

            "community_candidates": len(com_results),

            "total_before_merge": len(merged) + top_k

        }

    }

# ── Test ───────────────────────────────────────────────────────
if __name__ == '__main__':
    # Load graph first
    _, graph_nodes = _load_graph()

    node_types = set()

    for node in graph_nodes.values():
        node_types.add(node.get("node_type"))

    print("\nNode Types in Graph:")
    print(sorted(node_types))
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
