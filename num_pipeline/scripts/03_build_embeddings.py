import json
import shutil
from pathlib import Path
from collections import defaultdict
import chromadb
from sentence_transformers import SentenceTransformer

GRAPH_JSON = "data/processed/hierarchical_graph.json"
COMMUNITY_MAP = "graphify-out/community_map.json"
CHROMA_DIR = "data/chroma_db"
COLLECTION = "automotive_kg"
MODEL_NAME = "all-MiniLM-L6-v2"

# Fresh start
if Path(CHROMA_DIR).exists():
    shutil.rmtree(CHROMA_DIR)
Path(CHROMA_DIR).mkdir(parents=True)

with open(GRAPH_JSON) as f:
    gdata = json.load(f)
with open(COMMUNITY_MAP) as f:
    community_map = json.load(f)

multi_cat_ids = {k for k, v in community_map.items()
                 if v.get('is_multi_category')}

model = SentenceTransformer(MODEL_NAME)
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.create_collection(COLLECTION)

texts, ids, metas = [], [], []

for node in gdata.get('nodes', []):
    nid = node.get('id', '')
    ntype = node.get('node_type', '')

    if ntype not in ('Subcategory', 'Symptom', 'DiagnosisStep'):
        continue

    label = node.get('label', '')
    cat = node.get('category', '')
    subcat = node.get('subcategory', '')
    cid = str(node.get('community', -1))

    if ntype == 'Subcategory':
        text = f"{cat} > {label}"
    elif ntype == 'Symptom':
        text = f"{cat} > {subcat} | Symptom: {label}"
    else:
        ra = node.get('result_a', '')
        rb = node.get('result_b', '')
        text = f"{cat} > {subcat} | Diagnosis step: {label}"
        if ra:
            text += f" | Result A: {ra}"
        if rb:
            text += f" | Result B: {rb}"

    texts.append(text)
    ids.append(nid)
    metas.append({
        'node_id': nid,
        'node_type': ntype,
        'category': cat,
        'subcategory': subcat,
        'community_id': cid,
        'is_multi_category': str(cid in multi_cat_ids),
        'label': label
    })

print(f"Embedding {len(texts)} nodes...")
embeddings = model.encode(texts, show_progress_bar=True).tolist()

batch = 100
for i in range(0, len(texts), batch):
    collection.add(
        ids=ids[i:i+batch],
        embeddings=embeddings[i:i+batch],
        metadatas=metas[i:i+batch],
        documents=texts[i:i+batch]
    )

print(f"\nStored {len(texts)} embeddings in ChromaDB")

# Node type breakdown in ChromaDB
type_counts = defaultdict(int)
for m in metas:
    type_counts[m['node_type']] += 1
print("\nEmbedded by type:")
for t, c in sorted(type_counts.items()):
    print(f"  {t}: {c}")

# Self-test queries
test_queries = [
    "brake pedal feels spongy",
    "car engine overheating",
    "check engine light on",
    "transmission slipping gears"
]

print("\n=== SELF-TEST QUERIES ===")
for query in test_queries:
    qemb = model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[qemb],
        n_results=3,
        include=['metadatas', 'distances', 'documents']
    )
    print(f"\nQuery: '{query}'")
    for meta, dist, doc in zip(
        results['metadatas'][0],
        results['distances'][0],
        results['documents'][0]
    ):
        print(f"  [{meta['node_type']:<14}] "
              f"score={1-dist:.3f} | "
              f"com={meta['community_id']:>2} | "
              f"{doc[:55]}")

print("\nDone. ChromaDB ready at:", CHROMA_DIR)