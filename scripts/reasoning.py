import json

# Load graph triples
with open("graph/triples.json", "r", encoding="utf-8") as f:
    triples = json.load(f)

def search_graph(query):

    query = query.lower()

    results = []

    query_words = query.split()

    for triple in triples:

        subject = triple["subject"].lower()
        relation = triple["relation"].lower()
        obj = triple["object"].lower()

        combined = f"{subject} {relation} {obj}"

        # Match ANY keyword from query
        if any(word in combined for word in query_words):
            results.append(triple)

    return results