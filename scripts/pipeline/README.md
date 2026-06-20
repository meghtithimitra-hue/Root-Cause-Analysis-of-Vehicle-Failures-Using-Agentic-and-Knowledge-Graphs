# Hybrid Retrieval Pipeline

## Input

`hybrid_retrieve(query: str, top_k=10) -> dict`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | — | Natural-language fault description |
| `top_k` | `int` | `10` | Maximum candidates in the merged result |

## Output

```json
{
  "query": "brake pedal feels spongy when I press it",
  "candidates": [
    {
      "node_id": "sym:<md5>",
      "node_type": "Symptom",
      "label": "Spongy brake pedal",
      "category": "ABS System",
      "subcategory": "Brake Shoe & Drum",
      "score": 0.739,
      "source": "vector"
    }
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `query` | `str` | The original input query (passthrough, no rewriting). |
| `candidates` | `list[dict]` | Ranked candidate list, descending by `score`. |

Each candidate:

| Field | Type | Description |
|-------|------|-------------|
| `node_id` | `str` | Canonical ID in the knowledge graph (e.g. `cat:abs_system`, `subcat:brake_rotor`, `sym:<md5>`). |
| `node_type` | `str` | One of `Category`, `Subcategory`, `Symptom`, `DiagnosisStep`. |
| `label` | `str` | Human-readable text of the node. |
| `category` | `str` | Parent category name (empty if the node itself is a Category). |
| `subcategory` | `str` | Parent subcategory name (empty if the node itself is a Subcategory or Category). |
| `score` | `float` | Composite relevance score, higher is better (range 0–1+ due to boost). |
| `source` | `str` | `"vector"` — found via ChromaDB cosine similarity; `"graph"` — found via substring BFS; `"both"` — found by both paths and boosted. |

## Retrieval architecture

```
                      query
                     /      \
                    /        \
         embed_query()    graph_search()
              |                |
         vector_search()   substring match
              |            → seed nodes
         ChromaDB          → BFS (max_hops=2)
              |                |
              +---- merge -----+
                      |
                 deduplicate
                 boost "both"
                      |
                 rank by score
```

## What was intentionally left out

This is a simplified hybrid retrieval stage — the following components are **not** implemented here and are available for the partner to add:

- **Intent classification** — distinguishing symptom descriptions from diagnosis-step questions or part queries, to route to different retrievers or prompt templates.
- **Multi-query expansion** — generating alternative phrasings of the query and running multiple retrieval passes.
- **Fuzzy / typo correction** — handling misspelled component names (e.g. "break pad" → "brake pad").
- **Community-aware expansion** — boosting results from the same Graphify community cluster the top hit belongs to.
- **Category-aware reranking** — preferring results from categories that match the vehicle system implied by the query.

These were omitted to keep the initial integration lean. Each can be added as a wrapper around or a pre-processing step before `hybrid_retrieve()` without changing its API contract.
