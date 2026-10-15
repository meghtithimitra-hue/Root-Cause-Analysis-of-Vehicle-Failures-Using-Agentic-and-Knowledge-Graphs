# Hybrid Retrieval Pipeline

## Overview
Three independent retrieval paths merged into one ranked candidate list.

## Input
`hybrid_retrieve(query: str, top_k: int = 10) -> dict`

## Output
```json
{
  "query": "brake pedal feels spongy",
  "candidates": [
    {
      "node_id": "sym_...",
      "node_type": "Symptom",
      "label": "Spongy brake pedal",
      "category": "ABS System",
      "subcategory": "Brake Shoe & Drum",
      "score": 1.373,
      "source": "all"
    }
  ],
  "source_breakdown": {"all": 1, "vector+graph": 3, "community": 6},
  "retrieval_stats": {
    "vector_candidates": 10,
    "graph_candidates": 30,
    "community_candidates": 10
  }
}
```

## 3 Retrieval Paths

### 1. Vector Search
ChromaDB cosine similarity using all-MiniLM-L6-v2 embeddings.
Finds semantically similar nodes even with different wording.
Score = 1.0 - cosine_distance

### 2. Graph Search
BFS traversal from substring-matched seed nodes, max 2 hops.
Captures hierarchy context — parent categories, sibling
subcategories, related symptoms.
Score = 1.0 - (hop * 0.3) + (word_overlap * 0.1)

### 3. Community Search
Uses Louvain community assignments from graphify.
Top 3 vector results vote on best community (majority wins).
Returns all nodes in that community.
Multi-category communities score 0.6, single-category 0.4.

## Scoring / Boost Logic
| Source | Score |
|---|---|
| vector only | 1.0 - cosine_distance |
| graph only | 0.3 |
| community only | 0.4 (0.6 if multi-category) |
| vector + graph | base + 0.5 boost |
| vector + community | base + 0.3 boost |
| graph + community | base + 0.2 boost |
| all three | base + 0.8 boost |

## Handoff Contract for Next Stage
The `candidates` list is the input to:
Community-Aware GraphRAG → Reasoning Path Generation →
Evidence-Based Ranking → Confidence Scoring → Mode Decision

Each candidate has: node_id, node_type, label, category,
subcategory, score, source, community_id, is_multi_category

## Intentionally omitted (for partner to add)
- Intent classification
- Multi-query expansion
- Fuzzy matching / typo correction
- Category-aware query expansion