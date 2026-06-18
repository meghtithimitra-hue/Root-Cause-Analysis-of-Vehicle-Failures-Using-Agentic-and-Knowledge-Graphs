"""
Community-aware graph retriever for automotive fault diagnosis.

Retrieves reasoning paths (Symptom → Component → DiagnosticTest → Result / RepairAction)
plus the symptom's Leiden community and neighboring nodes from that community.
"""

import os
import re
from collections import Counter
from neo4j import GraphDatabase


def get_driver():
    return GraphDatabase.driver(
        "bolt://localhost:7687",
        auth=("neo4j", "neo4j@123")
    )


def _fetch_community(driver, symptom_name: str) -> dict | None:
    query = """
    MATCH (sym:Symptom)-[:BELONGS_TO]->(c:Community)
    WHERE toLower(sym.name) = toLower($name)
    RETURN c.community_id AS id, c.name AS name, c.size AS size
    """
    with driver.session(database="neo4j") as session:
        rows = session.run(query, name=symptom_name).data()
    if not rows:
        return None
    return {
        "id": rows[0]["id"],
        "name": rows[0]["name"],
        "size": rows[0]["size"],
    }


def _fetch_community_context(driver, symptom_name: str) -> dict | None:
    query = """
    MATCH (sym:Symptom)-[:BELONGS_TO]->(c:Community)<-[:BELONGS_TO]-(n)
    WHERE toLower(sym.name) = toLower($name)
    RETURN n.name AS name, labels(n) AS labels
    """
    with driver.session(database="neo4j") as session:
        rows = session.run(query, name=symptom_name).data()
    if not rows:
        return {}
    context = {"symptoms": [], "components": [], "tests": [], "results": [], "repairs": []}
    for row in rows:
        labels = [l for l in row["labels"] if l != "Community"]
        label = labels[0] if labels else "Unknown"
        mapping = {
            "Symptom": "symptoms",
            "Component": "components",
            "DiagnosticTest": "tests",
            "Result": "results",
            "RepairAction": "repairs",
        }
        key = mapping.get(label)
        if key:
            context[key].append(row["name"])
    # Remove empty keys
    return {k: v for k, v in context.items() if v}


def _compute_path_strength(reasoning_paths: list[list[str]]) -> float:
    if not reasoning_paths:
        return 0.0
    lengths = [len(p) for p in reasoning_paths]
    avg_len = sum(lengths) / len(lengths)
    max_len = max(lengths)
    return avg_len / max_len if max_len > 0 else 0.0


def _compute_support_count(reasoning_paths: list[list[str]]) -> float:
    if not reasoning_paths:
        return 0.0
    return min(len(reasoning_paths) / 10.0, 1.0)


def _compute_community_agreement(driver, reasoning_paths: list[list[str]]) -> float:
    all_nodes: set[str] = set()
    for path in reasoning_paths:
        all_nodes.update(path)
    if not all_nodes:
        return 0.0
    query = """
    UNWIND $nodes AS node_name
    OPTIONAL MATCH (n {name: node_name})-[:BELONGS_TO]->(c:Community)
    RETURN n.name AS name, c.community_id AS community_id
    """
    with driver.session(database="neo4j") as session:
        rows = session.run(query, nodes=list(all_nodes)).data()
    if not rows:
        return 0.0
    comm_counts: Counter = Counter()
    total_with_community = 0
    for row in rows:
        cid = row.get("community_id")
        if cid is not None:
            comm_counts[cid] += 1
            total_with_community += 1
    if total_with_community == 0:
        return 0.0
    return comm_counts.most_common(1)[0][1] / total_with_community


def _compute_test_agreement(all_test_result_pairs: list[str]) -> float:
    results: list[str] = []
    for pair in all_test_result_pairs:
        if " -> " in pair:
            _, res = pair.split(" -> ", 1)
            results.append(res)
    if not results:
        return 0.0
    counts = Counter(results)
    return counts.most_common(1)[0][1] / len(results)


# ── Synonym map for query expansion ────────────────────────────────────────

_SYNONYM_MAP = {
    "hot": "Engine Overheating",
    "overheat": "Engine Overheating",
    "overheating": "Engine Overheating",
    "high temperature": "Engine Overheating",
    "temp high": "Engine Overheating",
    "engine hot": "Engine Overheating",
    "car hot": "Engine Overheating",
    "brake vibration": "Brake pedal pulsation",
    "brake shake": "Brake pedal pulsation",
    "brakes shaking": "Brake pedal pulsation",
    "brake shudder": "Brake pedal pulsation",
    "brake pulsation": "Brake pedal pulsation",
    "pulsating brake": "Brake pedal pulsation",
    "vibration brake": "Brake pedal pulsation",
    "shaking brake": "Brake pedal pulsation",
    "no start": "Engine No Start",
    "wont start": "Engine No Start",
    "crank no start": "Engine No Start",
    "cranks no start": "Engine No Start",
    "rough idle": "Engine Rough Idle",
    "idle rough": "Engine Rough Idle",
    "rough running": "Engine Rough Idle",
    "check engine": "Check Engine Light",
    "engine light": "Check Engine Light",
    "cel on": "Check Engine Light",
    "misfire": "Engine Misfire",
    "hesitation": "Engine Hesitation",
    "stalling": "Engine Stalling",
    "stall": "Engine Stalling",
    "stalls": "Engine Stalling",
    "smoke exhaust": "Excessive Smoke",
    "engine noise": "Unusual Engine Noise",
    "knocking": "Unusual Engine Noise",
    "ticking": "Unusual Engine Noise",
    "oil leak": "Fluid Leak",
    "fluid leak": "Fluid Leak",
    "coolant leak": "Coolant Leak",
    "brake fade": "Brake Fade",
    "soft brake": "Brake Fade",
    "spongy brake": "Brake Fade",
    "hard brake": "Hard Brake Pedal",
    "stiff brake": "Hard Brake Pedal",
    "brake squeal": "Brake Squeal",
    "squealing brake": "Brake Squeal",
    "car pulls": "Vehicle Pulls",
    "pull left": "Vehicle Pulls",
    "pull right": "Vehicle Pulls",
    "drift": "Vehicle Pulls",
    "dead battery": "Dead Battery",
    "battery dead": "Dead Battery",
    "crank slow": "Dead Battery",
    "dim light": "Dead Battery",
    "ac warm": "AC Not Cooling",
    "ac not cold": "AC Not Cooling",
    "air condition": "AC Not Cooling",
    "blower not working": "HVAC Blower Failure",
    "no heat": "HVAC Blower Failure",
    "transmission slip": "Transmission Slipping",
    "slipping transmission": "Transmission Slipping",
    "hard shift": "Transmission Slipping",
    "poor gas": "Poor Fuel Economy",
    "bad mpg": "Poor Fuel Economy",
    "fuel smell": "Fuel Smell",
    "gas smell": "Fuel Smell",
    "smell gas": "Fuel Smell",
    "over heating": "Engine Overheating",
    "over heated": "Engine Overheating",
}


def _synonym_expand(query: str) -> list[str]:
    """Expand a user query using the synonym map. Returns up to 5 unique symptom names."""
    q = query.lower().strip()
    results: set[str] = set()
    for phrase, symptom in _SYNONYM_MAP.items():
        if phrase in q:
            results.add(symptom)
    return list(results)[:5]


# ── Intent classification ──────────────────────────────────────────────────

def classify_query(query: str) -> str:
    """Classify a user query into one of: SYMPTOM, CAUSE, REPAIR, TEST."""
    q = query.lower().strip()
    cause_patterns = ["why", "what cause", "reason for", "root cause", "what makes", "what would cause", "why is"]
    for p in cause_patterns:
        if p in q:
            return "CAUSE"
    repair_patterns = ["fix", "repair", "replace", "how to", "how do i", "solution", "remedy", "correct", "what should i do", "how can i"]
    for p in repair_patterns:
        if p in q:
            return "REPAIR"
    test_patterns = ["inspect", "check", "test", "diagnos", "examine", "look at", "measure", "verify", "what should i check"]
    for p in test_patterns:
        if p in q:
            return "TEST"
    return "SYMPTOM"


# ── Fuzzy matching ─────────────────────────────────────────────────────────

def _levenshtein(s1: str, s2: str) -> int:
    """Compute Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if not s2:
        return len(s1)
    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr_row.append(min(curr_row[j] + 1, prev_row[j + 1] + 1, prev_row[j] + cost))
        prev_row = curr_row
    return prev_row[-1]


def _fuzzy_match_symptom(query: str, all_symptoms: list[str]) -> tuple[str | None, float]:
    """
    Find the best matching symptom name in the graph using exact match,
    CONTAINS, then Levenshtein similarity fallback.
    Returns (best_name, score) where score is 0.0–1.0.
    """
    q = query.lower().strip()
    if not q or not all_symptoms:
        return None, 0.0
    best_name: str | None = None
    best_score = 0.0
    for symptom in all_symptoms:
        s = symptom.lower()
        # Exact match
        if s == q:
            return symptom, 1.0
        # CONTAINS (query in symptom or symptom in query)
        contained = q in s or s in q
        # Word-level overlap
        q_words = set(q.split())
        s_words = set(s.split())
        overlap = len(q_words & s_words)
        word_ratio = overlap / max(len(q_words | s_words), 1)
        if contained or word_ratio > 0.3:
            raw = word_ratio if word_ratio > 0 else min(len(q), len(s)) / max(len(q), len(s), 1)
            score = min(raw + 0.15, 1.0)
            if score > best_score:
                best_score = score
                best_name = symptom
                if score >= 1.0:
                    return best_name, best_score
        # Levenshtein similarity
        dist = _levenshtein(q, s)
        max_len = max(len(q), len(s))
        if max_len == 0:
            continue
        sim = 1.0 - (dist / max_len)
        if sim > 0.6 and sim > best_score:
            best_score = sim
            best_name = symptom
    return best_name, round(best_score, 4)


def _fetch_all_symptom_names(driver) -> list[str]:
    """Return all Symptom node names from the graph."""
    query = "MATCH (s:Symptom) RETURN s.name AS name"
    with driver.session(database="neo4j") as session:
        rows = session.run(query).data()
    return [r["name"] for r in rows]


# ── Query expansion ────────────────────────────────────────────────────────

def expand_query(query: str) -> list[str]:
    """
    Expand a user query into multiple automotive search queries.
    Uses synonym dictionary. Falls back gracefully if no expansions found.
    Returns up to 5 unique queries.
    """
    if not query or not query.strip():
        return []
    expanded = _synonym_expand(query)
    if expanded:
        return expanded[:5]
    return [query.strip()]


# ── Community helpers ──────────────────────────────────────────────────────

def _fetch_community_members(driver, symptom_name: str, limit: int = 10) -> list[str]:
    """Return up to `limit` community member names for a given symptom."""
    query = """
    MATCH (sym:Symptom)-[:BELONGS_TO]->(c:Community)<-[:BELONGS_TO]-(n)
    WHERE toLower(sym.name) = toLower($name)
    RETURN n.name AS name
    LIMIT $limit
    """
    try:
        with driver.session(database="neo4j") as session:
            rows = session.run(query, name=symptom_name, limit=limit).data()
        return [r["name"] for r in rows]
    except Exception:
        return []


def _flatten_community_context(ctx: dict) -> list[str]:
    """Flatten the community_context dict into a single list of member names."""
    members: list[str] = []
    for key in ("symptoms", "components", "tests", "results", "repairs"):
        members.extend(ctx.get(key, []))
    return members[:10]


# ── Intent-based re-ranking ────────────────────────────────────────────────

def _intent_rerank_paths(reasoning_paths: list[list[str]], intent: str, result: dict) -> list[list[str]]:
    """
    Re-rank reasoning paths to prioritise nodes matching the detected intent.
    SYMPTOM → no change
    CAUSE   → prioritise paths ending in a Result node
    REPAIR  → prioritise paths ending in a RepairAction node
    TEST    → prioritise paths ending in a DiagnosticTest node
    """
    if intent == "SYMPTOM" or not reasoning_paths:
        return reasoning_paths
    endings: set[str] = set()
    if intent == "CAUSE":
        endings = set(result.get("results", []))
    elif intent == "REPAIR":
        endings = set(result.get("repair_actions", []))
    elif intent == "TEST":
        endings = set(result.get("diagnostic_tests", []))
    if not endings:
        return reasoning_paths

    def sort_key(path: list[str]) -> tuple:
        bonus = -2 if path and path[-1] in endings else 0
        return (bonus, -len(path))

    return sorted(reasoning_paths, key=sort_key)


# ── Multi-result ranking ──────────────────────────────────────────────────

def _rank_results(results: list[dict]) -> dict | None:
    """
    Rank multiple retrieval results using:
    score = 0.40 * fuzzy_match_score + 0.30 * support_count + 0.30 * path_strength
    Returns the top result.
    """
    if not results:
        return None
    for r in results:
        score = (
            0.40 * r.get("match_score", 0) +
            0.30 * r.get("support_count", 0) +
            0.30 * r.get("path_strength", 0)
        )
        r["_rank_score"] = score
    results.sort(key=lambda x: -x.get("_rank_score", 0))
    return results[0]


# ── Smart retrieval (full pipeline) ────────────────────────────────────────

def smart_retrieve(query: str) -> dict:
    """
    Full pipeline retrieval:
      1. Classify query intent
      2. Expand query into multiple search queries
      3. Fuzzy-match each expanded query to a graph Symptom node
      4. Run graph retrieval for each match
      5. Rank merged results
      6. Apply intent-based re-ranking
      7. Attach metadata (intent, expanded queries, community members, etc.)
    """
    intent = classify_query(query)
    expanded = expand_query(query)

    driver = get_driver()
    try:
        all_symptoms = _fetch_all_symptom_names(driver)
    except Exception:
        all_symptoms = []
    finally:
        driver.close()

    matched_results: list[dict] = []
    for eq in expanded:
        try:
            best_name, score = _fuzzy_match_symptom(eq, all_symptoms)
        except Exception:
            continue
        if best_name and score >= 0.5:
            try:
                result = retrieve_symptom(best_name)
            except Exception:
                continue
            if result.get("reasoning_paths"):
                result["match_score"] = score
                result["matched_symptom"] = best_name
                matched_results.append(result)

    if not matched_results:
        try:
            best_name, score = _fuzzy_match_symptom(query, all_symptoms)
        except Exception:
            best_name, score = None, 0.0
        if best_name and score >= 0.5:
            try:
                result = retrieve_symptom(best_name)
            except Exception:
                result = None
            if result and result.get("reasoning_paths"):
                result["match_score"] = score
                result["matched_symptom"] = best_name
                matched_results.append(result)

    result: dict
    if matched_results:
        best = _rank_results(matched_results)
        if best is None:
            result = _empty_result(query)
        else:
            paths = best.get("reasoning_paths", [])
            best["reasoning_paths"] = _intent_rerank_paths(paths, intent, best)
            best["intent"] = intent
            best["raw_query"] = query
            best["expanded_queries"] = expanded
            ctx = best.get("community_context", {})
            best["community_members"] = _flatten_community_context(ctx)
            result = best
    else:
        result = _empty_result(query)
        result["intent"] = intent
        result["expanded_queries"] = expanded

    return result


def _empty_result(query: str) -> dict:
    """Return an empty result dict for a query that produced no matches."""
    return {
        "symptom": query,
        "matched_symptom": query,
        "match_score": 0.0,
        "intent": "SYMPTOM",
        "raw_query": query,
        "expanded_queries": [],
        "diagnostic_tests": [],
        "results": [],
        "repair_actions": [],
        "reasoning_paths": [],
        "top_paths": [],
        "community": None,
        "community_context": {},
        "community_members": [],
        "evidence_score": 0,
        "path_strength": 0,
        "support_count": 0,
        "community_agreement": 0,
        "test_agreement": 0,
    }


def retrieve_symptom(symptom_name: str) -> dict:
    driver = get_driver()
    query = """
    MATCH (sym:Symptom)<-[:SHOWS_SYMPTOM]-(comp:Component)
    WHERE toLower(sym.name) = toLower($name)
    OPTIONAL MATCH (comp)-[:DIAGNOSED_BY]->(dt:DiagnosticTest)
    OPTIONAL MATCH (dt)-[:HAS_RESULT]->(res:Result)
    OPTIONAL MATCH (comp)-[:REQUIRES_FIX]->(ra:RepairAction)
    RETURN comp.name AS component,
           collect(DISTINCT dt.name) AS diagnostic_tests,
           collect(DISTINCT dt.name + " -> " + res.name) AS test_result_pairs,
           collect(DISTINCT res.name) AS results,
           collect(DISTINCT ra.name) AS repair_actions
    """
    with driver.session(database="neo4j") as session:
        rows = session.run(query, name=symptom_name).data()

    community = _fetch_community(driver, symptom_name)
    community_context = _fetch_community_context(driver, symptom_name)

    if not rows:
        driver.close()
        return {
            "symptom": symptom_name,
            "diagnostic_tests": [],
            "results": [],
            "repair_actions": [],
            "reasoning_paths": [],
            "community": community,
            "community_context": community_context or {},
        }

    all_tests: set[str] = set()
    all_results: set[str] = set()
    all_repairs: set[str] = set()
    all_test_result_pairs: list[str] = []
    reasoning_paths: list[list[str]] = []

    for row in rows:
        comp = row["component"]
        tests = row["diagnostic_tests"]
        test_result_pairs = row["test_result_pairs"]
        repairs = row["repair_actions"]

        all_tests.update(tests)
        all_results.update(row["results"])
        all_repairs.update(repairs)
        all_test_result_pairs.extend(test_result_pairs)

        if tests:
            for t in tests:
                matching_results = [
                    p.split(" -> ", 1)[1]
                    for p in test_result_pairs
                    if p.startswith(t + " -> ")
                ]
                if matching_results:
                    for r in matching_results:
                        reasoning_paths.append([symptom_name, comp, t, r])
                else:
                    reasoning_paths.append([symptom_name, comp, t])
        else:
            reasoning_paths.append([symptom_name, comp])

        if repairs:
            for r in repairs:
                reasoning_paths.append([symptom_name, comp, r])

    path_strength = _compute_path_strength(reasoning_paths)
    support_count = _compute_support_count(reasoning_paths)
    community_agreement = _compute_community_agreement(driver, reasoning_paths)
    test_agreement = _compute_test_agreement(all_test_result_pairs)

    driver.close()

    evidence_score = round(100 * (
        0.35 * path_strength +
        0.25 * support_count +
        0.20 * community_agreement +
        0.20 * test_agreement
    ), 1)

    return {
        "symptom": symptom_name,
        "diagnostic_tests": sorted(all_tests),
        "results": sorted(all_results),
        "repair_actions": sorted(all_repairs),
        "reasoning_paths": reasoning_paths,
        "top_paths": [p for _, p in sorted((len(p), p) for p in reasoning_paths)[:3]],
        "community": community,
        "community_context": community_context or {},
        "evidence_score": evidence_score,
        "path_strength": round(path_strength, 4),
        "support_count": round(support_count, 4),
        "community_agreement": round(community_agreement, 4),
        "test_agreement": round(test_agreement, 4),
    }


def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python graph_retriever.py <query>")
        sys.exit(1)
    query = " ".join(sys.argv[1:])
    result = smart_retrieve(query)
    print(f"Query: {result.get('raw_query', query)}")
    print(f"Intent: {result.get('intent', 'N/A')}")
    print(f"Matched Symptom: {result['symptom']}")
    if result.get("expanded_queries"):
        print(f"Expanded queries: {', '.join(result['expanded_queries'])}")
    if result.get("community"):
        c = result["community"]
        print(f"Community: {c['name']} (id={c['id']}, size={c['size']})")
        ctx = result.get("community_context", {})
        if ctx.get("symptoms"):
            print(f"  Nearby symptoms: {', '.join(ctx['symptoms'])}")
        if ctx.get("components"):
            print(f"  Nearby components: {', '.join(ctx['components'])}")
    print()
    if result["diagnostic_tests"]:
        print("Diagnostic Tests:")
        for t in result["diagnostic_tests"]:
            print(f"  - {t}")
    else:
        print("Diagnostic Tests: (none)")
    if result["results"]:
        print("Results:")
        for r in result["results"]:
            print(f"  - {r}")
    else:
        print("Results: (none)")
    if result["repair_actions"]:
        print("Repair Actions:")
        for r in result["repair_actions"]:
            print(f"  - {r}")
    else:
        print("Repair Actions: (none)")
    print()
    print(f"Evidence Score: {result['evidence_score']}%")
    print(f"  Path Strength:       {result['path_strength']:.4f}")
    print(f"  Support Count:       {result['support_count']:.4f}")
    print(f"  Community Agreement: {result['community_agreement']:.4f}")
    print(f"  Test Agreement:      {result['test_agreement']:.4f}")
    print()
    print("Top 3 Paths:")
    for path in result["top_paths"]:
        print(f"  {' -> '.join(path)}")


if __name__ == "__main__":
    main()
