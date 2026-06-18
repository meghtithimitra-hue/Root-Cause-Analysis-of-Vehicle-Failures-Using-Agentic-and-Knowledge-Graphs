"""
Automotive GraphRAG reasoning engine.

Converts community-aware graph retrieval results into a structured diagnosis
with evidence scores, community context, and ranked reasoning paths.
"""

from dataclasses import dataclass
from typing import Any

_FAULT_KEYWORDS = frozenset({
    "worn", "faulty", "damaged", "low", "contaminated", "blown",
    "leaking", "scored", "warped", "clogged", "cracked", "stuck",
    "broken", "dirty", "failed", "empty", "bad", "dead", "loose",
    "seized", "high", "excessive", "insufficient", "abnormal",
    "noisy", "slow", "weak", "burned", "corroded", "blocked",
    "incorrect", "misaligned", "slipping", "intermittent",
    "rotting", "frozen", "jammed",
})


@dataclass
class ScoredPath:
    path: list[str]
    score: int
    summary: str = ""


def score_path(
    path: list[str],
    fault_conditions: set[str],
    repair_actions: set[str],
) -> int:
    if not path:
        return 0
    score = 1
    last = path[-1]
    if last in repair_actions:
        score += 3
    has_fault = any(
        set(node.lower().split()) & fault_conditions
        for node in path
    )
    if has_fault:
        score += 2
    if len(path) >= 4:
        score += 1
    return score


def rank_paths(
    paths: list[list[str]],
    fault_conditions: set[str],
    repair_actions: set[str],
    top_n: int = 3,
) -> list[ScoredPath]:
    with_scores = []
    for p in paths:
        s = score_path(p, fault_conditions, repair_actions)
        last_in_repairs = p[-1] in repair_actions
        has_fault_last = bool(set(p[-1].lower().split()) & fault_conditions)
        with_scores.append((s, has_fault_last and not last_in_repairs, -len(p), p))
    with_scores.sort(key=lambda x: (-x[0], -x[1], x[2]))
    return [ScoredPath(path=p, score=s) for s, _, _, p in with_scores[:top_n]]


def extract_fault_conditions(results: list[str]) -> set[str]:
    found: set[str] = set()
    for r in results:
        for word in r.lower().split():
            stripped = word.strip(",.!?;:'\"()[]{}")
            if stripped in _FAULT_KEYWORDS:
                found.add(stripped)
    return found


# ── Helper utilities ───────────────────────────────────────────────────────

def _derive_cause(top_paths: list[ScoredPath]) -> str:
    if not top_paths:
        return "Unknown"
    candidates = []
    for sp in top_paths:
        for i, node in enumerate(sp.path):
            words = set(node.lower().split())
            if words & _FAULT_KEYWORDS:
                candidates.append((sp.score, i, node))
    if candidates:
        candidates.sort(key=lambda x: (-x[0], x[1]))
        return candidates[0][2]
    top = top_paths[0].path
    return top[1] if len(top) >= 2 else top[0]


def _derive_repair(top_paths: list[ScoredPath], retrieval_result: dict) -> str:
    for sp in top_paths:
        if sp.path[-1] in retrieval_result.get("repair_actions", []):
            return sp.path[-1]
    for sp in top_paths:
        if len(sp.path) >= 2:
            comp = sp.path[1]
            for r in retrieval_result.get("repair_actions", []):
                if comp.lower() in r.lower():
                    return r
    repairs = retrieval_result.get("repair_actions", [])
    return repairs[0] if repairs else "Further diagnosis required"


def _build_reasoning_summary(
    symptom: str,
    top_paths: list[ScoredPath],
    retrieval_result: dict,
) -> str:
    community = retrieval_result.get("community")
    if community:
        lines = [f"{symptom} appears in the {community['name']} community."]
    else:
        lines = [f"{symptom} has no community assignment."]
    if top_paths:
        comps = set()
        for sp in top_paths:
            if len(sp.path) >= 2:
                comps.add(sp.path[1])
        if len(comps) == 1:
            lines.append(f"Multiple diagnostic paths converge on {list(comps)[0]}.")
        elif comps:
            lines.append(
                f"Diagnostic paths implicate {', '.join(sorted(comps))}."
            )
        lines.append(
            f"The top-ranked path has a score of {top_paths[0].score}."
        )
    return " ".join(lines)


# ── Public API ─────────────────────────────────────────────────────────────

def generate_explanation(result: dict) -> str:
    symptom = result.get("symptom", "Unknown symptom")
    cause = result.get("most_likely_cause", "Unknown")
    repair = result.get("recommended_repair", "Unknown")
    evidence = result.get("evidence_score", 0)
    community = result.get("community", {})
    summary = result.get("reasoning_summary", "")
    top_paths: list[ScoredPath] = result.get("top_reasoning_paths", [])
    vs = result.get("vector_similarity", 0)
    ps = result.get("path_strength", 0)
    sc = result.get("support_count", 0)
    ca = result.get("community_agreement", 0)
    ta = result.get("test_agreement", 0)
    intent = result.get("intent", "")
    expanded = result.get("expanded_queries", [])

    lines = [
        "=" * 54,
        "  AUTOMOTIVE DIAGNOSTIC REPORT",
        "=" * 54,
        "",
        f"Symptom:",
        f"  {symptom}",
    ]
    if intent:
        lines += [f"Detected Intent:  {intent}"]
    if community:
        lines += ["", f"Community:", f"  {community.get('name', 'N/A')}"]
    lines += [
        "",
        f"Most Likely Cause:",
        f"  {cause}",
        "",
        f"Recommended Repair:",
        f"  {repair}",
        "",
        f"Evidence Score:",
        f"  {evidence:.0f}%",
        "",
        "Evidence Breakdown:",
        f"  Vector Similarity:    {vs:.4f}  (weight 0.30)",
        f"  Path Strength:        {ps:.4f}  (weight 0.25)",
        f"  Community Agreement:  {ca:.4f}  (weight 0.20)",
        f"  Support Count:        {sc:.4f}  (weight 0.15)",
        f"  Test Agreement:       {ta:.4f}  (weight 0.10)",
    ]
    if expanded:
        lines += ["", "Expanded Queries:"]
        for eq in expanded:
            lines.append(f"  - {eq}")
    lines += [
        "",
        "Reasoning Summary:",
        f"  {summary}",
        "",
        "Top diagnostic paths:",
    ]
    for i, sp in enumerate(top_paths, 1):
        lines.append(f"  [{sp.score}] {' -> '.join(sp.path)}")
    lines += ["", "-" * 54]
    return "\n".join(lines)


def explain_diagnosis(retrieval_result: dict) -> dict:
    if not retrieval_result:
        return {
            "most_likely_cause": "No data available",
            "recommended_repair": "Further diagnosis required",
            "evidence_score": 0,
            "community": None,
            "reasoning_summary": "No retrieval data provided.",
            "top_reasoning_paths": [],
            "explanation": "No retrieval data provided.",
        }

    symptom = retrieval_result.get("symptom", "Unknown")
    results: list[str] = retrieval_result.get("results", [])
    repair_actions: set[str] = set(retrieval_result.get("repair_actions", []))
    reasoning_paths: list[list[str]] = retrieval_result.get("reasoning_paths", [])
    community = retrieval_result.get("community")

    if not reasoning_paths:
        return {
            "most_likely_cause": f"{symptom} — insufficient data",
            "recommended_repair": "Further diagnosis required",
            "evidence_score": 0,
            "community": community,
            "reasoning_summary": f"No reasoning paths found for '{symptom}'.",
            "top_reasoning_paths": [],
            "explanation": f"No reasoning paths found for symptom '{symptom}'.",
        }

    fault_conditions = extract_fault_conditions(results)
    top_paths = rank_paths(reasoning_paths, fault_conditions, repair_actions)
    cause = _derive_cause(top_paths)
    repair = _derive_repair(top_paths, retrieval_result)
    summary = _build_reasoning_summary(symptom, top_paths, retrieval_result)

    result = {
        "symptom": symptom,
        "most_likely_cause": cause,
        "recommended_repair": repair,
        "evidence_score": retrieval_result.get("evidence_score", 0),
        "vector_similarity": retrieval_result.get("vector_similarity", 0),
        "vector_matches": retrieval_result.get("vector_matches", []),
        "path_strength": retrieval_result.get("path_strength", 0),
        "support_count": retrieval_result.get("support_count", 0),
        "community_agreement": retrieval_result.get("community_agreement", 0),
        "test_agreement": retrieval_result.get("test_agreement", 0),
        "community": community,
        "reasoning_summary": summary,
        "top_reasoning_paths": top_paths,
        "intent": retrieval_result.get("intent"),
        "raw_query": retrieval_result.get("raw_query"),
        "expanded_queries": retrieval_result.get("expanded_queries"),
        "matched_symptom": retrieval_result.get("matched_symptom"),
        "community_members": retrieval_result.get("community_members"),
        "match_score": retrieval_result.get("match_score"),
    }
    result["explanation"] = generate_explanation(result)
    return result


def main():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from graph_retriever import retrieve_symptom

    symptom = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Brake pedal pulsation"
    retrieval = retrieve_symptom(symptom)
    diagnosis = explain_diagnosis(retrieval)
    print(diagnosis["explanation"])


if __name__ == "__main__":
    main()
