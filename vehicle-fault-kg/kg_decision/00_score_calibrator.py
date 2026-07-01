"""
00_score_calibrator.py

Calibrates hybrid retrieval scores before community expansion. Operates on
the output of hybrid_retrieve() and returns the same dict shape with adjusted
scores.

Generalized for the new 7-source retrieval pipeline (vehicle-fault-kg).
Preserves the original mathematical core:
  1. Clamp negative vector scores to 0.
  2. Saturating multiplicative boost:  vec_cal = clamped * (1 + alpha * (1 - clamped)).
  3. Noisy-OR fusion with a baseline:  combined = 1 - (1 - vec_cal) * (1 - baseline).

Source-specific offsets recover the vector component from the boosted score.
Offsets match the boost values in vehicle-fault-kg/scripts/pipeline/hybrid_retrieval.py.

Usage:
    import importlib
    _cal = importlib.import_module("kg_decision.00_score_calibrator")
    calibrated = _cal.calibrate_scores(hybrid_retrieve("brake pedal spongy"))
"""

# ---------------------------------------------------------------------------
# Calibration constants
# ---------------------------------------------------------------------------
GRAPH_BASELINE = 0.3       # noisy-OR baseline for graph-only evidence
COMMUNITY_BASELINE = 0.4   # noisy-OR baseline for community-only evidence
SATURATION_ALPHA = 0.4     # saturating boost coefficient

# Score decomposition offsets — each source type's boost amount
# Used to recover the original vector component: vec_raw = score - offset
#
# WARNING: The "all" offset of 1.3 matches the *cumulative* boost applied by
# hybrid_retrieval.py's merge loop: vec + 0.5 (graph) + 0.8 (community-all)
# = vec + 1.3.  The README in scripts/pipeline/ says "base + 0.8 boost", which
# is the community-step increment only.  If the retrieval is ever aligned with
# its own README, this offset MUST change to 0.8.
SOURCE_OFFSETS = {
    "vector+graph":       0.5,   # graph boost from hybrid_retrieval.py L253
    "vector+community":   0.3,   # community boost from hybrid_retrieval.py L259
    "all":                1.3,   # 0.5 (graph) + 0.8 (community-all) — see WARNING above
}

# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------
def _calibrate_vector(vec_score: float) -> float:
    """Clamp a vector-derived score to [0, ∞)."""
    return max(0.0, vec_score)


def _calibrate_combined(score: float, offset: float, baseline: float) -> float:
    """Calibrate a candidate found by vector + one or more structural paths.

    Steps
    -----
    1. Decompose  – recover the original vector component via `score - offset`.
    2. Clamp      – vector score to [0, ∞).
    3. Saturating – vec_cal = clamped * (1 + alpha * (1 - clamped)).
                    Gives the biggest relative boost at mid-range scores
                    and near-zero boost when the vector score is already high.
    4. Noisy-OR   – combined = 1 - (1 - vec_cal) * (1 - baseline).
                    Probabilistic fusion treating vector and structural
                    evidence as independent sources.
    """
    vec_raw = score - offset
    vec_clamped = _calibrate_vector(vec_raw)
    vec_cal = vec_clamped * (1.0 + SATURATION_ALPHA * (1.0 - vec_clamped))
    combined = 1.0 - (1.0 - vec_cal) * (1.0 - baseline)
    return combined


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def calibrate_scores(hybrid_result: dict) -> dict:
    """Calibrate the score field of every candidate in a hybrid_retrieve()
    output dictionary.

    Parameters
    ----------
    hybrid_result : dict
        Output of *hybrid_retrieve()* — must have keys *query* (str) and
        *candidates* (list of dicts each containing at least *score* and
        *source*).  Extra keys (*source_breakdown*, *retrieval_stats*) are
        forwarded unchanged.

    Returns
    -------
    dict
        Same structure as the input with adjusted *score* values.
        Candidates remain sorted by score descending.
    """
    query = hybrid_result["query"]
    candidates = hybrid_result["candidates"]

    calibrated = []
    for c in candidates:
        entry = dict(c)
        source = entry.get("source", "")
        score = entry.get("score", 0.0)

        if source == "vector":
            entry["score"] = _calibrate_vector(score)

        elif source == "vector+graph":
            entry["score"] = _calibrate_combined(
                score, offset=SOURCE_OFFSETS["vector+graph"],
                baseline=GRAPH_BASELINE
            )

        elif source == "vector+community":
            entry["score"] = _calibrate_combined(
                score, offset=SOURCE_OFFSETS["vector+community"],
                baseline=COMMUNITY_BASELINE
            )

        elif source == "all":
            # Combined baseline: both graph and community contributed
            all_baseline = 1.0 - (1.0 - GRAPH_BASELINE) * (1.0 - COMMUNITY_BASELINE)
            entry["score"] = _calibrate_combined(
                score, offset=SOURCE_OFFSETS["all"],
                baseline=all_baseline
            )

        # else: "graph", "community", "graph+community" pass through unchanged
        #   - "graph" is always a flat 0.3 (no vector component to calibrate)
        #   - "community" is already 0.4/0.6 (no vector component)
        #   - "graph+community" is 0.3 + 0.2 = 0.5 (structural only, no vector)

        calibrated.append(entry)

    calibrated.sort(key=lambda x: -x["score"])
    result = {"query": query, "candidates": calibrated}
    # Forward any extra keys from the new retrieval output
    for key in ("source_breakdown", "retrieval_stats"):
        if key in hybrid_result:
            result[key] = hybrid_result[key]
    return result


# ---------------------------------------------------------------------------
# __main__ smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.pipeline.hybrid_retrieval import hybrid_retrieve

    query = "brake pedal feels spongy when pressed"
    raw = hybrid_retrieve(query, top_k=10)
    cal = calibrate_scores(raw)

    print(f"Query: {cal['query']}\n")
    print(f"{'Rank':<5} {'Score':<8} {'Source':<20} {'Type':<16} {'Label'}")
    print("-" * 100)
    for i, c in enumerate(cal["candidates"], 1):
        print(
            f"{i:<5} {c['score']:<8.3f} {c['source']:<20} "
            f"{c.get('node_type', ''):<16} {c['label'][:55]}"
        )

    print("\n--- Before vs After ---")
    print(f"{'Rank':<5} {'Source':<20} {'Before':<8} {'After':<8}  Label")
    print("-" * 90)
    for i, (before, after) in enumerate(
        zip(raw["candidates"], cal["candidates"]), 1
    ):
        arrow = "  ← changed" if abs(before["score"] - after["score"]) > 0.001 else ""
        print(
            f"{i:<5} {after['source']:<20} "
            f"{before['score']:<8.3f} {after['score']:<8.3f}  "
            f"{after['label'][:50]}{arrow}"
        )
