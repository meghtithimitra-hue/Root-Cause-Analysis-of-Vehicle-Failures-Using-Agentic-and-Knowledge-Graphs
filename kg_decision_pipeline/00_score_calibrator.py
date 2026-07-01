"""
00_score_calibrator.py

Calibrates hybrid retrieval scores before community expansion. Operates on
the output of hybrid_retrieve() and returns the same dict shape with adjusted
scores.

Calibration steps by candidate source:

    *vector*   – clamp to max(0.0, score)
    *graph*    – unchanged (flat 0.3)
    *both*     – recover the vector component, clamp, apply a saturating
                 multiplicative boost, then combine via noisy-OR with a
                 graph baseline (0.3).

Usage:
    from kg_decision_pipeline.00_score_calibrator import calibrate_scores
    calibrated = calibrate_scores(hybrid_retrieve("brake pedal spongy"))
"""

GRAPH_BASELINE = 0.3
SATURATION_ALPHA = 0.4


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------
def _calibrate_vector(vec_score: float) -> float:
    """Clamp a vector-derived score to [0, ∞)."""
    return max(0.0, vec_score)


def _calibrate_both(score: float) -> float:
    """Calibrate a candidate found by both vector and graph search.

    Steps
    -----
    1. Decompose  – recover the original vector component by subtracting
                     the old flat +0.5 boost.
    2. Clamp      – vector score to [0, ∞).
    3. Saturating  – vec_cal = clamped * (1 + alpha * (1 - clamped)).
                     Gives the biggest relative boost at mid-range scores
                     and near-zero boost when the vector score is already
                     high (certain).
    4. Noisy-OR   – combined = 1 - (1 - vec_cal) * (1 - graph_baseline).
                     Probabilistic fusion treating vector and graph as
                     independent evidence sources.
    """
    vec_raw = score - 0.5
    vec_clamped = _calibrate_vector(vec_raw)
    vec_cal = vec_clamped * (1.0 + SATURATION_ALPHA * (1.0 - vec_clamped))
    combined = 1.0 - (1.0 - vec_cal) * (1.0 - GRAPH_BASELINE)
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
        *source*).

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

        if source == "both":
            entry["score"] = _calibrate_both(score)
        elif source == "vector":
            entry["score"] = _calibrate_vector(score)
        # graph and community sources pass through unchanged (flat 0.3 / 0.15)

        calibrated.append(entry)

    calibrated.sort(key=lambda x: -x["score"])
    return {"query": query, "candidates": calibrated}


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
    print(f"{'Rank':<5} {'Score':<8} {'Source':<10} {'Type':<16} {'Label'}")
    print("-" * 100)
    for i, c in enumerate(cal["candidates"], 1):
        print(
            f"{i:<5} {c['score']:<8.3f} {c['source']:<10} "
            f"{c.get('node_type', ''):<16} {c['label'][:55]}"
        )

    print("\n--- Before vs After ---")
    print(f"{'Rank':<5} {'Source':<10} {'Before':<8} {'After':<8}  Label")
    print("-" * 90)
    for i, (before, after) in enumerate(
        zip(raw["candidates"], cal["candidates"]), 1
    ):
        arrow = "  ← changed" if abs(before["score"] - after["score"]) > 0.001 else ""
        print(
            f"{i:<5} {after['source']:<10} "
            f"{before['score']:<8.3f} {after['score']:<8.3f}  "
            f"{after['label'][:50]}{arrow}"
        )
