"""
02_confidence_scorer.py

Takes the calibrated candidate list from calibrate_scores() and produces a
confidence score, per-candidate mode tag, and an overall decision mode for
the retrieval result.

This redesigned scorer separates *ranking* from *decision*:
  - The calibrator produces the ranking score (calibrated relevance).
  - The scorer computes an *effective confidence* by combining the calibrated
    score with a *path-confidence factor* derived from the retrieval evidence
    (found_by, source, community metadata).

The path factor discounts evidence that is structurally correlated (e.g.,
graph and community both derive from the same graph) or comes from a single
path.  This prevents noisy-OR overconfidence when evidence sources are not
fully independent.

Modes (per candidate):
  +----------------------+-----------+------------------+
  | source               | effective | tag              |
  +----------------------+-----------+------------------+
  | all (3 paths)        |  >= 0.75  | EXTRACTED        |
  | vector+graph         |  >= 0.75  | EXTRACTED        |
  | vector+community     |  >= 0.75  | EXTRACTED        |
  | any multi-path       |  0.55-0.74| INFERRED         |
  | multi-cat community  |  >= 0.55  | INFERRED         |
  | single-path          |  < 0.55   | AMBIGUOUS        |
  +----------------------+-----------+------------------+

Supports LLM fallback via Ollama when the result is AMBIGUOUS.

Usage:
    import importlib
    _scr = importlib.import_module("kg_decision.02_confidence_scorer")
    result = _scr.score_candidates(calibrated_result, skip_allowed=False)
"""

import json
import subprocess
from typing import Any

OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_TIMEOUT = 60

# ---------------------------------------------------------------------------
# Path-confidence factors
#
# This reflects how much independent evidence the retrieval paths provide.
#
# NOTE: The path factors here work in concert with the calibrator offsets in
# 00_score_calibrator.py.  If the retrieval's cumulative boost behavior
# changes (see WARNING in 00_score_calibrator.py::SOURCE_OFFSETS["all"]),
# the effective-confidence values and mode thresholds may need adjustment.
# ---------------------------------------------------------------------------
PATH_CONFIDENCE = {
    "all":                1.00,   # 3 independent paths — highest confidence
    "vector+graph":       0.95,   # vector (semantic) + graph (structural) — orthogonal, near-full confidence
    "vector+community":   0.85,   # vector + community (partially correlated w/ graph)
    "graph+community":    0.70,   # both structural — correlated, no semantic evidence
    "vector":             0.55,   # single semantic path only
    "community":          0.45,   # single community path only
    "graph":              0.35,   # single structural path only
}

# Bonus for multi-category communities (they represent genuine cross-system
# fault relationships and are higher quality clusters).
MULTI_CAT_COMMUNITY_BONUS = 0.10

# Thresholds for mode assignment (applied to effective_confidence)
EXTRACTED_THRESHOLD = 0.75
INFERRED_THRESHOLD = 0.55


# ---------------------------------------------------------------------------
# Candidate tagging
# ---------------------------------------------------------------------------
def _effective_confidence(candidate: dict) -> float:
    """Compute effective confidence = calibrated_score x path factor.

    The path factor is looked up from PATH_CONFIDENCE by source, with an
    optional bonus for multi-category community candidates.
    """
    source = candidate.get("source", "")
    score = candidate.get("score", 0.0)
    is_multi = candidate.get("is_multi_category", "False") == "True"
    # Also accept boolean from community-map if present
    if isinstance(candidate.get("is_multi_category"), bool):
        is_multi = candidate["is_multi_category"]

    pc = PATH_CONFIDENCE.get(source, 0.30)
    if source == "community" and is_multi:
        pc += MULTI_CAT_COMMUNITY_BONUS

    return score * pc


def _tag_candidate(candidate: dict) -> str:
    """Return EXTRACTED / INFERRED / AMBIGUOUS for a single candidate."""
    effective = _effective_confidence(candidate)

    if effective >= EXTRACTED_THRESHOLD:
        return "EXTRACTED"
    if effective >= INFERRED_THRESHOLD:
        return "INFERRED"
    return "AMBIGUOUS"


# ---------------------------------------------------------------------------
# Clarifying-question generation
# ---------------------------------------------------------------------------
def _generate_clarifying_question(query: str, candidates: list[dict]) -> str:
    """Build a clarifying question when the result is AMBIGUOUS.

    Uses the top candidate labels to identify what details are missing.
    """
    top_labels = [c.get("label", "") for c in candidates[:3] if c.get("label")]
    labels_text = ", ".join(f'"{l}"' for l in top_labels) if top_labels else "(none)"

    return (
        f'The query "{query}" matched ambiguous results '
        f'(top labels: {labels_text}). '
        "Can you provide more detail — e.g. type of symptom, "
        "when it occurs (starting/idling/driving), any warning lights, "
        "or recent repairs?"
    )


# ---------------------------------------------------------------------------
# LLM fallback via Ollama
# ---------------------------------------------------------------------------
def _call_ollama(
    query: str,
    candidates: list[dict],
    model: str = OLLAMA_MODEL,
) -> str:
    """Call ollama serve with the query and top candidates as context."""
    top_n = candidates[:5]
    rows = []
    for i, c in enumerate(top_n, 1):
        rows.append(
            f"  {i}. [{c.get('source','?')}] "
            f"score={c.get('score',0):.3f}  "
            f"type={c.get('node_type','?')}  "
            f"label={c.get('label','?')}"
        )
    candidates_str = "\n".join(rows)

    prompt = (
        "You are an automotive diagnostic assistant. "
        "Given a fault query and retrieved knowledge-graph candidates, "
        "provide the most likely root cause and a recommended "
        "diagnosis step.\n\n"
        f"Query: {query}\n\n"
        f"Top candidates:\n{candidates_str}\n\n"
        "Answer concisely: root cause first, then diagnosis step."
    )

    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
    })

    try:
        result = subprocess.run(
            [
                "curl", "-s", "--max-time", str(OLLAMA_TIMEOUT),
                "http://localhost:11434/api/generate",
                "-d", payload,
            ],
            capture_output=True, text=True, timeout=OLLAMA_TIMEOUT + 5,
        )
        if result.returncode != 0:
            return f"LLM call failed: {result.stderr.strip() or 'unknown error'}"
        data = json.loads(result.stdout)
        return data.get("response", "No response from model.")
    except FileNotFoundError:
        return "curl not found — install curl or configure an alternate HTTP client."
    except Exception as exc:
        return f"LLM fallback error: {exc}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def score_candidates(
    calibrated_result: dict,
    skip_allowed: bool = False,
    llm_model: str = OLLAMA_MODEL,
) -> dict[str, Any]:
    """Score and tag retrieval candidates, determine overall mode, and
    optionally invoke LLM fallback.

    Parameters
    ----------
    calibrated_result : dict
        Output of *calibrate_scores()* — must have keys *query* (str) and
        *candidates* (list of dicts with at least *score*, *source*, *label*,
        *node_type*, *is_multi_category*).
    skip_allowed : bool
        If True, bypass candidate-scoring and go straight to LLM fallback.
    llm_model : str
        Ollama model tag to use for fallback.

    Returns
    -------
    dict with keys:
        mode                str   — "EXTRACTED" | "INFERRED" | "AMBIGUOUS"
        confidence          float — top candidate's effective confidence (or 0.0)
        candidates_scored   list  — each candidate with an added "tag" field
        clarifying_question str | None
        skip_allowed        bool
        llm_fallback_answer str | None
    """
    query = calibrated_result["query"]
    candidates = calibrated_result["candidates"]

    # -- Skip bypass: go straight to LLM ------------------------------------
    if skip_allowed:
        llm_answer = _call_ollama(query, candidates, model=llm_model)
        return {
            "mode": "AMBIGUOUS",
            "confidence": 0.0,
            "candidates_scored": [],
            "clarifying_question": None,
            "skip_allowed": True,
            "llm_fallback_answer": llm_answer,
        }

    # -- Tag every candidate ------------------------------------------------
    candidates_scored = []
    for c in candidates:
        tagged = dict(c)
        tagged["tag"] = _tag_candidate(c)
        candidates_scored.append(tagged)

    # -- Overall mode (based on top candidate) ------------------------------
    if not candidates_scored:
        return {
            "mode": "AMBIGUOUS",
            "confidence": 0.0,
            "candidates_scored": [],
            "clarifying_question": _generate_clarifying_question(query, []),
            "skip_allowed": False,
            "llm_fallback_answer": None,
        }

    top = candidates_scored[0]
    confidence = _effective_confidence(top)
    mode = top["tag"]

    clarifying_question = None
    llm_answer = None

    if mode == "AMBIGUOUS":
        clarifying_question = _generate_clarifying_question(query, candidates_scored)
        llm_answer = _call_ollama(query, candidates_scored, model=llm_model)

    return {
        "mode": mode,
        "confidence": confidence,
        "candidates_scored": candidates_scored,
        "clarifying_question": clarifying_question,
        "skip_allowed": False,
        "llm_fallback_answer": llm_answer,
    }


# ---------------------------------------------------------------------------
# __main__ smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import importlib
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.pipeline.hybrid_retrieval import hybrid_retrieve
    _cal = importlib.import_module("kg_decision.00_score_calibrator")
    calibrate_scores = _cal.calibrate_scores

    query = "brake warning light is on and pedal feels soft"
    raw = hybrid_retrieve(query, top_k=10)
    calibrated = calibrate_scores(raw)
    result = score_candidates(calibrated)

    print(f"Query: {result.get('mode', '?')}  "
          f"(confidence={result.get('confidence', 0):.3f})")
    print(f"Mode: {result['mode']}")
    print(f"Skip allowed: {result['skip_allowed']}")
    if result.get("clarifying_question"):
        print(f"Clarifying question: {result['clarifying_question']}")
    if result.get("llm_fallback_answer"):
        print(f"LLM fallback: {result['llm_fallback_answer'][:200]}...")
    print(f"\n{'Rank':<5} {'Tag':<12} {'Score':<7} {'Eff':<7} {'Source':<20} {'Label'}")
    print("-" * 110)
    for i, c in enumerate(result.get("candidates_scored", []), 1):
        tag = c.get('tag', '?')
        score = c.get('score', 0.0)
        eff = _effective_confidence(c)
        print(
            f"{i:<5} {tag:<12} {score:<7.3f} {eff:<7.3f} "
            f"{c.get('source','?'):<20} {c['label'][:55]}"
        )
