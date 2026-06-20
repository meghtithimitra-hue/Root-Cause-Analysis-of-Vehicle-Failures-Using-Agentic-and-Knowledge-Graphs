"""
02_confidence_scorer.py

Takes the expanded candidate list from expand_candidates() and produces a
confidence score, per-candidate mode tag, and an overall decision mode for
the retrieval result.  Supports LLM fallback via Ollama when the result
is AMBIGUOUS.

Modes (per candidate, based on source × score):
    +------------------+-----------+------------------+
    | source           | score     | tag              |
    +------------------+-----------+------------------+
    | "both"           |  > 1.0    | EXTRACTED        |
    | "both"           |  0.6–1.0  | INFERRED         |
    | anything else    |  any      | AMBIGUOUS        |
    +------------------+-----------+------------------+

Overall mode uses the top candidate's tag.  When mode is AMBIGUOUS a
clarifying question is generated; if skip_allowed is True or the question
cannot resolve, the pipeline falls back to llama3.1:8b via Ollama.

Usage:
    from kg_decision_pipeline import expand_candidates, score_candidates
    expanded = expand_candidates(hybrid_retrieve("brake pedal spongy"))
    result = score_candidates(expanded, skip_allowed=False)
"""

import json
import subprocess
from typing import Any

OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_TIMEOUT = 60


# ---------------------------------------------------------------------------
# Candidate tagging
# ---------------------------------------------------------------------------
def _tag_candidate(candidate: dict) -> str:
    """Return EXTRACTED / INFERRED / AMBIGUOUS for a single candidate."""
    source = candidate.get("source", "")
    score = candidate.get("score", 0.0)

    if source == "both" and score > 1.0:
        return "EXTRACTED"
    if source == "both" and score >= 0.6:
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
        f"The query \"{query}\" matched ambiguous results "
        f"(top labels: {labels_text}). "
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


def _format_candidates_for_llm(candidates: list[dict]) -> str:
    rows = []
    for i, c in enumerate(candidates[:5], 1):
        rows.append(
            f"  {i}. [{c.get('source','?')}] "
            f"score={c.get('score',0):.3f}  "
            f"type={c.get('node_type','?')}  "
            f"label={c.get('label','?')}"
        )
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def score_candidates(
    expanded_result: dict,
    skip_allowed: bool = False,
    llm_model: str = OLLAMA_MODEL,
) -> dict[str, Any]:
    """Score and tag retrieval candidates, determine overall mode, and
    optionally invoke LLM fallback.

    Parameters
    ----------
    expanded_result : dict
        Output of *expand_candidates()* — must have keys *query* (str) and
        *candidates* (list of dicts with at least *score*, *source*, *label*).
    skip_allowed : bool
        If True, bypass candidate-scoring and go straight to LLM fallback.
    llm_model : str
        Ollama model tag to use for fallback.

    Returns
    -------
    dict with keys:
        mode                str   — "EXTRACTED" | "INFERRED" | "AMBIGUOUS"
        confidence          float — top candidate's score (or 0.0 if empty)
        candidates_scored   list  — each candidate with an added "tag" field
        clarifying_question str | None
        skip_allowed        bool
        llm_fallback_answer str | None
    """
    query = expanded_result["query"]
    candidates = expanded_result["candidates"]

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
        tagged = dict(c)  # shallow copy
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
    confidence = top.get("score", 0.0)
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
    import sys
    import importlib
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from scripts.pipeline.hybrid_retrieval import hybrid_retrieve
    expander = importlib.import_module(
        "kg_decision_pipeline.01_community_expander"
    )
    expand_candidates = expander.expand_candidates

    query = "brake warning light is on and pedal feels soft"
    raw = hybrid_retrieve(query, top_k=10)
    expanded = expand_candidates(raw)
    result = score_candidates(expanded)

    print(f"Query: {result.get('mode', '?')}  "
          f"(confidence={result.get('confidence', 0):.3f})")
    print(f"Mode: {result['mode']}")
    print(f"Skip allowed: {result['skip_allowed']}")
    if result.get("clarifying_question"):
        print(f"Clarifying question: {result['clarifying_question']}")
    if result.get("llm_fallback_answer"):
        print(f"LLM fallback: {result['llm_fallback_answer'][:200]}...")
    print(f"\n{'Rank':<5} {'Tag':<12} {'Score':<6} {'Source':<10} {'Label'}")
    print("-" * 100)
    for i, c in enumerate(result.get("candidates_scored", []), 1):
        print(
            f"{i:<5} {c.get('tag','?'):<12} {c['score']:<6.3f} "
            f"{c.get('source','?'):<10} {c['label'][:55]}"
        )
