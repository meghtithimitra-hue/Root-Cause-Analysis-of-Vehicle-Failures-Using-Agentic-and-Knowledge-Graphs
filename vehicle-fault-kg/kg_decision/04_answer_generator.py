"""
04_answer_generator.py

Generates the final answer from the reasoning path and mode decision.

If mode is EXTRACTED or INFERRED, the answer is constructed from the graph
context alone (reasoning chain, diagnosis steps, matched symptoms) without
calling an LLM.

If mode is AMBIGUOUS and *skip_allowed* is False, the clarifying question
from the scorer is returned as-is.

If mode is AMBIGUOUS and *skip_allowed* is True, the module calls Ollama
(``llama3.1:8b``) via HTTP POST and returns the LLM response.

Reused from the original kg_decision_pipeline with no contract changes.
The mode-based dispatch, graph answer formatting, and LLM fallback are all
independent of how candidates were retrieved.

Usage:
    import importlib
    _gen = importlib.import_module("kg_decision.04_answer_generator")
    answer = _gen.generate_answer(reasoning_path, scored_result)
"""

import json
import subprocess
from typing import Any

OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_TIMEOUT = 60


# ---------------------------------------------------------------------------
# Graph-only answer formatter
# ---------------------------------------------------------------------------
def format_graph_answer(path: dict) -> str:
    """Build a structured answer using only the reasoning-path content."""
    lines = []

    cat = path.get("top_category")
    sub = path.get("top_subcategory")
    symptoms = path.get("matched_symptoms", [])
    steps = path.get("diagnosis_steps", [])
    chain = path.get("reasoning_chain", [])

    if cat:
        lines.append(f"System: {cat}")
    if sub:
        lines.append(f"Subsystem: {sub}")

    if symptoms:
        lines.append("")
        lines.append("Matched symptoms:")
        for s in symptoms:
            lines.append(f"  - {s}")

    if steps:
        lines.append("")
        lines.append("Recommended diagnosis steps:")
        for d in steps:
            lines.append(f"  - {d}")

    if chain:
        lines.append("")
        lines.append("Reasoning:")
        for i, step in enumerate(chain, 1):
            lines.append(f"  {i}. {step}")

    return "\n".join(lines).strip()


def _format_subsystem_summary(path: dict) -> str:
    """Brief summary showing only the matched system and subsystem."""
    lines = []
    cat = path.get("top_category")
    sub = path.get("top_subcategory")
    if cat:
        lines.append(f"System: {cat}")
    if sub:
        lines.append(f"Subsystem: {sub}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Ollama caller
# ---------------------------------------------------------------------------
def _call_ollama(
    query: str,
    candidates: list[dict],
    model: str = OLLAMA_MODEL,
) -> str:
    """Call ``ollama serve`` with the query and top candidate labels."""
    top_n = candidates[:5]
    labels = [c.get("label", "?") for c in top_n]
    context = "\n".join(f"  {i}. {l}" for i, l in enumerate(labels, 1))

    prompt = (
        "You are an automotive diagnostic assistant.\n\n"
        f"Query: {query}\n\n"
        f"Top retrieved candidates:\n{context}\n\n"
        "Provide a concise diagnosis: most likely root cause and "
        "recommended next step."
    )

    payload = json.dumps({"model": model, "prompt": prompt, "stream": False})

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
def generate_answer(
    reasoning_path: dict,
    scored_result: dict,
    llm_model: str = OLLAMA_MODEL,
) -> dict[str, Any]:
    """Generate the final answer based on mode and reasoning path.

    Parameters
    ----------
    reasoning_path : dict
        Output of *build_reasoning_path()* — must have keys *query*, *mode*,
        *top_subcategory*, *top_category*, *matched_symptoms*,
        *diagnosis_steps*, *reasoning_chain*.
    scored_result : dict
        Output of *score_candidates()* — must have keys *mode*,
        *skip_allowed*, *clarifying_question*, and *candidates_scored*.
    llm_model : str
        Ollama model tag for LLM fallback.

    Returns
    -------
    dict with keys:
        mode                str
        answer              str
        source              "graph" | "llm"
        clarifying_question str | None
    """
    mode = scored_result.get("mode", "AMBIGUOUS")
    skip_allowed = scored_result.get("skip_allowed", False)
    clarifying_question = scored_result.get("clarifying_question")
    query = reasoning_path.get("query", "")
    candidates = scored_result.get("candidates_scored", [])

    # -- EXTRACTED → full graph answer ---------------------------------------
    if mode == "EXTRACTED":
        answer = format_graph_answer(reasoning_path)
        return {
            "mode": mode,
            "answer": answer,
            "source": "graph",
            "clarifying_question": None,
            "is_intermediate": False,
        }

    # -- INFERRED → intermediate if unconfirmed symptoms exist ---------------
    if mode == "INFERRED":
        unconfirmed = reasoning_path.get("unconfirmed_symptoms", [])
        if unconfirmed:
            return {
                "mode": mode,
                "answer": _format_subsystem_summary(reasoning_path),
                "source": "graph",
                "clarifying_question": None,
                "is_intermediate": True,
                "unconfirmed_symptoms": unconfirmed,
                "top_category": reasoning_path.get("top_category"),
                "top_subcategory": reasoning_path.get("top_subcategory"),
                "matched_symptoms": reasoning_path.get("matched_symptoms", []),
                "diagnosis_steps": reasoning_path.get("diagnosis_steps", []),
                "reasoning_chain": reasoning_path.get("reasoning_chain", []),
            }
        # No unconfirmed symptoms → show full answer like EXTRACTED
        answer = format_graph_answer(reasoning_path)
        return {
            "mode": mode,
            "answer": answer,
            "source": "graph",
            "clarifying_question": None,
            "is_intermediate": False,
        }

    # -- AMBIGUOUS + skip → no-match response --------------------------------
    FIXED_NO_MATCH = (
        "No confident match found in the knowledge graph. "
        "Please describe your vehicle fault in more detail "
        "\u2014 include the symptom, when it occurs and any warning lights."
    )
    if skip_allowed:
        return {
            "mode": mode,
            "answer": FIXED_NO_MATCH,
            "source": "no_match",
            "llm_fallback_answer": FIXED_NO_MATCH,
            "clarifying_question": None,
        }

    # -- AMBIGUOUS + no skip → clarifying question ---------------------------
    fallback = clarifying_question or (
        "The query is ambiguous — please provide more detail "
        "(e.g. symptom type, when it occurs, warning lights)."
    )
    return {
        "mode": mode,
        "answer": fallback,
        "source": "graph",
        "clarifying_question": fallback,
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
    _scr = importlib.import_module("kg_decision.02_confidence_scorer")
    _rpb = importlib.import_module("kg_decision.03_reasoning_path")
    calibrate_scores = _cal.calibrate_scores
    score_candidates = _scr.score_candidates
    build_reasoning_path = _rpb.build_reasoning_path

    query = "brake warning light is on and pedal feels soft"
    raw = hybrid_retrieve(query, top_k=10)
    calibrated = calibrate_scores(raw)
    scored = score_candidates(calibrated)
    rpath = build_reasoning_path(scored)
    answer = generate_answer(rpath, scored)

    print(f"Mode: {answer['mode']}")
    print(f"Source: {answer['source']}")
    print(f"Clarifying question: {answer.get('clarifying_question')}")
    print(f"\nAnswer:\n{answer['answer']}")
