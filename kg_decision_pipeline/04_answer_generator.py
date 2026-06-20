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

Usage:
    from kg_decision_pipeline.04_answer_generator import generate_answer
    answer = generate_answer(reasoning_path, scored_result)
"""

import json
import subprocess
from typing import Any

OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_TIMEOUT = 60


# ---------------------------------------------------------------------------
# Graph-only answer formatter
# ---------------------------------------------------------------------------
def _format_graph_answer(path: dict) -> str:
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

    # -- EXTRACTED / INFERRED → graph-only answer ----------------------------
    if mode in ("EXTRACTED", "INFERRED"):
        answer = _format_graph_answer(reasoning_path)
        return {
            "mode": mode,
            "answer": answer,
            "source": "graph",
            "clarifying_question": None,
        }

    # -- AMBIGUOUS + skip → LLM fallback ------------------------------------
    if skip_allowed:
        llm_answer = _call_ollama(query, candidates, model=llm_model)
        return {
            "mode": mode,
            "answer": llm_answer,
            "source": "llm",
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
    import sys
    import importlib
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from scripts.pipeline.hybrid_retrieval import hybrid_retrieve
    expander = importlib.import_module("kg_decision_pipeline.01_community_expander")
    scorer = importlib.import_module("kg_decision_pipeline.02_confidence_scorer")
    path_builder = importlib.import_module("kg_decision_pipeline.03_reasoning_path")

    query = "brake warning light is on and pedal feels soft"
    raw = hybrid_retrieve(query, top_k=10)
    expanded = expander.expand_candidates(raw)
    scored = scorer.score_candidates(expanded)
    rpath = path_builder.build_reasoning_path(scored)
    answer = generate_answer(rpath, scored)

    print(f"Mode: {answer['mode']}")
    print(f"Source: {answer['source']}")
    print(f"Clarifying question: {answer.get('clarifying_question')}")
    print(f"\nAnswer:\n{answer['answer']}")
