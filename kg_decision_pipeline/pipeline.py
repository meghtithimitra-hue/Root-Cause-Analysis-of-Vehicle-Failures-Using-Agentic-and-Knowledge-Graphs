"""
pipeline.py

Single entry point that chains all four stages of the KG decision pipeline:

    1. hybrid_retrieve()          – vector + graph search
    2. expand_candidates()        – Leiden community expansion
    3. score_candidates()         – tag EXTRACTED / INFERRED / AMBIGUOUS
    4. build_reasoning_path()     – walk hierarchical graph
    5. generate_answer()          – final answer (graph or LLM)

Usage:
    from kg_decision_pipeline.pipeline import run_pipeline
    result = run_pipeline("brake pedal feels spongy")
"""

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.pipeline.hybrid_retrieval import hybrid_retrieve

_expander = importlib.import_module("kg_decision_pipeline.01_community_expander")
_scorer = importlib.import_module("kg_decision_pipeline.02_confidence_scorer")
_path_builder = importlib.import_module("kg_decision_pipeline.03_reasoning_path")
_answer_gen = importlib.import_module("kg_decision_pipeline.04_answer_generator")


def run_pipeline(
    query: str,
    skip_clarification: bool = False,
    top_k: int = 10,
) -> dict:
    """Run the full KG decision pipeline end-to-end.

    Parameters
    ----------
    query : str
        Natural-language vehicle fault description.
    skip_clarification : bool
        If True, bypass the graph-scoring stage and go straight to
        LLM fallback for an AMBIGUOUS result.  Default False.
    top_k : int
        Number of candidates to retrieve in the hybrid search stage.

    Returns
    -------
    dict with keys:
        mode                str
        answer              str
        source              "graph" | "llm"
        clarifying_question str | None
    """
    step1 = hybrid_retrieve(query, top_k=top_k)
    step2 = _expander.expand_candidates(step1)
    step3 = _scorer.score_candidates(step2, skip_allowed=skip_clarification)
    step4 = _path_builder.build_reasoning_path(step3)
    step5 = _answer_gen.generate_answer(step4, step3)
    step5["top_category"] = step4.get("top_category")
    step5["top_subcategory"] = step4.get("top_subcategory")
    step5["matched_symptoms"] = step4.get("matched_symptoms", [])
    step5["diagnosis_steps"] = step4.get("diagnosis_steps", [])
    step5["reasoning_chain"] = step4.get("reasoning_chain", [])
    return step5


# ---------------------------------------------------------------------------
# __main__ smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_queries = [
        ("brake pedal feels spongy when pressed", "EXTRACTED"),
        ("engine makes a knocking sound at high RPM", "INFERRED"),
        ("my car is broken help", "AMBIGUOUS"),
    ]

    for q, expected_mode in test_queries:
        print(f"\n{'=' * 70}")
        print(f"Query: {q!r}  |  Expected: {expected_mode}")
        print(f"{'=' * 70}")
        result = run_pipeline(q)
        print(f"  Mode:     {result.get('mode', '?')}")
        print(f"  Source:   {result.get('source', '?')}")
        cq = result.get("clarifying_question")
        if cq:
            print(f"  Clarify:  {cq}")
        answer = result.get("answer", "")
        if len(answer) > 300:
            answer = answer[:300] + f"\n  ... ({len(answer)} chars total)"
        print(f"  Answer:\n{answer}")
