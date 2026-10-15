"""
pipeline.py

Single entry point that chains all stages of the KG decision pipeline
for the vehicle-fault-kg retrieval layer:

    1. hybrid_retrieve()          – vector + graph + community search
    2. calibrate_scores()         – score calibration (clamp, saturate, noisy-OR)
    3. score_candidates()         – path-aware confidence, tag EXTRACTED / INFERRED / AMBIGUOUS
    4. build_reasoning_path()     – walk hierarchical graph for context
    5. generate_answer()          – final answer (graph or LLM)

Architecture (compared to original kg_decision_pipeline):
  - No community_expander step (community search is now inside hybrid_retrieve)
  - Calibrator generalized for 7-source retrieval output
  - Scorer redesigned with path-aware effective confidence
  - Reasoning path and answer generator reused with minimal changes

Usage:
    from kg_decision.pipeline import run_pipeline
    result = run_pipeline("brake pedal feels spongy")
"""

import importlib
import os
import sys
from pathlib import Path

# Ensure working directory is vehicle-fault-kg/ so relative paths in
# the retrieval layer (data/chroma_db, graphify-out/, etc.) resolve correctly.
_HERE = Path(__file__).resolve().parent
_VFK_ROOT = _HERE.parent
os.chdir(str(_VFK_ROOT))

sys.path.insert(0, str(_VFK_ROOT))

from scripts.pipeline.hybrid_retrieval import hybrid_retrieve

_calibrator = importlib.import_module("kg_decision.00_score_calibrator")
_scorer = importlib.import_module("kg_decision.02_confidence_scorer")
_path_builder = importlib.import_module("kg_decision.03_reasoning_path")
_answer_gen = importlib.import_module("kg_decision.04_answer_generator")

calibrate_scores = _calibrator.calibrate_scores
score_candidates = _scorer.score_candidates
build_reasoning_path = _path_builder.build_reasoning_path
generate_answer = _answer_gen.generate_answer


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
    step2 = calibrate_scores(step1)
    step3 = score_candidates(step2, skip_allowed=skip_clarification)
    step3["query"] = query               # forward query for reasoner
    step4 = build_reasoning_path(step3)
    step5 = generate_answer(step4, step3)
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
        ("Soft brake pedal", "EXTRACTED"),
        ("brake pedal feels spongy when pressed", "INFERRED"),
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
