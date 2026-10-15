"""
Pipeline package — Vehicle Fault Diagnosis System.

Modules:
    query_preprocessor: Input normalization and expansion
    hybrid_retrieval: KG + BM25 + Symptom embedding retrieval
    fault_mapper: Candidate scoring and ranking
    evidence_fusion: KG + sensor evidence fusion
    reasoning_engine: Mode determination + reasoning chain
    llm_provider: LLM abstraction (Ollama)
    explanation_generator: Natural language explanation
"""

from .reasoning_engine import reason, DiagnosticDecision
from .explanation_generator import generate_explanation, generate_ai_assisted_analysis
from .llm_provider import get_llm_provider

__all__ = [
    "reason",
    "DiagnosticDecision",
    "generate_explanation",
    "generate_ai_assisted_analysis",
    "get_llm_provider"
]
