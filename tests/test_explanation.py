"""Unit tests for decision_engine.explanation."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "num_pipeline", "scripts"))

from decision_engine.explanation import (
    generate_brief_summary,
    generate_ai_assisted_analysis,
    _get_disclaimer,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _top_candidate(label="Spongy Brake Pedal", confidence=0.82,
                   navic_fault="FAULT_INJ_PRS", source="vector+graph"):
    return {
        "label": label,
        "confidence": confidence,
        "navic_fault": navic_fault,
        "source": source,
    }


# ===================================================================
# generate_brief_summary
# ===================================================================

class TestBriefSummary:
    def test_extracted(self):
        result = generate_brief_summary("EXTRACTED", _top_candidate(), 0.82)
        assert "Diagnosis" in result
        assert "Spongy Brake Pedal" in result
        assert "FAULT_INJ_PRS" in result

    def test_ambiguous(self):
        result = generate_brief_summary("AMBIGUOUS", _top_candidate(), 0.25)
        assert "Insufficient" in result
        assert "Spongy Brake Pedal" in result

    def test_includes_fault_id(self):
        result = generate_brief_summary("EXTRACTED", _top_candidate(), 0.82)
        assert "FAULT_INJ_PRS" in result

    def test_non_empty_string(self):
        for mode in ["EXTRACTED", "AMBIGUOUS"]:
            result = generate_brief_summary(mode, _top_candidate(), 0.5)
            assert isinstance(result, str)
            assert len(result) > 0


# ===================================================================
# generate_ai_assisted_analysis (template path — no LLM)
# ===================================================================

class TestAIAssistedAnalysis:
    def test_with_candidates(self):
        candidates = [
            {"label": "Spongy Brake Pedal", "score": 0.35, "source": "vector+graph"},
            {"label": "Brake Fluid Leak", "score": 0.25, "source": "vector"},
        ]
        result = generate_ai_assisted_analysis(
            "brake feels weird", candidates, llm_provider=None,
        )
        assert "brake feels weird" in result
        assert "Spongy Brake Pedal" in result
        assert "Brake Fluid Leak" in result
        assert _get_disclaimer() in result

    def test_without_candidates(self):
        result = generate_ai_assisted_analysis(
            "quantum fluctuation", [], llm_provider=None,
        )
        assert "quantum fluctuation" in result
        assert "No closely related" in result
        assert _get_disclaimer() in result

    def test_disclaimer_always_present(self):
        result = generate_ai_assisted_analysis("x", [], llm_provider=None)
        assert "NOT A DIAGNOSIS" in result
        assert "AI-ASSISTED" in result

    def test_returns_string(self):
        result = generate_ai_assisted_analysis("x", [], llm_provider=None)
        assert isinstance(result, str)
        assert len(result) > 0


# ===================================================================
# Edge cases
# ===================================================================

class TestEdgeCases:
    def test_brief_summary_zero_confidence(self):
        result = generate_brief_summary("AMBIGUOUS", _top_candidate(), 0.0)
        assert "Insufficient" in result
        assert "Spongy Brake Pedal" in result
