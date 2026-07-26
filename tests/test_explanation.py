"""Unit tests for decision_engine.explanation."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "num_pipeline", "scripts"))

from decision_engine.explanation import (
    generate_explanation,
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


def _display_candidates():
    return [
        _top_candidate("Spongy Brake Pedal", 0.82),
        _top_candidate("Brake Fluid Leak", 0.55, "FAULT_INJ_PRS", "vector"),
    ]


def _reasoning_chain():
    return [
        {"step": "Query Analysis", "detail": "User query: brake pedal.", "metrics": {}},
        {"step": "KG Retrieval", "detail": "Retrieved 2 candidates.", "metrics": {}},
        {"step": "Confidence", "detail": "Final confidence: 0.82.", "metrics": {}},
    ]


def _sensor_evidence():
    return {
        "FAULT_INJ_PRS": {
            "critical": ["prs_brk"],
            "warning": [],
            "normal": ["prs_cmpr_up"],
            "sensor_confidence": 0.82,
        },
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

    def test_inferred(self):
        result = generate_brief_summary("INFERRED", _top_candidate(), 0.55)
        assert "Best guess" in result
        assert "Spongy Brake Pedal" in result

    def test_ambiguous(self):
        result = generate_brief_summary("AMBIGUOUS", _top_candidate(), 0.25)
        assert "Insufficient" in result
        assert "Spongy Brake Pedal" in result

    def test_includes_fault_id(self):
        result = generate_brief_summary("EXTRACTED", _top_candidate(), 0.82)
        assert "FAULT_INJ_PRS" in result

    def test_non_empty_string(self):
        for mode in ["EXTRACTED", "INFERRED", "AMBIGUOUS"]:
            result = generate_brief_summary(mode, _top_candidate(), 0.5)
            assert isinstance(result, str)
            assert len(result) > 0


# ===================================================================
# generate_explanation (template path — no LLM)
# ===================================================================

class TestExplanationTemplate:
    def test_extracted_non_empty(self):
        result = generate_explanation(
            "EXTRACTED", _top_candidate(), _display_candidates(),
            _reasoning_chain(), _sensor_evidence(), llm_provider=None,
        )
        assert len(result) > 0

    def test_inferred_non_empty(self):
        result = generate_explanation(
            "INFERRED", _top_candidate(confidence=0.55), _display_candidates(),
            _reasoning_chain(), _sensor_evidence(), llm_provider=None,
        )
        assert len(result) > 0

    def test_ambiguous_non_empty(self):
        result = generate_explanation(
            "AMBIGUOUS", _top_candidate(confidence=0.25), [],
            _reasoning_chain(), {}, llm_provider=None,
        )
        assert len(result) > 0

    def test_extracted_mentions_diagnosis(self):
        result = generate_explanation(
            "EXTRACTED", _top_candidate(), _display_candidates(),
            _reasoning_chain(), _sensor_evidence(), llm_provider=None,
        )
        assert "Diagnosis" in result
        assert "Spongy Brake Pedal" in result

    def test_inferred_mentions_best_guess(self):
        result = generate_explanation(
            "INFERRED", _top_candidate(confidence=0.55), _display_candidates(),
            _reasoning_chain(), _sensor_evidence(), llm_provider=None,
        )
        assert "Best guess" in result

    def test_ambiguous_mentions_insufficient(self):
        result = generate_explanation(
            "AMBIGUOUS", _top_candidate(confidence=0.25), [],
            _reasoning_chain(), {}, llm_provider=None,
        )
        assert "Insufficient" in result

    def test_extracted_includes_sensor_status(self):
        result = generate_explanation(
            "EXTRACTED", _top_candidate(), _display_candidates(),
            _reasoning_chain(), _sensor_evidence(), llm_provider=None,
        )
        assert "Sensor" in result

    def test_extracted_no_sensor_data(self):
        result = generate_explanation(
            "EXTRACTED", _top_candidate(), _display_candidates(),
            _reasoning_chain(), {}, llm_provider=None,
        )
        assert "No numerical sensor data" in result

    def test_inferred_shows_alternatives(self):
        result = generate_explanation(
            "INFERRED", _top_candidate(confidence=0.55), _display_candidates(),
            _reasoning_chain(), _sensor_evidence(), llm_provider=None,
        )
        assert "Alternative" in result

    def test_reasoning_chain_included(self):
        result = generate_explanation(
            "EXTRACTED", _top_candidate(), _display_candidates(),
            _reasoning_chain(), _sensor_evidence(), llm_provider=None,
        )
        assert "Reasoning" in result

    def test_returns_string(self):
        result = generate_explanation(
            "EXTRACTED", _top_candidate(), _display_candidates(),
            _reasoning_chain(), _sensor_evidence(), llm_provider=None,
        )
        assert isinstance(result, str)


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
    def test_empty_reasoning_chain(self):
        result = generate_explanation(
            "EXTRACTED", _top_candidate(), _display_candidates(),
            [], _sensor_evidence(), llm_provider=None,
        )
        assert len(result) > 0

    def test_no_display_candidates(self):
        result = generate_explanation(
            "AMBIGUOUS", _top_candidate(confidence=0.2), [],
            _reasoning_chain(), {}, llm_provider=None,
        )
        assert len(result) > 0

    def test_brief_summary_zero_confidence(self):
        result = generate_brief_summary("AMBIGUOUS", _top_candidate(), 0.0)
        assert "Insufficient" in result
        assert "Spongy Brake Pedal" in result
