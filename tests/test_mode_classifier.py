"""Unit tests for decision_engine.mode_classifier."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "num_pipeline", "scripts"))

from decision_engine.mode_classifier import (
    THRESHOLD_EXTRACTED,
    MAX_DISPLAY,
    classify_mode,
    get_mode_description,
    select_display_candidates,
)


# ===================================================================
# classify_mode
# ===================================================================

class TestClassifyMode:
    def test_extracted_at_threshold(self):
        assert classify_mode(THRESHOLD_EXTRACTED) == "EXTRACTED"

    def test_extracted_above(self):
        assert classify_mode(0.90) == "EXTRACTED"

    def test_extracted_well_above(self):
        assert classify_mode(1.0) == "EXTRACTED"

    def test_ambiguous_just_below_extracted(self):
        assert classify_mode(0.299) == "AMBIGUOUS"

    def test_ambiguous_zero(self):
        assert classify_mode(0.0) == "AMBIGUOUS"

    def test_ambiguous_negative(self):
        assert classify_mode(-0.1) == "AMBIGUOUS"


# ===================================================================
# get_mode_description
# ===================================================================

class TestGetModeDescription:
    def test_extracted(self):
        desc = get_mode_description("EXTRACTED")
        assert "High-confidence" in desc

    def test_ambiguous(self):
        desc = get_mode_description("AMBIGUOUS")
        assert "more details" in desc

    def test_unknown(self):
        desc = get_mode_description("UNKNOWN")
        assert desc == "Unknown mode."

    def test_all_non_empty(self):
        for mode in ["EXTRACTED", "AMBIGUOUS"]:
            assert len(get_mode_description(mode)) > 0


# ===================================================================
# select_display_candidates
# ===================================================================

def _make_candidate(confidence, label="X"):
    return {"confidence": confidence, "label": label}


class TestSelectDisplayCandidates:
    def test_extracted_band(self):
        """EXTRACTED band: candidates >= 0.30."""
        candidates = [
            _make_candidate(0.90, "A"),
            _make_candidate(0.50, "B"),
            _make_candidate(0.20, "C"),  # below band
        ]
        result = select_display_candidates(candidates, "EXTRACTED")
        assert len(result) == 2
        assert [c["label"] for c in result] == ["A", "B"]

    def test_ambiguous_band(self):
        """AMBIGUOUS band: [0.0, 0.30)."""
        candidates = [
            _make_candidate(0.80, "A"),   # above band
            _make_candidate(0.20, "C"),   # in band
            _make_candidate(0.10, "D"),   # in band
        ]
        result = select_display_candidates(candidates, "AMBIGUOUS")
        assert len(result) == 2
        assert [c["label"] for c in result] == ["C", "D"]

    def test_max_cap(self):
        """Respects max_display limit."""
        candidates = [_make_candidate(0.80 - i * 0.01) for i in range(10)]
        result = select_display_candidates(candidates, "EXTRACTED", max_display=3)
        assert len(result) == 3

    def test_sorted_descending(self):
        """Output is sorted by confidence descending."""
        candidates = [
            _make_candidate(0.30, "C"),
            _make_candidate(0.55, "A"),
            _make_candidate(0.45, "B"),
        ]
        result = select_display_candidates(candidates, "EXTRACTED")
        assert [c["label"] for c in result] == ["A", "B", "C"]

    def test_empty_band(self):
        """No candidates in the band → empty list."""
        candidates = [
            _make_candidate(0.90, "A"),
            _make_candidate(0.80, "B"),
        ]
        result = select_display_candidates(candidates, "AMBIGUOUS")
        assert result == []

    def test_empty_input(self):
        """No candidates at all → empty list."""
        result = select_display_candidates([], "EXTRACTED")
        assert result == []

    def test_boundary_at_extracted_threshold(self):
        """Candidate exactly at 0.30 is EXTRACTED, not AMBIGUOUS."""
        candidates = [_make_candidate(0.30)]
        assert select_display_candidates(candidates, "EXTRACTED") == [candidates[0]]
        assert select_display_candidates(candidates, "AMBIGUOUS") == []

    def test_default_max_display(self):
        """Default max_display matches the constant."""
        assert MAX_DISPLAY == 5
        candidates = [_make_candidate(0.80 - i * 0.005) for i in range(8)]
        result = select_display_candidates(candidates, "EXTRACTED")
        assert len(result) == 5
