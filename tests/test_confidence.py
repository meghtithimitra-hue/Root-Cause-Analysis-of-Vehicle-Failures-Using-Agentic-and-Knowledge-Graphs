"""Unit tests for decision_engine.confidence."""

import sys
import os

# Ensure the scripts directory is on the path so decision_engine is importable.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "num_pipeline", "scripts"))

from decision_engine.confidence import (
    CALIBRATION_REFERENCE,
    W_COVERAGE,
    W_RETRIEVAL,
    W_SEPARATION,
    SENSOR_BOOST_MAX,
    SENSOR_BOOST_THRESHOLD,
    calibrate_retrieval,
    compute_separation,
    compute_coverage,
    compute_sensor_boost,
    compute_confidence,
)


# ===================================================================
# calibrate_retrieval
# ===================================================================

class TestCalibrateRetrieval:
    def test_zero(self):
        assert calibrate_retrieval(0.0) == 0.0

    def test_negative(self):
        assert calibrate_retrieval(-0.1) == 0.0

    def test_midrange(self):
        assert calibrate_retrieval(0.275) == 0.5

    def test_at_reference(self):
        assert calibrate_retrieval(CALIBRATION_REFERENCE) == 1.0

    def test_above_reference_caps_at_one(self):
        assert calibrate_retrieval(0.80) == 1.0

    def test_large_value(self):
        assert calibrate_retrieval(10.0) == 1.0

    def test_small_positive(self):
        result = calibrate_retrieval(0.055)
        assert abs(result - 0.1) < 1e-9


# ===================================================================
# compute_separation
# ===================================================================

class TestComputeSeparation:
    def test_single_candidate(self):
        assert compute_separation([0.5]) == 1.0

    def test_empty_list(self):
        assert compute_separation([]) == 0.0

    def test_clear_leader(self):
        # gap = 0.5 - 0.3 = 0.2, relative = 0.2 / 0.5 = 0.4
        assert abs(compute_separation([0.5, 0.3]) - 0.4) < 1e-9

    def test_tied(self):
        result = compute_separation([0.45, 0.44])
        expected = (0.45 - 0.44) / 0.45
        assert abs(result - expected) < 1e-9

    def test_equal_scores(self):
        assert compute_separation([0.4, 0.4, 0.4]) == 0.0

    def test_top_is_zero(self):
        assert compute_separation([0.0, 0.0]) == 0.0

    def test_three_candidates(self):
        # gap = 0.5 - 0.4 = 0.1, relative = 0.1 / 0.5 = 0.2
        assert abs(compute_separation([0.5, 0.4, 0.1]) - 0.2) < 1e-9

    def test_second_below_zero(self):
        # Negative second score shouldn't happen in practice, but
        # gap would be large; verify it caps at 1.0.
        result = compute_separation([0.3, -0.1])
        assert result == 1.0


# ===================================================================
# compute_coverage
# ===================================================================

class TestComputeCoverage:
    def test_full_coverage(self):
        """All original symptoms matched by entity detection."""
        assert compute_coverage(
            ["a", "b", "c"], ["a", "b", "c"]
        ) == 1.0

    def test_partial_coverage(self):
        """Only one of three original symptoms matched."""
        result = compute_coverage(["a", "b", "c"], ["a"])
        assert abs(result - 1 / 3) < 1e-9

    def test_no_coverage(self):
        """No original symptoms matched."""
        assert compute_coverage(["a", "b"], ["x", "y"]) == 0.0

    def test_empty_original(self):
        """No original symptoms provided — returns 0."""
        assert compute_coverage([], ["a", "b"]) == 0.0

    def test_empty_matched(self):
        """Original symptoms exist but none were matched."""
        assert compute_coverage(["a", "b"], []) == 0.0

    def test_both_empty(self):
        assert compute_coverage([], []) == 0.0

    def test_case_insensitive(self):
        assert compute_coverage(["Brake"], ["brake"]) == 1.0

    def test_whitespace_handling(self):
        assert compute_coverage(
            [" brake pedal "], ["brake pedal"]
        ) == 1.0

    def test_partial_overlap(self):
        result = compute_coverage(
            ["brake pedal", "engine knock", "oil leak"],
            ["brake pedal", "oil leak", "unrelated"],
        )
        assert abs(result - 2 / 3) < 1e-9

    def test_coverage_uses_entity_labels_not_expanded(self):
        """Coverage is computed against original entity labels,
        NOT the preprocessor's expanded queries."""
        original = ["brake pedal feels spongy"]
        # Entity labels detected from original input (before expansion)
        entities = ["Spongy Brake Pedal"]
        # Expanded queries include additional graph-expanded terms
        expanded = [
            "brake pedal feels spongy",
            "spongy brake pedal",
            "brake booster failure",
            "brake caliper issue",
        ]
        # With entity labels: 0 matches (label differs from raw symptom)
        # This is expected — entity matching is label-based, not substring
        cov_entities = compute_coverage(original, entities)
        # With expanded queries: 1 match (raw query appears in expanded)
        cov_expanded = compute_coverage(original, expanded)
        # The values differ, proving the source matters
        assert cov_entities != cov_expanded


# ===================================================================
# compute_sensor_boost
# ===================================================================

class TestComputeSensorBoost:
    def test_no_sensor_data(self):
        assert compute_sensor_boost({}, "FAULT_X") == 0.0

    def test_fault_not_in_results(self):
        results = {"FAULT_Y": {"sensor_confidence": 0.9}}
        assert compute_sensor_boost(results, "FAULT_X") == 0.0

    def test_below_threshold(self):
        results = {"FAULT_X": {"sensor_confidence": 0.5}}
        assert compute_sensor_boost(results, "FAULT_X") == 0.0

    def test_at_threshold(self):
        results = {"FAULT_X": {"sensor_confidence": SENSOR_BOOST_THRESHOLD}}
        expected = SENSOR_BOOST_THRESHOLD * SENSOR_BOOST_MAX
        assert abs(compute_sensor_boost(results, "FAULT_X") - expected) < 1e-9

    def test_above_threshold(self):
        results = {"FAULT_X": {"sensor_confidence": 0.9}}
        expected = 0.9 * SENSOR_BOOST_MAX
        assert abs(compute_sensor_boost(results, "FAULT_X") - expected) < 1e-9

    def test_max_cap(self):
        results = {"FAULT_X": {"sensor_confidence": 1.0}}
        assert compute_sensor_boost(results, "FAULT_X") == SENSOR_BOOST_MAX


# ===================================================================
# compute_confidence (integration of all components)
# ===================================================================

class TestComputeConfidence:
    def test_maximum_possible(self):
        """All components at maximum → confidence capped at 1.0."""
        result = compute_confidence(
            retrieval_scores=[0.55],
            original_symptoms=["brake"],
            matched_symptoms=["brake"],
            sensor_results={"FAULT_X": {"sensor_confidence": 1.0}},
            top_fault="FAULT_X",
        )
        assert result == 1.0

    def test_minimum_possible(self):
        """All components at zero → confidence = 0.0."""
        result = compute_confidence(
            retrieval_scores=[0.0],
            original_symptoms=["brake"],
            matched_symptoms=[],
            sensor_results={},
            top_fault="",
        )
        assert result == 0.0

    def test_typical_inferred_range(self):
        """Moderate inputs should land in the INFERRED band [0.40, 0.75)."""
        result = compute_confidence(
            retrieval_scores=[0.40, 0.35],
            original_symptoms=["brake pedal", "spongy"],
            matched_symptoms=["brake pedal", "spongy", "extra"],
            sensor_results={},
            top_fault="",
        )
        assert 0.40 <= result < 0.75

    def test_typical_extracted_range(self):
        """Strong inputs should reach EXTRACTED (>= 0.75)."""
        result = compute_confidence(
            retrieval_scores=[0.55],
            original_symptoms=["brake"],
            matched_symptoms=["brake"],
            sensor_results={"F1": {"sensor_confidence": 0.95}},
            top_fault="F1",
        )
        assert result >= 0.75

    def test_typical_ambiguous_range(self):
        """Weak inputs should fall into AMBIGUOUS (< 0.40)."""
        result = compute_confidence(
            retrieval_scores=[0.10, 0.09],
            original_symptoms=["a", "b", "c"],
            matched_symptoms=[],
            sensor_results={},
            top_fault="",
        )
        assert result < 0.40

    def test_deterministic(self):
        """Same inputs always produce the same output."""
        args = (
            [0.45, 0.30],
            ["brake", "pedal"],
            ["brake"],
            {"F1": {"sensor_confidence": 0.8}},
            "F1",
        )
        assert compute_confidence(*args) == compute_confidence(*args)

    def test_weights_sum_to_one(self):
        """Verify weight constants are consistent."""
        assert abs(W_RETRIEVAL + W_SEPARATION + W_COVERAGE - 1.0) < 1e-9

    def test_separation_uses_retrieval_scores(self):
        """Separation should be computed from retrieval scores, not fused."""
        # Two candidates with clear retrieval gap
        result = compute_confidence(
            retrieval_scores=[0.50, 0.20],
            original_symptoms=["x"],
            matched_symptoms=["x"],
            sensor_results={},
            top_fault="",
        )
        # Separation = (0.50 - 0.20) / 0.50 = 0.6
        # cal = 0.50 / 0.55 = ~0.909
        # cov = 1.0
        # confidence = 0.60*0.909 + 0.20*0.6 + 0.20*1.0 = ~0.865
        assert result >= 0.75  # Should be EXTRACTED range

    def test_capped_at_one(self):
        """Confidence never exceeds 1.0 even with sensor boost."""
        result = compute_confidence(
            retrieval_scores=[0.55],
            original_symptoms=["x"],
            matched_symptoms=["x"],
            sensor_results={"F1": {"sensor_confidence": 1.0}},
            top_fault="F1",
        )
        assert result <= 1.0
