"""Unit tests for decision_engine.reasoning_chain."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "num_pipeline", "scripts"))

from decision_engine.reasoning_chain import build_reasoning_chain


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _preprocessed(original="brake pedal feels spongy", entities=None, expansion_terms=None):
    default_entities = [
        {"label": "Spongy Brake Pedal", "node_type": "Symptom"},
        {"label": "Brake Booster", "node_type": "Category"},
    ]
    default_expansion = ["brake booster", "brake caliper"]
    return {
        "original": original,
        "processed": original,
        "intent": "symptom_report",
        "error_codes": [],
        "entities": default_entities if entities is None else entities,
        "expansion_terms": default_expansion if expansion_terms is None else expansion_terms,
        "expanded_queries": [],
        "retrieval_hints": {},
        "expected_sensors": [],
    }


def _retrieval_result(candidates=None):
    if candidates is None:
        candidates = [
            {
                "label": "Spongy Brake Pedal",
                "score": 0.52,
                "source": "vector+graph",
                "found_by": ["vector", "graph"],
            },
            {
                "label": "Brake Fluid Leak",
                "score": 0.38,
                "source": "vector",
                "found_by": ["vector"],
            },
        ]
    return {
        "query": "brake pedal feels spongy",
        "candidates": candidates,
        "source_breakdown": {"vector+graph": 1, "vector": 1},
        "retrieval_stats": {
            "vector_candidates": 5,
            "graph_candidates": 4,
            "community_candidates": 3,
            "total_before_merge": 8,
        },
    }


def _mapped_faults():
    return [
        {"label": "Spongy Brake Pedal", "navic_fault": "FAULT_INJ_PRS",
         "mapping_type": "label"},
        {"label": "Brake Fluid Leak", "navic_fault": "FAULT_INJ_PRS",
         "mapping_type": "category"},
    ]


def _components(raw=0.52, cal=0.945, sep=0.269, cov=1.0, boost=0.0, final=0.828):
    return {
        "raw_retrieval_score": raw,
        "calibrated_retrieval": cal,
        "separation": sep,
        "coverage": cov,
        "sensor_boost": boost,
        "final_confidence": final,
    }


# ===================================================================
# Structure tests
# ===================================================================

class TestChainStructure:
    def test_returns_list(self):
        result = build_reasoning_chain(
            _preprocessed(), _retrieval_result(), _mapped_faults(),
            [], _components(), "INFERRED",
        )
        assert isinstance(result, list)

    def test_steps_are_dicts_with_required_keys(self):
        result = build_reasoning_chain(
            _preprocessed(), _retrieval_result(), _mapped_faults(),
            [], _components(), "INFERRED",
        )
        for step in result:
            assert "step" in step
            assert "detail" in step
            assert "metrics" in step

    def test_metrics_are_dicts(self):
        result = build_reasoning_chain(
            _preprocessed(), _retrieval_result(), _mapped_faults(),
            [], _components(), "INFERRED",
        )
        for step in result:
            assert isinstance(step["metrics"], dict)

    def test_steps_have_string_values(self):
        result = build_reasoning_chain(
            _preprocessed(), _retrieval_result(), _mapped_faults(),
            [], _components(), "INFERRED",
        )
        for step in result:
            assert isinstance(step["step"], str)
            assert isinstance(step["detail"], str)
            assert len(step["step"]) > 0
            assert len(step["detail"]) > 0


# ===================================================================
# Core step tests
# ===================================================================

class TestCoreSteps:
    def test_always_has_five_core_steps(self):
        result = build_reasoning_chain(
            _preprocessed(), _retrieval_result(), _mapped_faults(),
            [], _components(), "EXTRACTED",
        )
        step_names = [s["step"] for s in result]
        for name in ["Query Analysis", "Knowledge Graph Retrieval",
                      "Fault Mapping", "Confidence Calculation",
                      "Mode Determination"]:
            assert name in step_names

    def test_extracted_has_no_symptom_gap(self):
        result = build_reasoning_chain(
            _preprocessed(), _retrieval_result(), _mapped_faults(),
            [], _components(), "EXTRACTED",
        )
        step_names = [s["step"] for s in result]
        assert "Symptom Gap Analysis" not in step_names

    def test_inferred_has_symptom_gap(self):
        result = build_reasoning_chain(
            _preprocessed(), _retrieval_result(), _mapped_faults(),
            [], _components(), "INFERRED",
        )
        step_names = [s["step"] for s in result]
        assert "Symptom Gap Analysis" in step_names

    def test_ambiguous_has_symptom_gap(self):
        result = build_reasoning_chain(
            _preprocessed(), _retrieval_result(), _mapped_faults(),
            [], _components(), "AMBIGUOUS",
        )
        step_names = [s["step"] for s in result]
        assert "Symptom Gap Analysis" in step_names

    def test_chain_length_inferred(self):
        result = build_reasoning_chain(
            _preprocessed(), _retrieval_result(), _mapped_faults(),
            [], _components(), "INFERRED",
        )
        assert len(result) == 6

    def test_chain_length_extracted(self):
        result = build_reasoning_chain(
            _preprocessed(), _retrieval_result(), _mapped_faults(),
            [], _components(), "EXTRACTED",
        )
        assert len(result) == 5


# ===================================================================
# Metrics structure tests
# ===================================================================

class TestMetricsStructure:
    def test_query_analysis_metrics(self):
        result = build_reasoning_chain(
            _preprocessed(original="engine knock"),
            _retrieval_result(), _mapped_faults(),
            [], _components(), "INFERRED",
        )
        qa = next(s for s in result if s["step"] == "Query Analysis")
        m = qa["metrics"]
        assert m["original_query"] == "engine knock"
        assert m["intent"] == "symptom_report"
        assert m["entity_count"] == 2
        assert isinstance(m["entity_labels"], list)

    def test_kg_retrieval_metrics_has_provenance(self):
        result = build_reasoning_chain(
            _preprocessed(), _retrieval_result(), _mapped_faults(),
            [], _components(), "INFERRED",
        )
        kr = next(s for s in result if s["step"] == "Knowledge Graph Retrieval")
        m = kr["metrics"]
        assert m["candidate_count"] == 2
        assert m["top_candidate"] == "Spongy Brake Pedal"
        assert abs(m["top_score"] - 0.52) < 1e-9
        assert isinstance(m["provenance"], list)
        assert len(m["provenance"]) == 2

    def test_kg_retrieval_provenance_fields(self):
        result = build_reasoning_chain(
            _preprocessed(), _retrieval_result(), _mapped_faults(),
            [], _components(), "INFERRED",
        )
        kr = next(s for s in result if s["step"] == "Knowledge Graph Retrieval")
        prov = kr["metrics"]["provenance"]
        for entry in prov:
            assert "label" in entry
            assert "score" in entry
            assert "source" in entry
            assert "found_by" in entry

    def test_kg_retrieval_source_breakdown(self):
        result = build_reasoning_chain(
            _preprocessed(), _retrieval_result(), _mapped_faults(),
            [], _components(), "INFERRED",
        )
        kr = next(s for s in result if s["step"] == "Knowledge Graph Retrieval")
        m = kr["metrics"]
        assert m["source_breakdown"] == {"vector+graph": 1, "vector": 1}
        assert m["retrieval_stats"]["total_before_merge"] == 8

    def test_fault_mapping_metrics(self):
        result = build_reasoning_chain(
            _preprocessed(), _retrieval_result(), _mapped_faults(),
            [], _components(), "INFERRED",
        )
        fm = next(s for s in result if s["step"] == "Fault Mapping")
        m = fm["metrics"]
        assert m["mapped_count"] == 2
        assert m["label_mapped"] == 1
        assert m["category_mapped"] == 1
        assert m["top_mapping"]["label"] == "Spongy Brake Pedal"
        assert m["top_mapping"]["mapping_type"] == "label"

    def test_confidence_metrics(self):
        comps = _components(raw=0.52, cal=0.945, sep=0.27, cov=1.0, boost=0.03, final=0.85)
        result = build_reasoning_chain(
            _preprocessed(), _retrieval_result(), _mapped_faults(),
            [], comps, "EXTRACTED",
        )
        cc = next(s for s in result if s["step"] == "Confidence Calculation")
        m = cc["metrics"]
        assert abs(m["raw_retrieval_score"] - 0.52) < 1e-9
        assert abs(m["calibrated_retrieval"] - 0.945) < 1e-9
        assert abs(m["separation"] - 0.27) < 1e-9
        assert abs(m["coverage"] - 1.0) < 1e-9
        assert abs(m["sensor_boost"] - 0.03) < 1e-9
        assert abs(m["final_confidence"] - 0.85) < 1e-9

    def test_mode_determination_metrics(self):
        result = build_reasoning_chain(
            _preprocessed(), _retrieval_result(), _mapped_faults(),
            [], _components(final=0.82), "EXTRACTED",
        )
        md = next(s for s in result if s["step"] == "Mode Determination")
        m = md["metrics"]
        assert m["mode"] == "EXTRACTED"
        assert abs(m["confidence"] - 0.82) < 1e-9
        assert "0.75" in m["threshold_applied"]

    def test_symptom_gap_metrics(self):
        result = build_reasoning_chain(
            _preprocessed(), _retrieval_result(), _mapped_faults(),
            [], _components(), "INFERRED",
        )
        sg = next(s for s in result if s["step"] == "Symptom Gap Analysis")
        m = sg["metrics"]
        assert m["entity_count"] == 2
        assert m["expansion_count"] == 2


# ===================================================================
# Content tests
# ===================================================================

class TestChainContent:
    def test_query_analysis_mentions_original(self):
        result = build_reasoning_chain(
            _preprocessed(original="engine knocking"),
            _retrieval_result(), _mapped_faults(),
            [], _components(), "INFERRED",
        )
        qa = next(s for s in result if s["step"] == "Query Analysis")
        assert "engine knocking" in qa["detail"]

    def test_kg_retrieval_mentions_top_candidate(self):
        result = build_reasoning_chain(
            _preprocessed(), _retrieval_result(), _mapped_faults(),
            [], _components(), "INFERRED",
        )
        kr = next(s for s in result if s["step"] == "Knowledge Graph Retrieval")
        assert "Spongy Brake Pedal" in kr["detail"]

    def test_confidence_mentions_components(self):
        comps = _components(cal=0.90, sep=0.50, cov=0.80, boost=0.03, final=0.77)
        result = build_reasoning_chain(
            _preprocessed(), _retrieval_result(), _mapped_faults(),
            [], comps, "EXTRACTED",
        )
        cc = next(s for s in result if s["step"] == "Confidence Calculation")
        assert "0.900" in cc["detail"]
        assert "0.500" in cc["detail"]
        assert "0.800" in cc["detail"]

    def test_mode_determination_mentions_threshold(self):
        result = build_reasoning_chain(
            _preprocessed(), _retrieval_result(), _mapped_faults(),
            [], _components(final=0.82), "EXTRACTED",
        )
        md = next(s for s in result if s["step"] == "Mode Determination")
        assert ">= 0.75" in md["detail"]
        assert "EXTRACTED" in md["detail"]

    def test_symptom_gap_ambiguous_no_entities(self):
        result = build_reasoning_chain(
            _preprocessed(entities=[], expansion_terms=[]),
            _retrieval_result(candidates=[]), [],
            [], _components(final=0.1), "AMBIGUOUS",
        )
        sg = next(s for s in result if s["step"] == "Symptom Gap Analysis")
        assert "No direct KG concepts" in sg["detail"]


# ===================================================================
# Edge cases
# ===================================================================

class TestEdgeCases:
    def test_empty_retrieval(self):
        result = build_reasoning_chain(
            _preprocessed(), _retrieval_result(candidates=[]),
            [], [], _components(), "AMBIGUOUS",
        )
        kr = next(s for s in result if s["step"] == "Knowledge Graph Retrieval")
        assert kr["metrics"]["candidate_count"] == 0
        assert kr["metrics"]["provenance"] == []

    def test_empty_mapped_faults(self):
        result = build_reasoning_chain(
            _preprocessed(), _retrieval_result(),
            [], [], _components(), "AMBIGUOUS",
        )
        fm = next(s for s in result if s["step"] == "Fault Mapping")
        assert fm["metrics"]["mapped_count"] == 0
