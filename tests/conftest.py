"""Shared pytest fixtures for the decision engine test suite.

All fixtures produce deterministic data matching the real pipeline
output structures so tests are reproducible.
"""

import pytest


# ---------------------------------------------------------------------------
# Preprocessed query (output of preprocess_query)
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_preprocessed():
    return {
        "original": "brake pedal feels spongy",
        "processed": "brake pedal feels spongy",
        "intent": "symptom_report",
        "error_codes": [],
        "entities": [
            {
                "node_id": "sym_001",
                "label": "Spongy Brake Pedal",
                "node_type": "Symptom",
                "category": "Brake System",
                "subcategory": "Brake Pedal",
                "community": 3,
                "match_ratio": 0.85,
                "weighted_score": 0.9,
                "confidence": 0.9,
            },
            {
                "node_id": "sym_002",
                "label": "Brake Booster",
                "node_type": "Category",
                "category": "Brake System",
                "subcategory": "Brake Booster",
                "community": 3,
                "match_ratio": 0.6,
                "weighted_score": 0.6,
                "confidence": 0.6,
            },
        ],
        "expansion_terms": ["brake booster", "brake caliper", "brake fluid"],
        "expanded_queries": [
            "brake pedal feels spongy",
            "spongy brake pedal",
            "brake booster failure",
            "brake caliper issue",
        ],
        "retrieval_hints": {
            "communities": [3],
            "categories": ["Brake System"],
            "node_types": ["Symptom", "Category"],
        },
        "expected_sensors": ["prs_brk", "prs_cmpr_dn"],
    }


# ---------------------------------------------------------------------------
# Retrieval result (output of hybrid_retrieve)
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_retrieval_result():
    return {
        "query": "brake pedal feels spongy",
        "processed_query": "brake pedal feels spongy",
        "expected_sensors": ["prs_brk", "prs_cmpr_dn"],
        "retrieval_hints": {
            "communities": [3],
            "categories": ["Brake System"],
            "node_types": ["Symptom", "Category"],
        },
        "entities": [],
        "candidates": [
            {
                "node_id": "sym_001",
                "node_type": "Symptom",
                "label": "Spongy Brake Pedal",
                "category": "Brake System",
                "subcategory": "Brake Pedal",
                "community_id": "3",
                "is_multi_category": "False",
                "source": "vector+graph",
                "found_by": ["vector", "graph"],
                "score": 0.52,
                "raw_score": 0.52,
                "norm_score": 0.85,
            },
            {
                "node_id": "sym_003",
                "node_type": "Symptom",
                "label": "Brake Fluid Leak",
                "category": "Brake System",
                "subcategory": "Brake Fluid",
                "community_id": "3",
                "is_multi_category": "False",
                "source": "vector",
                "found_by": ["vector"],
                "score": 0.38,
                "raw_score": 0.38,
                "norm_score": 0.62,
            },
            {
                "node_id": "cat_002",
                "node_type": "Category",
                "label": "Brake Booster",
                "category": "Brake System",
                "subcategory": "Brake Booster",
                "community_id": "3",
                "is_multi_category": "False",
                "source": "graph",
                "found_by": ["graph"],
                "score": 0.25,
                "raw_score": 0.25,
                "norm_score": 0.41,
            },
        ],
        "source_breakdown": {"vector+graph": 1, "vector": 1, "graph": 1},
        "retrieval_stats": {
            "vector_candidates": 5,
            "graph_candidates": 4,
            "community_candidates": 3,
            "total_before_merge": 8,
        },
    }


# ---------------------------------------------------------------------------
# Mapped faults (output of map_faults)
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_mapped_faults():
    return [
        {
            "kg_category": "Brake System",
            "label": "Spongy Brake Pedal",
            "navic_fault": "FAULT_INJ_PRS",
            "kg_score": 0.52,
            "source": "vector+graph",
            "node_type": "Symptom",
            "mapping_type": "label",
            "mapping_confidence": 1.0,
        },
        {
            "kg_category": "Brake System",
            "label": "Brake Fluid Leak",
            "navic_fault": "FAULT_INJ_PRS",
            "kg_score": 0.38,
            "source": "vector",
            "node_type": "Symptom",
            "mapping_type": "label",
            "mapping_confidence": 1.0,
        },
        {
            "kg_category": "Brake System",
            "label": "Brake Booster",
            "navic_fault": "FAULT_INJ_PRS",
            "kg_score": 0.25,
            "source": "graph",
            "node_type": "Category",
            "mapping_type": "category",
            "mapping_confidence": 0.85,
        },
    ]


# ---------------------------------------------------------------------------
# Sensor results (output of analyze_fault_candidates)
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_sensor_results():
    return {
        "FAULT_INJ_PRS": {
            "fault": "FAULT_INJ_PRS",
            "speed": 1000,
            "num_sensors_checked": 4,
            "critical": ["prs_brk"],
            "warning": ["prs_cmpr_dn"],
            "normal": ["prs_cmpr_up", "prs_rsv"],
            "sensor_confidence": 0.82,
            "sensor_results": [
                {
                    "sensor": "prs_brk",
                    "current_value": 220.0,
                    "nominal_mean": 180.0,
                    "fault_mean": 230.0,
                    "z_score": 5.2,
                    "percent_change": 22.2,
                    "status": "CRITICAL",
                    "sensor_confidence": 0.92,
                },
                {
                    "sensor": "prs_cmpr_dn",
                    "current_value": 15.0,
                    "nominal_mean": 12.0,
                    "fault_mean": 16.5,
                    "z_score": 3.1,
                    "percent_change": 25.0,
                    "status": "WARNING",
                    "sensor_confidence": 0.75,
                },
            ],
            "kg_category": "Brake System",
            "kg_score": 0.52,
            "kg_label": "Spongy Brake Pedal",
            "node_type": "Symptom",
            "source": "vector+graph",
        },
    }


# ---------------------------------------------------------------------------
# Fused result (output of fuse_evidence)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Matched symptoms sources
# ---------------------------------------------------------------------------

@pytest.fixture
def original_symptoms():
    """Raw user-provided symptoms (from preprocessed['original'])."""
    return ["brake pedal feels spongy"]


@pytest.fixture
def entity_labels():
    """KG nodes directly detected from original input, before expansion.
    Use these (not expanded_queries) for coverage computation.
    """
    return ["Spongy Brake Pedal", "Brake Booster"]


@pytest.fixture
def expanded_queries():
    """Preprocessor-expanded queries (after graph neighbor/sibling expansion).
    NOT used for coverage — included only for reference.
    """
    return [
        "brake pedal feels spongy",
        "spongy brake pedal",
        "brake booster failure",
        "brake caliper issue",
    ]


@pytest.fixture
def mock_fused_candidates():
    return [
        {
            "node_id": "sym_001",
            "label": "Spongy Brake Pedal",
            "category": "Brake System",
            "source": "vector+graph",
            "score": 0.52,
            "kg_score": 0.26,
            "sensor_score": 0.82,
            "final_score": 0.41,
            "critical_sensors": ["prs_brk"],
            "warning_sensors": ["prs_cmpr_dn"],
            "normal_sensors": ["prs_cmpr_up", "prs_rsv"],
        },
        {
            "node_id": "sym_003",
            "label": "Brake Fluid Leak",
            "category": "Brake System",
            "source": "vector",
            "score": 0.38,
            "kg_score": 0.19,
            "sensor_score": 0.0,
            "final_score": 0.29,
            "critical_sensors": [],
            "warning_sensors": [],
            "normal_sensors": [],
        },
        {
            "node_id": "cat_002",
            "label": "Brake Booster",
            "category": "Brake System",
            "source": "graph",
            "score": 0.25,
            "kg_score": 0.125,
            "sensor_score": 0.0,
            "final_score": 0.21,
            "critical_sensors": [],
            "warning_sensors": [],
            "normal_sensors": [],
        },
    ]


@pytest.fixture
def mock_fused_result(mock_fused_candidates):
    return {
        "query": "brake pedal feels spongy",
        "fused_candidates": mock_fused_candidates,
    }
