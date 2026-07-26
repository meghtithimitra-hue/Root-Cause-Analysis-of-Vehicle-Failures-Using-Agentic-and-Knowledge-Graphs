"""Reasoning chain builder.

Produces a step-by-step reasoning chain documenting how the diagnostic
conclusion was reached.  Each step is a dict with:

- ``"step"``   — title (str)
- ``"detail"`` — human-readable description (str)
- ``"metrics"`` — structured data (dict) for programmatic access

The chain is purely informational — it does not influence mode
determination or confidence.  It is consumed by the explanation
generator and displayed in the UI.
"""

from typing import Any, Dict, List


def build_reasoning_chain(
    preprocessed: Dict[str, Any],
    retrieval_result: Dict[str, Any],
    mapped_faults: List[Dict],
    fused_candidates: List[Dict],
    confidence_components: Dict[str, float],
    mode: str,
) -> List[Dict[str, Any]]:
    """Build the reasoning chain from pipeline outputs.

    Parameters
    ----------
    preprocessed : dict
        Output of ``preprocess_query()``.
    retrieval_result : dict
        Output of ``hybrid_retrieve()``.
    mapped_faults : list[dict]
        Output of ``map_faults()``.
    fused_candidates : list[dict]
        Fused candidates with ``"confidence"`` key.
    confidence_components : dict
        Intermediate values from the confidence calculation.
        Expected keys: ``calibrated_retrieval``, ``separation``,
        ``coverage``, ``sensor_boost``, ``raw_retrieval_score``,
        ``final_confidence``.
    mode : str
        Determined mode ("EXTRACTED", "INFERRED", "AMBIGUOUS").

    Returns
    -------
    list[dict]
        Ordered list of ``{"step": str, "detail": str, "metrics": dict}``
        dicts.
    """
    chain: List[Dict[str, Any]] = []

    chain.append(_step_query_analysis(preprocessed))
    chain.append(_step_kg_retrieval(retrieval_result))
    chain.append(_step_fault_mapping(mapped_faults))
    chain.append(_step_confidence(confidence_components))
    chain.append(_step_mode_determination(mode, confidence_components))

    if mode in ("INFERRED", "AMBIGUOUS"):
        chain.append(_step_symptom_gap(preprocessed))

    return chain


# ---------------------------------------------------------------------------
# Step builders
# ---------------------------------------------------------------------------

def _step_query_analysis(preprocessed: Dict[str, Any]) -> Dict[str, Any]:
    original = preprocessed.get("original", "")
    intent = preprocessed.get("intent", "unknown")
    entities = preprocessed.get("entities", [])
    entity_labels = [e.get("label", "") for e in entities[:5]]

    if entity_labels:
        detail = (
            f"User query: \"{original}\". Detected intent: {intent}. "
            f"KG entities identified: {', '.join(entity_labels)}."
        )
    else:
        detail = (
            f"User query: \"{original}\". Detected intent: {intent}. "
            f"No KG entities identified from the query."
        )

    return {
        "step": "Query Analysis",
        "detail": detail,
        "metrics": {
            "original_query": original,
            "intent": intent,
            "entity_count": len(entities),
            "entity_labels": entity_labels,
        },
    }


def _step_kg_retrieval(retrieval_result: Dict[str, Any]) -> Dict[str, Any]:
    candidates = retrieval_result.get("candidates", [])
    n = len(candidates)
    source_breakdown = retrieval_result.get("source_breakdown", {})
    stats = retrieval_result.get("retrieval_stats", {})

    if candidates:
        top = candidates[0]
        detail = (
            f"Retrieved {n} candidate(s). "
            f"Top: \"{top.get('label', '')}\" "
            f"(score: {top.get('score', 0.0):.3f}, "
            f"source: {top.get('source', 'unknown')})."
        )
    else:
        detail = f"Retrieved {n} candidate(s). No candidates found."

    # Build per-candidate provenance list
    provenance = []
    for c in candidates:
        provenance.append({
            "label": c.get("label", ""),
            "score": c.get("score", 0.0),
            "source": c.get("source", ""),
            "found_by": c.get("found_by", []),
        })

    return {
        "step": "Knowledge Graph Retrieval",
        "detail": detail,
        "metrics": {
            "candidate_count": n,
            "top_candidate": candidates[0].get("label", "") if candidates else None,
            "top_score": candidates[0].get("score", 0.0) if candidates else 0.0,
            "source_breakdown": source_breakdown,
            "retrieval_stats": stats,
            "provenance": provenance,
        },
    }


def _step_fault_mapping(mapped_faults: List[Dict]) -> Dict[str, Any]:
    if not mapped_faults:
        return {
            "step": "Fault Mapping",
            "detail": "No faults could be mapped from the retrieved candidates.",
            "metrics": {
                "mapped_count": 0,
                "label_mapped": 0,
                "category_mapped": 0,
            },
        }

    label_mapped = sum(1 for f in mapped_faults if f.get("mapping_type") == "label")
    category_mapped = sum(1 for f in mapped_faults if f.get("mapping_type") == "category")
    top = mapped_faults[0]

    detail = (
        f"Mapped {len(mapped_faults)} fault(s): "
        f"{label_mapped} direct label match(es), "
        f"{category_mapped} category fallback(s). "
        f"Top: \"{top.get('label', '')}\" → {top.get('navic_fault', '')} "
        f"({top.get('mapping_type', 'unknown')})."
    )

    return {
        "step": "Fault Mapping",
        "detail": detail,
        "metrics": {
            "mapped_count": len(mapped_faults),
            "label_mapped": label_mapped,
            "category_mapped": category_mapped,
            "top_mapping": {
                "label": top.get("label", ""),
                "navic_fault": top.get("navic_fault", ""),
                "mapping_type": top.get("mapping_type", ""),
            },
        },
    }


def _step_confidence(components: Dict[str, float]) -> Dict[str, Any]:
    cal = components.get("calibrated_retrieval", 0.0)
    sep = components.get("separation", 0.0)
    cov = components.get("coverage", 0.0)
    boost = components.get("sensor_boost", 0.0)
    raw = components.get("raw_retrieval_score", 0.0)
    final = components.get("final_confidence", 0.0)

    detail = (
        f"Retrieval calibration: {cal:.3f} (raw: {raw:.3f}). "
        f"Separation: {sep:.3f}. Coverage: {cov:.3f}. "
    )
    if boost > 0:
        detail += f"Sensor boost: +{boost:.3f}. "
    detail += f"Final confidence: {final:.3f}."

    return {
        "step": "Confidence Calculation",
        "detail": detail,
        "metrics": {
            "raw_retrieval_score": raw,
            "calibrated_retrieval": cal,
            "separation": sep,
            "coverage": cov,
            "sensor_boost": boost,
            "final_confidence": final,
        },
    }


def _step_mode_determination(
    mode: str, components: Dict[str, float]
) -> Dict[str, Any]:
    final = components.get("final_confidence", 0.0)

    thresholds = {
        "EXTRACTED": ">= 0.75",
        "INFERRED": ">= 0.40",
        "AMBIGUOUS": "< 0.40",
    }
    threshold_desc = thresholds.get(mode, "unknown")

    detail = f"Confidence {final:.3f} {threshold_desc} → mode: {mode}."

    return {
        "step": "Mode Determination",
        "detail": detail,
        "metrics": {
            "mode": mode,
            "confidence": final,
            "threshold_applied": threshold_desc,
        },
    }


def _step_symptom_gap(preprocessed: Dict[str, Any]) -> Dict[str, Any]:
    entities = preprocessed.get("entities", [])
    expansion_terms = preprocessed.get("expansion_terms", [])

    n_entities = len(entities)
    n_expansion = len(expansion_terms)

    if n_entities == 0:
        detail = (
            "No direct KG concepts matched the user's description. "
            "Providing additional symptoms may help find relevant fault patterns."
        )
    else:
        detail = (
            f"Identified {n_entities} KG concept(s), expanded to "
            f"{n_expansion} related terms. Additional symptoms may "
            f"help narrow the diagnosis."
        )

    return {
        "step": "Symptom Gap Analysis",
        "detail": detail,
        "metrics": {
            "entity_count": n_entities,
            "expansion_count": n_expansion,
        },
    }
