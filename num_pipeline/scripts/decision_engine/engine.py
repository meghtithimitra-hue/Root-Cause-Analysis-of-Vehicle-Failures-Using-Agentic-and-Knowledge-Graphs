"""Diagnostic engine orchestrator.

Thin coordinator that sequences the decision engine modules
and assembles the final ``DiagnosticReport``.  All computation
lives in the specialized modules — this file contains only
data extraction, sequencing, and report assembly.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .confidence import (
    W_COVERAGE,
    W_RETRIEVAL,
    W_SEPARATION,
    compute_confidence,
    compute_coverage,
    compute_separation,
    compute_sensor_boost,
    calibrate_retrieval,
)
from .mode_classifier import classify_mode, select_display_candidates
from .reasoning_chain import build_reasoning_chain
from .explanation import (
    generate_brief_summary,
    generate_diagnosis_summary,
    generate_inspection_steps,
)


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class DiagnosticReport:
    """Single output object returned by the decision engine.

    Consumed by the pipeline wrapper (``run_diagnostic.py``) and
    the Streamlit UI (``app.py``).
    """

    # --- Mode ---
    mode: str = "AMBIGUOUS"

    # --- Candidates ---
    top_candidate: Dict[str, Any] = field(default_factory=dict)
    display_candidates: List[Dict[str, Any]] = field(default_factory=list)

    # --- Confidence ---
    confidence: float = 0.0
    confidence_components: Dict[str, float] = field(default_factory=dict)

    # --- Reasoning ---
    reasoning_chain: List[Dict[str, Any]] = field(default_factory=list)

    # --- Summary ---
    summary: str = ""
    diagnosis_summary: str = ""
    inspection_steps: List[str] = field(default_factory=list)

    # --- Sensor evidence (per-fault badges) ---
    sensor_evidence: Dict[str, Any] = field(default_factory=dict)

    # --- Sensor debug metadata (temporary) ---
    sensor_debug: Dict[str, Any] = field(default_factory=dict)

    # --- Raw sensor results (for explanation enrichment) ---
    sensor_results_raw: Dict[str, Any] = field(default_factory=dict)

    # --- Provenance ---
    original_symptoms: List[str] = field(default_factory=list)
    query_text: str = ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_diagnostic_engine(
    preprocessed: Dict[str, Any],
    retrieval_result: Dict[str, Any],
    mapped_faults: List[Dict],
    sensor_results: Dict[str, Any],
    fused_result: Dict[str, Any],
    llm_provider=None,
) -> DiagnosticReport:
    """Run the decision engine on existing pipeline outputs.

    This function is a thin orchestrator.  All computation happens
    in the imported specialist modules.

    Parameters
    ----------
    preprocessed : dict
        Output of ``preprocess_query()``.
    retrieval_result : dict
        Output of ``hybrid_retrieve()``.
    mapped_faults : list[dict]
        Output of ``map_faults()``.
    sensor_results : dict
        Output of ``analyze_fault_candidates()`` (or ``{}``).
    fused_result : dict
        Output of ``fuse_evidence()`` — must contain
        ``"fused_candidates"`` list.
    llm_provider : optional
        ``LLMProvider`` instance for LLM-enhanced explanations.

    Returns
    -------
    DiagnosticReport
    """
    all_fused = fused_result.get("fused_candidates", [])
    original_symptoms = _extract_original_symptoms(preprocessed)
    matched_symptoms = _extract_matched_symptoms(preprocessed)
    query_text = retrieval_result.get("query", preprocessed.get("original", ""))

    # --- 0. Filter out empty-label candidates ---
    # Candidates with blank labels are preprocessing artefacts and
    # must not participate in confidence computation or display.
    fused_candidates = [
        c for c in all_fused
        if c.get("label", "").strip()
    ]

    # --- 1. Confidence components (individual functions) ---
    retrieval_scores = _get_retrieval_scores(retrieval_result)
    top_fault = _get_top_fault(fused_candidates, mapped_faults)

    raw_top = retrieval_scores[0] if retrieval_scores else 0.0
    cal = calibrate_retrieval(raw_top)
    sep = compute_separation(retrieval_scores)
    cov = compute_coverage(original_symptoms, matched_symptoms)
    boost = compute_sensor_boost(sensor_results, top_fault)
    final_conf = compute_confidence(
        retrieval_scores, original_symptoms, matched_symptoms,
        sensor_results, top_fault,
    )

    confidence_components = {
        "raw_retrieval_score": raw_top,
        "calibrated_retrieval": cal,
        "separation": sep,
        "coverage": cov,
        "sensor_boost": boost,
        "final_confidence": final_conf,
    }

    # --- 2. Mode classification ---
    mode = classify_mode(final_conf)

    # --- 3. Annotate candidates with confidence for display selection ---
    annotated = _annotate_candidates(
        fused_candidates, mapped_faults, sensor_results,
        original_symptoms, matched_symptoms, retrieval_scores,
    )

    # --- 4. Display candidates within mode band ---
    display = select_display_candidates(annotated, mode)

    # --- 5. Reasoning chain ---
    chain = build_reasoning_chain(
        preprocessed, retrieval_result, mapped_faults,
        fused_candidates, confidence_components, mode,
    )

    # --- 6. Sensor evidence badges ---
    sensor_badges = _compute_sensor_badges(display, sensor_results)

    # --- 7. Assemble report ---
    top = display[0] if display else {}

    summary = generate_brief_summary(mode, top, final_conf)

    diagnosis_summary = ""
    inspection_steps_list = []
    if mode in ("EXTRACTED", "INFERRED") and top.get("label"):
        diagnosis_summary = generate_diagnosis_summary(
            mode, top, original_symptoms, sensor_badges, llm_provider,
        )
        inspection_steps_list = generate_inspection_steps(top)

    return DiagnosticReport(
        mode=mode,
        top_candidate=top,
        display_candidates=display,
        confidence=final_conf,
        confidence_components=confidence_components,
        reasoning_chain=chain,
        summary=summary,
        diagnosis_summary=diagnosis_summary,
        inspection_steps=inspection_steps_list,
        sensor_evidence=sensor_badges,
        sensor_results_raw=sensor_results,
        original_symptoms=original_symptoms,
        query_text=query_text,
    )


# ---------------------------------------------------------------------------
# Private helpers (data extraction only — no computation)
# ---------------------------------------------------------------------------

def _extract_original_symptoms(preprocessed):
    """Extract original user text as individual words for coverage.

    Returns individual words from the raw query text (before any
    preprocessing or entity detection) so that coverage measures how
    well the user's actual words appear in the matched KG entities.
    """
    original = preprocessed.get("original", "") or preprocessed.get("original_text", "")
    if not original:
        return []
    return [w for w in original.split() if w.strip()]


def _extract_matched_symptoms(preprocessed):
    """Extract individual words from matched entity labels for coverage.

    Returns individual words from all entity labels detected by the
    preprocessor. Coverage is the fraction of original user words that
    appear in this set.
    """
    entities = preprocessed.get("entities", [])
    words = set()
    for e in entities:
        label = e.get("label", "")
        for w in label.split():
            words.add(w)
    return list(words)


def _get_retrieval_scores(retrieval_result):
    """Get retrieval scores in descending order.

    Empty-label candidates are excluded — they are preprocessing
    artefacts that must not influence separation computation.
    """
    candidates = retrieval_result.get("candidates", [])
    scores = []
    for c in candidates:
        label = c.get("label", "").strip()
        if not label:
            continue
        scores.append(c.get("score", 0.0))
    scores.sort(reverse=True)
    return scores


def _get_top_fault(fused_candidates, mapped_faults):
    """Determine the NavicEngine fault of the top candidate."""
    if fused_candidates:
        top_label = fused_candidates[0].get("label", "")
        for mf in mapped_faults:
            if mf.get("label") == top_label:
                return mf.get("navic_fault", "")
    return ""


def _check_contradicted(navic_fault, sensor_results):
    """Check whether sensor data contradicts this fault.

    A fault is contradicted only when sensor data for a *different*
    fault contains critical/warning indicators while this fault has
    none.  Normal readings for other faults do not constitute
    contradiction.
    """
    for other_fault, si in sensor_results.items():
        if other_fault == navic_fault:
            continue
        if si.get("critical") or si.get("warning"):
            return "Contradicted"
    return "No Evidence"


def _annotate_candidates(
    fused_candidates, mapped_faults, sensor_results,
    original_symptoms, matched_symptoms, retrieval_scores,
):
    """Add confidence and metadata to each candidate for display.

    Each candidate gets:
    - ``confidence``: calibrated retrieval score (candidate-specific)
      plus shared separation/coverage base (query-level) plus
      optional sensor boost (candidate-specific).
    - ``navic_fault``: from mapped_faults
    - ``mapping_type``: from mapped_faults
    - ``sensor_status``: Supported / Contradicted / No Evidence
      (Contradicted only when another fault has sensor indicators)

    Separation and coverage are query-level metrics — computed once
    and applied identically to every candidate.
    """
    # Build label → mapped fault lookup
    fault_lookup = {mf["label"]: mf for mf in mapped_faults}

    # Query-level components (same for all candidates)
    sep = compute_separation(retrieval_scores)
    cov = compute_coverage(original_symptoms, matched_symptoms)

    annotated = []
    for c in fused_candidates:
        label = c.get("label", "")
        mf = fault_lookup.get(label, {})
        navic_fault = mf.get("navic_fault", "")
        mapping_type = mf.get("mapping_type", "")

        # Candidate-specific components
        cand_score = c.get("score", 0.0)
        cal = calibrate_retrieval(cand_score)
        boost = compute_sensor_boost(sensor_results, navic_fault)

        cand_conf = (
            W_RETRIEVAL * cal
            + W_SEPARATION * sep
            + W_COVERAGE * cov
            + boost
        )
        cand_conf = min(cand_conf, 1.0)

        # Sensor badge
        # "Supported"      — sensor has critical/warning indicators for this fault
        # "No Evidence"    — sensor data exists but shows only normal readings,
        #                    or no sensor data for this fault at all
        # "Contradicted"   — sensor data exists for a *different* fault with
        #                    critical/warning indicators, and this fault has none
        sensor_status = "No Evidence"
        if navic_fault and navic_fault in sensor_results:
            si = sensor_results[navic_fault]
            if si.get("critical") or si.get("warning"):
                sensor_status = "Supported"
            else:
                # This fault has sensor data but no indicators.
                # Check if another fault has critical/warning evidence.
                sensor_status = _check_contradicted(
                    navic_fault, sensor_results,
                )

        entry = dict(c)  # shallow copy — don't mutate original
        entry["confidence"] = cand_conf
        entry["navic_fault"] = navic_fault
        entry["mapping_type"] = mapping_type
        entry["sensor_status"] = sensor_status
        annotated.append(entry)

    return annotated


def _compute_sensor_badges(display_candidates, sensor_results):
    """Build sensor evidence dict keyed by fault ID for display candidates."""
    badges = {}
    for c in display_candidates:
        fault = c.get("navic_fault", "")
        if not fault or fault in badges:
            continue
        if fault in sensor_results:
            si = sensor_results[fault]
            badges[fault] = {
                "status": c.get("sensor_status", "No Evidence"),
                "critical": si.get("critical", []),
                "warning": si.get("warning", []),
                "normal": si.get("normal", []),
                "sensor_confidence": si.get("sensor_confidence", 0.0),
            }
        else:
            badges[fault] = {
                "status": "No Evidence",
                "critical": [],
                "warning": [],
                "normal": [],
                "sensor_confidence": 0.0,
            }
    return badges
