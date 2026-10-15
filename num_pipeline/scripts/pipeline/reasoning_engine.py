"""
Reasoning Engine — determines diagnostic mode and builds reasoning chain.

This module is SEPARATE from evidence_fusion.py. Evidence fusion provides
raw data; the reasoning engine interprets it and determines the diagnostic
mode based on evidence quality and completeness (not raw score thresholds).

Architectural principle: This module does reasoning. It does NOT call the LLM.
The LLM is only used by explanation_generator.py for presentation.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class DiagnosticDecision:
    """
    Output of the reasoning engine.
    """
    mode: str  # "AMBIGUOUS" | "INFERRED" | "EXTRACTED"
    predicted_system: Optional[str] = None
    predicted_subsystem: Optional[str] = None
    reasoning_chain: List[Dict[str, str]] = field(default_factory=list)
    confirmed_faults: List[Dict] = field(default_factory=list)
    contradicted_faults: List[Dict] = field(default_factory=list)
    remaining_faults: List[Dict] = field(default_factory=list)
    mode_reason: str = ""
    evidence_quality: str = ""  # "strong" | "moderate" | "weak"
    confidence: float = 0.0  # internal, not exposed in UI


def reason(evidence: Dict) -> DiagnosticDecision:
    """
    Determine diagnostic mode and build reasoning chain from fused evidence.

    Mode determination is based on OVERALL EVIDENCE QUALITY AND COMPLETENESS,
    not by directly thresholding numerical scores. Scores may be used internally
    as one signal among many, but the mode represents the system's state of
    knowledge about the fault.

    Args:
        evidence: Output from evidence_fusion.fuse()

    Returns:
        DiagnosticDecision with mode, reasoning chain, and categorized faults
    """

    candidates = evidence.get("candidate_faults", [])
    sensor_status = evidence.get("sensor_status", "NOT AVAILABLE")
    kg_status = evidence.get("kg_status", "NOT AVAILABLE")
    matched_symptoms = evidence.get("matched_symptoms", [])
    missing_symptoms = evidence.get("missing_symptoms", [])

    # ── Step 1: Assess evidence quality ───────────────────────────

    evidence_quality = _assess_evidence_quality(
        candidates, sensor_status, kg_status,
        matched_symptoms, missing_symptoms
    )

    # ── Step 2: Determine mode ────────────────────────────────────

    mode, mode_reason = _determine_mode(
        candidates, sensor_status, kg_status,
        matched_symptoms, missing_symptoms, evidence_quality
    )

    # ── Step 3: Categorize faults ─────────────────────────────────

    confirmed, contradicted, remaining = _categorize_faults(candidates)

    # ── Step 4: Build reasoning chain ─────────────────────────────

    reasoning_chain = _build_reasoning_chain(
        candidates, sensor_status, kg_status,
        matched_symptoms, missing_symptoms, evidence_quality, mode
    )

    # ── Step 5: Extract predicted system/subsystem ────────────────

    predicted_system = None
    predicted_subsystem = None
    if candidates:
        top = candidates[0]
        predicted_system = top.get("system", None)
        predicted_subsystem = top.get("subsystem", None)

    # ── Step 6: Compute internal confidence ───────────────────────

    confidence = _compute_confidence(
        candidates, evidence_quality, sensor_status, mode
    )

    return DiagnosticDecision(
        mode=mode,
        predicted_system=predicted_system,
        predicted_subsystem=predicted_subsystem,
        reasoning_chain=reasoning_chain,
        confirmed_faults=confirmed,
        contradicted_faults=contradicted,
        remaining_faults=remaining,
        mode_reason=mode_reason,
        evidence_quality=evidence_quality,
        confidence=confidence
    )


# ─── Evidence Quality Assessment ───────────────────────────────────

def _assess_evidence_quality(
    candidates: List[Dict],
    sensor_status: str,
    kg_status: str,
    matched_symptoms: List[str],
    missing_symptoms: List[str]
) -> str:
    """
    Assess the overall quality and completeness of evidence.

    Quality is determined by:
    - How many symptoms were matched vs provided
    - Whether sensor data is available and what it says
    - How many candidate faults exist and how well they scored
    - Whether evidence is consistent or contradictory

    Returns: "strong" | "moderate" | "weak"
    """

    if not candidates:
        return "weak"

    # Count matched vs total symptoms
    total_symptoms = len(matched_symptoms) + len(missing_symptoms)
    match_ratio = len(matched_symptoms) / total_symptoms if total_symptoms > 0 else 0.0

    # Check top candidate strength
    top_score = candidates[0].get("final_score", 0.0)
    top_matched = len(candidates[0].get("matched_symptoms", []))

    # Check for multiple strong candidates (ambiguity)
    strong_candidates = sum(
        1 for c in candidates
        if c.get("final_score", 0.0) >= 0.6
    )

    # Sensor contribution
    sensor_contributes = sensor_status in ("CONFIRMS", "CONTRADICTS")

    # ── Scoring ───────────────────────────────────────────────────

    quality_score = 0.0

    # Symptom coverage (0.0 - 0.4)
    quality_score += match_ratio * 0.4

    # Top candidate strength (0.0 - 0.3)
    quality_score += top_score * 0.3

    # Multiple strong candidates penalty (-0.1 per extra)
    if strong_candidates > 1:
        quality_score -= (strong_candidates - 1) * 0.1

    # Sensor confirmation bonus (+0.15) or contradiction penalty (-0.1)
    if sensor_status == "CONFIRMS":
        quality_score += 0.15
    elif sensor_status == "CONTRADICTS":
        quality_score -= 0.1

    # KG confirmation bonus (+0.1)
    if kg_status == "CONFIRMS":
        quality_score += 0.1

    # ── Classify ──────────────────────────────────────────────────

    if quality_score >= 0.65:
        return "strong"
    elif quality_score >= 0.4:
        return "moderate"
    else:
        return "weak"


# ─── Mode Determination ────────────────────────────────────────────

def _determine_mode(
    candidates: List[Dict],
    sensor_status: str,
    kg_status: str,
    matched_symptoms: List[str],
    missing_symptoms: List[str],
    evidence_quality: str
) -> tuple:
    """
    Determine diagnostic mode based on evidence quality/completenessor, not raw score thresholds.

    Mode represents the system's state of knowledge:
    - AMBIGUOUS: needs more information from technician
    - INFERRED: has a best guess, needs confirmation
    - EXTRACTED: has high confidence diagnosis

    Returns: (mode, mode_reason)
    """

    # ── No candidates at all ──────────────────────────────────────

    if not candidates:
        return (
            "AMBIGUOUS",
            "No matching faults found in the knowledge graph. "
            "Please provide more symptoms or describe the issue in more detail."
        )

    top = candidates[0]
    top_score = top.get("final_score", 0.0)
    top_matched_count = len(top.get("matched_symptoms", []))
    total_matched = len(matched_symptoms)

    # ── Strong evidence → EXTRACTED ───────────────────────────────

    if evidence_quality == "strong":
        # Check if sensor data contradicts despite strong KG evidence
        if sensor_status == "CONTRADICTS":
            return (
                "INFERRED",
                f"Knowledge graph strongly suggests {top.get('system', 'unknown')} "
                f"> {top.get('subsystem', 'unknown')}, but sensor readings contradict. "
                "Recommend verification before final diagnosis."
            )

        # Strong KG + sensor confirms or unavailable → EXTRACTED
        reason = (
            f"Multiple symptoms matched with high confidence. "
            f"Top candidate: {top.get('system', 'unknown')} > "
            f"{top.get('subsystem', 'unknown')}."
        )
        if sensor_status == "CONFIRMS":
            reason += " Sensor data confirms the diagnosis."
        elif sensor_status == "NOT AVAILABLE":
            reason += " No sensor data available, but graph evidence is strong."
        return ("EXTRACTED", reason)

    # ── Moderate evidence → INFERRED ──────────────────────────────

    if evidence_quality == "moderate":
        # Multiple strong candidates → still INFERRED but note ambiguity
        strong_count = sum(
            1 for c in candidates
            if c.get("final_score", 0.0) >= 0.6
        )

        if strong_count > 1:
            return (
                "INFERRED",
                f"Multiple possible faults identified ({strong_count} candidates "
                f"with similar confidence). Top suggestion: "
                f"{top.get('system', 'unknown')} > {top.get('subsystem', 'unknown')}. "
                f"Additional symptoms or sensor data would help narrow down."
            )

        return (
            "INFERRED",
            f"Moderate evidence points to {top.get('system', 'unknown')} > "
            f"{top.get('subsystem', 'unknown')}. "
            f"Matched {top_matched_count} of {total_matched} provided symptoms. "
            f"{'Sensor data available but inconclusive.' if sensor_status == 'NOT AVAILABLE' else ''}"
        )

    # ── Weak evidence → AMBIGUOUS ─────────────────────────────────

    if total_matched == 0:
        return (
            "AMBIGUOUS",
            "The provided symptoms did not match any known faults clearly. "
            "Please describe the issue in more detail or add specific symptoms."
        )

    if len(candidates) > 5:
        return (
            "AMBIGUOUS",
            f"Too many possible faults ({len(candidates)}) with similar confidence. "
            f"Please provide more specific symptoms to narrow down."
        )

    return (
        "AMBIGUOUS",
        f"Weak evidence for any single fault. "
        f"Top suggestion: {top.get('system', 'unknown')} > "
        f"{top.get('subsystem', 'unknown')} (confidence: {top_score:.0%}). "
        f"Please verify or provide additional symptoms."
    )


# ─── Fault Categorization ──────────────────────────────────────────

def _categorize_faults(candidates: List[Dict]) -> tuple:
    """
    Categorize candidate faults into confirmed, contradicted, and remaining.

    - Confirmed: sensor_status == "CONFIRMED" and good KG score
    - Contradicted: sensor_status == "CONTRADICTED"
    - Remaining: everything else

    Returns: (confirmed, contradicted, remaining)
    """
    confirmed = []
    contradicted = []
    remaining = []

    for fault in candidates:
        sensor_status = fault.get("sensor_status", "NOT AVAILABLE")
        score = fault.get("final_score", 0.0)

        if sensor_status == "CONFIRMED" and score >= 0.6:
            confirmed.append(fault)
        elif sensor_status == "CONTRADICTED":
            contradicted.append(fault)
        else:
            remaining.append(fault)

    return confirmed, contradicted, remaining


# ─── Reasoning Chain Builder ───────────────────────────────────────

def _build_reasoning_chain(
    candidates: List[Dict],
    sensor_status: str,
    kg_status: str,
    matched_symptoms: List[str],
    missing_symptoms: List[str],
    evidence_quality: str,
    mode: str
) -> List[Dict[str, str]]:
    """
    Build a step-by-step reasoning chain explaining how the
    diagnosis was reached.

    Returns list of {"step": str, "detail": str} dicts.
    """
    chain = []

    # Step 1: Query analysis
    total_symptoms = len(matched_symptoms) + len(missing_symptoms)
    chain.append({
        "step": "Query Analysis",
        "detail": f"Received {total_symptoms} symptom(s): "
                  f"{', '.join(matched_symptoms + missing_symptoms)}"
    })

    # Step 2: KG matching
    if candidates:
        chain.append({
            "step": "Knowledge Graph Matching",
            "detail": f"Found {len(candidates)} candidate fault(s). "
                      f"Top match: {candidates[0].get('system', 'Unknown')} > "
                      f"{candidates[0].get('subsystem', 'Unknown')} "
                      f"(score: {candidates[0].get('final_score', 0.0):.2f})"
        })
    else:
        chain.append({
            "step": "Knowledge Graph Matching",
            "detail": "No matching faults found in the knowledge graph."
        })

    # Step 3: Sensor validation
    if sensor_status == "NOT AVAILABLE":
        chain.append({
            "step": "Sensor Validation",
            "detail": "No sensor data available for this fault. "
                      "Diagnosis relies on knowledge graph evidence only."
        })
    elif sensor_status == "CONFIRMS":
        chain.append({
            "step": "Sensor Validation",
            "detail": "Sensor readings CONFIRM the fault. "
                      "Anomalous readings detected in relevant sensors."
        })
    elif sensor_status == "CONTRADICTS":
        chain.append({
            "step": "Sensor Validation",
            "detail": "Sensor readings CONTRADICT the fault. "
                      "Sensor readings appear normal for this fault type."
        })

    # Step 4: Evidence fusion
    chain.append({
        "step": "Evidence Fusion",
        "detail": f"Evidence quality: {evidence_quality}. "
                  f"KG status: {kg_status}. "
                  f"Sensor status: {sensor_status}."
    })

    # Step 5: Mode determination
    chain.append({
        "step": "Mode Determination",
        "detail": f"Mode: {mode}. "
                  f"Matched {len(matched_symptoms)}/{total_symptoms} symptoms."
    })

    # Step 6: Symptom gap analysis
    if missing_symptoms:
        chain.append({
            "step": "Symptom Gap Analysis",
            "detail": f"Missing symptoms (not in KG): {', '.join(missing_symptoms)}. "
                      f"These may indicate a fault not covered by the knowledge graph."
        })

    return chain


# ─── Internal Confidence ───────────────────────────────────────────

def _compute_confidence(
    candidates: List[Dict],
    evidence_quality: str,
    sensor_status: str,
    mode: str
) -> float:
    """
    Compute internal confidence score for the diagnosis.

    This is used internally for mode determination and ranking.
    It is NOT exposed in the UI (per architectural constraint #4).

    Returns: float between 0.0 and 1.0
    """
    if not candidates:
        return 0.0

    base_confidence = candidates[0].get("final_score", 0.0)

    # Adjust based on evidence quality
    quality_multiplier = {
        "strong": 1.0,
        "moderate": 0.8,
        "weak": 0.5
    }.get(evidence_quality, 0.5)

    # Adjust based on sensor status
    sensor_adjustment = {
        "CONFIRMS": 0.1,
        "CONTRADICTS": -0.15,
        "NOT AVAILABLE": 0.0
    }.get(sensor_status, 0.0)

    confidence = (base_confidence * quality_multiplier) + sensor_adjustment
    return max(0.0, min(1.0, confidence))


# ─── Public Helpers ────────────────────────────────────────────────

def format_reasoning_chain(chain: List[Dict[str, str]]) -> str:
    """
    Format reasoning chain into human-readable text.
    """
    lines = []
    for i, step in enumerate(chain, 1):
        lines.append(f"{i}. {step['step']}: {step['detail']}")
    return "\n".join(lines)


def get_mode_description(mode: str) -> str:
    """
    Get a user-friendly description of a diagnostic mode.
    """
    descriptions = {
        "AMBIGUOUS": (
            "The system needs more information. "
            "Please provide additional symptoms or describe the issue in more detail."
        ),
        "INFERRED": (
            "The system has a best guess based on available evidence. "
            "Please confirm or provide additional information."
        ),
        "EXTRACTED": (
            "The system has high confidence in the diagnosis. "
            "Please review the recommended actions."
        )
    }
    return descriptions.get(mode, "Unknown mode")
