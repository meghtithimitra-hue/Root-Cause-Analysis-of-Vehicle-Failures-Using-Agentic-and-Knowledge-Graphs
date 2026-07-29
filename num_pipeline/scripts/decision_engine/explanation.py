"""Explanation generator for the decision engine.

Generates plain-English diagnosis summaries and AI-assisted
analysis.  Uses LLM (Ollama) when available, with deterministic
template fallback.

This module is the **single source of truth** for all diagnosis
summaries displayed in the UI.  The Streamlit UI (``app.py``) only
renders what this module produces — it never generates its own
diagnosis text.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# System prompts for LLM calls
# ---------------------------------------------------------------------------

_DIAGNOSIS_SUMMARY_SYSTEM_PROMPT = (
    "You are an automotive diagnostic assistant. Using ONLY the "
    "provided knowledge graph data, write a concise plain-English "
    "diagnosis summary. Explain what the most likely fault is and "
    "why the system selected it based on the matched symptoms. "
    "Do NOT mention recommended inspection steps, confidence scores, "
    "retrieval scores, embeddings, calibration, or any implementation "
    "details. Keep it under 100 words."
)

_AI_ASSISTED_SYSTEM_PROMPT = (
    "You are an automotive diagnostic assistant. The system could not "
    "determine a confident diagnosis. Using the retrieved knowledge graph "
    "candidates as context, provide helpful guidance about possible issues "
    "related to the user's symptoms. Clearly state that this is general "
    "information, not a definitive diagnosis."
)


# ---------------------------------------------------------------------------
# KG graph helpers (absorbed from app.py)
# ---------------------------------------------------------------------------

_graph_data = None


def _load_graph():
    """Load the KG JSON (cached across calls)."""
    global _graph_data
    if _graph_data is not None:
        return _graph_data

    # Walk up from this file to find num_pipeline/data/processed/
    here = Path(__file__).resolve().parent
    graph_path = here.parent.parent / "data" / "processed" / "hierarchical_graph.json"
    if not graph_path.exists():
        return None

    with open(graph_path, encoding="utf-8") as f:
        _graph_data = json.load(f)
    return _graph_data


def _lookup_diagnosis_steps(subcategory_label: str) -> List[str]:
    """Return diagnosis steps for a subcategory from the KG."""
    if not subcategory_label:
        return []

    graph = _load_graph()
    if graph is None:
        return []

    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    subcat_id = None
    for n in nodes:
        if n.get("node_type") == "Subcategory" and n.get("label") == subcategory_label:
            subcat_id = n["id"]
            break
    if not subcat_id:
        return []

    step_ids = {
        e["target"] for e in edges
        if e.get("source") == subcat_id and e.get("relation") == "HAS_DIAGNOSIS_STEP"
    }
    return [n["label"] for n in nodes if n.get("id") in step_ids]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_brief_summary(
    mode: str,
    top_candidate: Dict[str, Any],
    confidence: float,
) -> str:
    """Generate a one-line summary of the diagnosis.

    Parameters
    ----------
    mode : str
        The determined mode.
    top_candidate : dict
        The top-ranked candidate.
    confidence : float
        Calibrated confidence.

    Returns
    -------
    str
        One-line summary string.
    """
    label = top_candidate.get("label", "Unknown")
    fault = top_candidate.get("navic_fault", "")
    fault_str = f" ({fault})" if fault else ""

    mode_labels = {
        "EXTRACTED": "Diagnosis",
        "INFERRED": "Best guess",
        "AMBIGUOUS": "Insufficient evidence",
    }
    mode_label = mode_labels.get(mode, mode)

    return f"{mode_label}: {label}{fault_str}"


def generate_diagnosis_summary(
    mode: str,
    top_candidate: Dict[str, Any],
    original_symptoms: List[str],
    sensor_evidence: Dict[str, Any],
    llm_provider=None,
) -> str:
    """Generate a plain-English diagnosis summary from KG data.

    This is the **single source of truth** for the diagnosis summary
    displayed in the UI.  Uses the LLM when available, with a
    deterministic template fallback.

    The summary explains what the fault is, why it was selected,
    what the symptoms indicate, and what to inspect next.  It never
    mentions confidence scores, retrieval, embeddings, or other
    internals.

    Parameters
    ----------
    mode : str
        EXTRACTED, INFERRED, or AMBIGUOUS.
    top_candidate : dict
        The top-ranked candidate.
    original_symptoms : list[str]
        User-reported symptoms.
    sensor_evidence : dict
        Per-fault sensor validation results.
    llm_provider : optional
        ``LLMProvider`` instance.

    Returns
    -------
    str
        Plain-English diagnosis summary (markdown).
    """
    if not top_candidate or not top_candidate.get("label"):
        return ""

    label = top_candidate.get("label", "")
    fault = top_candidate.get("navic_fault", "")
    category = top_candidate.get("category", "")
    subcategory = top_candidate.get("subcategory", "")

    # Sensor evidence
    sensor_detail = _format_sensor_detail(fault, sensor_evidence)

    # Matched symptoms
    symptoms_text = ", ".join(original_symptoms[:5]) if original_symptoms else "none provided"

    # Build structured context for LLM
    context = (
        f"Mode: {mode}\n"
        f"Top diagnosis: {label}\n"
        f"Knowledge graph category: {category}\n"
        f"Knowledge graph subcategory: {subcategory}\n"
        f"Matched symptoms: {symptoms_text}\n"
    )
    if sensor_detail:
        context += f"Sensor evidence: {sensor_detail}\n"

    # Try LLM
    if llm_provider is not None and llm_provider.is_available():
        prompt = (
            f"{context}\n\n"
            f"Write a concise diagnosis summary based on this knowledge "
            f"graph data."
        )
        response = llm_provider.generate(prompt, _DIAGNOSIS_SUMMARY_SYSTEM_PROMPT)
        if response:
            return response.strip()

    # Template fallback
    return _template_diagnosis_summary(
        mode, label, category, subcategory,
        symptoms_text, sensor_detail,
    )


def generate_inspection_steps(
    top_candidate: Dict[str, Any],
) -> List[str]:
    """Return the recommended inspection steps for the top candidate.

    Parameters
    ----------
    top_candidate : dict
        The top-ranked candidate.

    Returns
    -------
    list[str]
        Inspection step descriptions.
    """
    subcategory = top_candidate.get("subcategory", "") if top_candidate else ""
    return _lookup_diagnosis_steps(subcategory)


def generate_ai_assisted_analysis(
    symptoms: str,
    candidates: List[Dict[str, Any]],
    llm_provider=None,
) -> str:
    """Generate an AI-assisted analysis for AMBIGUOUS skip path.

    When the user skips clarification, this provides a helpful response
    grounded in whatever KG candidates were retrieved, or independent
    guidance if nothing was found.

    Parameters
    ----------
    symptoms : str
        Original user symptom string.
    candidates : list[dict]
        Retrieved KG candidates (may be empty).
    llm_provider : optional
        ``LLMProvider`` instance.

    Returns
    -------
    str
        AI-assisted analysis string with disclaimer.
    """
    context = _build_ai_assisted_context(symptoms, candidates)

    if llm_provider is not None and llm_provider.is_available():
        prompt = _build_ai_assisted_prompt(context)
        response = llm_provider.generate(prompt, _AI_ASSISTED_SYSTEM_PROMPT)
        if response:
            return response + "\n\n" + _get_disclaimer()

    return _template_ai_assisted(context) + "\n\n" + _get_disclaimer()


# ---------------------------------------------------------------------------
# Context builders
# ---------------------------------------------------------------------------

def _build_ai_assisted_context(symptoms, candidates):
    """Build context for AI-assisted analysis."""
    candidate_summaries = []
    for c in candidates[:3]:
        candidate_summaries.append({
            "label": c.get("label", ""),
            "score": c.get("score", c.get("confidence", 0.0)),
            "source": c.get("source", ""),
        })
    return {
        "symptoms": symptoms,
        "candidates": candidate_summaries,
        "has_candidates": len(candidates) > 0,
    }


# ---------------------------------------------------------------------------
# Sensor detail helper
# ---------------------------------------------------------------------------

def _format_sensor_detail(fault, sensor_evidence):
    """Format sensor evidence into a human-readable string.

    Uses the sensor dictionary to present display names and
    descriptions alongside raw INCA names.
    """
    if not fault or fault not in sensor_evidence:
        return ""

    from .sensor_explanation import enrich_sensor

    se = sensor_evidence[fault]
    parts = []

    for level, label in [("critical", "CRITICAL"), ("warning", "WARNING")]:
        sensors = se.get(level, [])
        if sensors:
            for s in sensors:
                info = enrich_sensor(s)
                desc = info["description"]
                desc_str = f" — {desc}" if desc else ""
                parts.append(
                    f"{info['display_name']} ({s}): {label}{desc_str}"
                )

    if parts:
        return "; ".join(parts)
    return "All sensor readings are within the normal range."


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_ai_assisted_prompt(context):
    """Build the LLM prompt for AI-assisted analysis."""
    prompt = f"User symptoms: \"{context['symptoms']}\"\n"

    if context["has_candidates"]:
        prompt += "Retrieved KG candidates (not confident enough for diagnosis):\n"
        for c in context["candidates"]:
            prompt += f"- {c['label']} (score: {c['score']:.3f}, source: {c['source']})\n"
        prompt += "\nProvide guidance based on these candidates."
    else:
        prompt += "No relevant KG candidates were found.\n"
        prompt += "Provide general guidance based on the symptoms described."

    return prompt


# ---------------------------------------------------------------------------
# Template fallbacks
# ---------------------------------------------------------------------------

def _template_diagnosis_summary(
    mode, label, category, subcategory,
    symptoms_text, sensor_detail,
):
    """Deterministic template for the diagnosis summary."""
    parts = []

    if mode == "EXTRACTED":
        parts.append(f"**Diagnosis:** {label}")
    elif mode == "INFERRED":
        parts.append(f"**Most likely fault:** {label}")
    else:
        return ""

    if category and subcategory:
        parts.append(
            f"This fault falls under **{subcategory}** in the "
            f"**{category}** system."
        )

    if symptoms_text and symptoms_text != "none provided":
        parts.append(
            f"The reported symptoms ({symptoms_text}) are consistent "
            f"with this type of fault."
        )

    if sensor_detail:
        parts.append(f"**Sensor status:** {sensor_detail}")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# AI-Assisted Analysis
# ---------------------------------------------------------------------------

def _template_ai_assisted(context):
    """Template for AI-assisted analysis (AMBIGUOUS skip path)."""
    symptoms = context["symptoms"]

    text = f"Based on the symptoms \"{symptoms}\", "
    text += "here is what the knowledge graph suggests:\n\n"

    if context["has_candidates"]:
        for c in context["candidates"]:
            text += f"- **{c['label']}** — found via {c['source']} retrieval.\n"
        text += "\nThese are the closest matches in the knowledge base, "
        text += "but none had sufficient confidence for a definitive diagnosis. "
        text += "Consider consulting a mechanic for hands-on inspection."
    else:
        text += "No closely related fault patterns were found in the knowledge base. "
        text += "The described symptoms do not match known patterns. "
        text += "Consider consulting a mechanic for a professional diagnosis."

    return text


def _get_disclaimer():
    return (
        "---\n"
        "\u26a0 AI-ASSISTED ANALYSIS \u2014 NOT A DIAGNOSIS\n"
        "This response is generated by an AI language model and is NOT a "
        "knowledge-graph-derived diagnosis. It should be treated as general "
        "informational guidance only."
    )
