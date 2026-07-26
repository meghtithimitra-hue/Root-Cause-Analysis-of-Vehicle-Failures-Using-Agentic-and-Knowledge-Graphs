"""Explanation generator for the decision engine.

Converts a diagnostic report into natural language explanations.
Uses LLM (Ollama) when available, with deterministic template fallback.

Also handles the AI-Assisted Analysis path for AMBIGUOUS skip.

This module is the **single source of truth** for all diagnosis
summaries and explanations.  The Streamlit UI (``app.py``) only
renders what this module produces — it never generates its own
diagnosis text.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# System prompts for LLM calls
# ---------------------------------------------------------------------------

_EXPLANATION_SYSTEM_PROMPT = (
    "You are an automotive diagnostic assistant. Explain the diagnosis "
    "clearly and concceely. Use the provided reasoning chain and evidence "
    "to justify your explanation. Be factual — do not speculate beyond "
    "the evidence."
)

_DIAGNOSIS_SUMMARY_SYSTEM_PROMPT = (
    "You are an automotive diagnostic assistant. Using ONLY the "
    "provided knowledge graph data, write a concise plain-English "
    "diagnosis summary. Explain what the most likely fault is, why "
    "the system selected it based on the matched symptoms, and what "
    "the recommended next inspection steps are. Do NOT mention "
    "confidence scores, retrieval scores, embeddings, calibration, "
    "or any implementation details. Keep it under 150 words."
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

def generate_explanation(
    mode: str,
    top_candidate: Dict[str, Any],
    display_candidates: List[Dict[str, Any]],
    reasoning_chain: List[Dict[str, Any]],
    sensor_evidence: Dict[str, Any],
    llm_provider=None,
) -> str:
    """Generate a natural language explanation for the diagnosis.

    Parameters
    ----------
    mode : str
        EXTRACTED, INFERRED, or AMBIGUOUS.
    top_candidate : dict
        The top-ranked candidate with confidence and metadata.
    display_candidates : list[dict]
        All candidates shown to the user (within mode band).
    reasoning_chain : list[dict]
        Output of ``build_reasoning_chain()``.
    sensor_evidence : dict
        Per-fault sensor validation results.
    llm_provider : optional
        ``LLMProvider`` instance, or None for template-only.

    Returns
    -------
    str
        Natural language explanation.
    """
    context = _build_context(
        mode, top_candidate, display_candidates,
        reasoning_chain, sensor_evidence,
    )

    if llm_provider is not None and llm_provider.is_available():
        prompt = _build_explanation_prompt(mode, context)
        response = llm_provider.generate(prompt, _EXPLANATION_SYSTEM_PROMPT)
        if response:
            return response

    return _template_explanation(mode, context)


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

    # Diagnosis steps from KG
    steps = _lookup_diagnosis_steps(subcategory)

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
    if steps:
        context += "Recommended inspection steps:\n"
        context += "\n".join(f"- {s}" for s in steps[:4]) + "\n"
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
        symptoms_text, steps, sensor_detail,
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

def _build_context(
    mode, top_candidate, display_candidates, reasoning_chain, sensor_evidence,
):
    """Build a context dict for template/LLM generation."""
    return {
        "mode": mode,
        "top_candidate": top_candidate,
        "display_candidates": display_candidates,
        "reasoning_chain": reasoning_chain,
        "sensor_evidence": sensor_evidence,
    }


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
    """Format sensor evidence into a human-readable string."""
    if not fault or fault not in sensor_evidence:
        return ""
    se = sensor_evidence[fault]
    critical = se.get("critical", [])
    warning = se.get("warning", [])
    if critical:
        return f"Sensor readings flagged critical issues: {', '.join(critical)}."
    elif warning:
        return f"Sensor readings showed warnings: {', '.join(warning)}."
    return "Sensor readings are within normal range."


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_explanation_prompt(mode, context):
    """Build the LLM prompt for explanation generation."""
    top = context["top_candidate"]
    chain_text = "\n".join(
        f"- {s['step']}: {s['detail']}" for s in context["reasoning_chain"]
    )

    prompt = f"Mode: {mode}\n"
    prompt += f"Top diagnosis: {top.get('label', 'Unknown')} "
    prompt += f"(confidence: {top.get('confidence', 0.0):.0%})\n"
    prompt += f"Reasoning chain:\n{chain_text}\n"
    prompt += "\nProvide a clear explanation of this diagnosis."

    return prompt


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

def _template_explanation(mode, context):
    """Dispatch to mode-specific template."""
    templates = {
        "EXTRACTED": _template_extracted,
        "INFERRED": _template_inferred,
        "AMBIGUOUS": _template_ambiguous,
    }
    template_fn = templates.get(mode, _template_ambiguous)
    return template_fn(context)


def _template_extracted(context):
    top = context["top_candidate"]
    label = top.get("label", "Unknown")
    fault = top.get("navic_fault", "")

    text = f"**Diagnosis: {label}**\n\n"
    text += "The system has high confidence in this diagnosis "
    text += "based on strong evidence from the knowledge graph.\n\n"

    chain = context["reasoning_chain"]
    if chain:
        text += "**Reasoning:**\n"
        for s in chain:
            text += f"- {s['detail']}\n"
        text += "\n"

    sensor = context.get("sensor_evidence", {})
    if fault and fault in sensor:
        si = sensor[fault]
        if si.get("critical") or si.get("warning"):
            text += "**Sensor confirmation:** Sensor data supports this diagnosis.\n"
        else:
            text += "**Sensor status:** No strong sensor confirmation available.\n"
    else:
        text += "**Sensor status:** No numerical sensor data available for this fault.\n"

    return text.strip()


def _template_inferred(context):
    top = context["top_candidate"]
    label = top.get("label", "Unknown")

    text = f"**Best guess: {label}**\n\n"
    text += "This is the most likely fault based on available evidence, "
    text += "but additional information could improve certainty.\n\n"

    alts = context["display_candidates"]
    if len(alts) > 1:
        text += "**Alternative possibilities:**\n"
        for c in alts[1:3]:
            text += f"- {c.get('label', '')}\n"
        text += "\n"

    chain = context["reasoning_chain"]
    if chain:
        text += "**Reasoning:**\n"
        for s in chain:
            text += f"- {s['detail']}\n"
        text += "\n"

    text += "You can provide additional symptoms to refine this diagnosis, "
    text += "or accept it as-is."

    return text.strip()


def _template_ambiguous(context):
    top = context["top_candidate"]

    text = "**Insufficient evidence for a confident diagnosis.**\n\n"

    if top and top.get("confidence", 0) > 0:
        label = top.get("label", "Unknown")
        text += f"The closest match is \"{label}\", "
        text += "but this does not meet the threshold for a reliable diagnosis.\n\n"

    text += "Please provide more details about your symptoms, such as:\n"
    text += "- When the issue occurs (e.g., at specific speeds, temperatures)\n"
    text += "- Any error codes displayed\n"
    text += "- Additional symptoms you have noticed\n"

    return text.strip()


def _template_diagnosis_summary(
    mode, label, category, subcategory,
    symptoms_text, steps, sensor_detail,
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

    if steps:
        parts.append("**Recommended next steps:**")
        for s in steps[:3]:
            parts.append(f"- {s}")

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
