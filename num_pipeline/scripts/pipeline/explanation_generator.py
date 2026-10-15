"""
Explanation Generator — converts DiagnosticDecision into natural language.

This module uses the LLM for presentation ONLY. It does NOT do diagnosis
or reasoning. The reasoning engine has already determined the mode,
reasoning chain, and categorized faults. This module simply presents
that information in a user-friendly format.

If LLM is unavailable, falls back to template-based explanations.
"""

from typing import Dict, Optional
from .reasoning_engine import DiagnosticDecision


def get_sensor_summary(sensor_analysis):
    """Get a concise sensor analysis summary."""
    if not sensor_analysis:
        return "No sensor data available"
    critical = sensor_analysis.get("critical", [])
    warning = sensor_analysis.get("warning", [])
    normal = sensor_analysis.get("normal", [])
    sensor_confidence = sensor_analysis.get("sensor_confidence", 0.0)
    summary_parts = []
    if critical:
        summary_parts.append(f"{len(critical)} critical")
    if warning:
        summary_parts.append(f"{len(warning)} warning")
    if normal:
        summary_parts.append(f"{len(normal)} normal")
    summary = ", ".join(summary_parts) if summary_parts else "No sensor readings"
    return f"Sensor Analysis ({sensor_confidence:.0%} confidence): {summary}"


# ─── System Prompt ──────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a vehicle fault diagnosis assistant. You present
diagnosis results to a technician in clear, concise language.

Rules:
- Present facts from the reasoning chain, do NOT add your own diagnosis
- Use the evidence provided, do NOT speculate beyond it
- Be concise and actionable
- If sensor data is unavailable, say so without downgrading graph evidence
- Never expose internal confidence scores to the user"""


# ─── Prompt Templates ──────────────────────────────────────────────

AMBIGUOUS_TEMPLATE = """Based on the following diagnosis results, provide a
clear explanation asking the technician for more information.

Mode: AMBIGUOUS
Reason: {mode_reason}

Symptoms provided: {symptoms}
Missing symptoms: {missing_symptoms}

Top candidates found:
{candidates}

Please explain:
1. What symptoms were analyzed
2. Why the system is uncertain
3. What additional information would help narrow down the diagnosis
4. Offer to try with different symptoms"""

INFERRED_TEMPLATE = """Based on the following diagnosis results, provide a
clear explanation of the best-guess diagnosis.

Mode: INFERRED
Reason: {mode_reason}

Predicted system: {predicted_system}
Predicted subsystem: {predicted_subsystem}

Reasoning chain:
{reasoning_chain}

Sensor status: {sensor_status}
Sensor summary: {sensor_summary}

Please explain:
1. What the most likely fault is
2. Why the system is not fully confident
3. What sensor data shows (if available)
4. Suggest next steps for confirmation"""

EXTRACTED_TEMPLATE = """Based on the following diagnosis results, provide a
clear, professional diagnosis explanation.

Mode: EXTRACTED
Reason: {mode_reason}

Confirmed faults:
{confirmed_faults}

Reasoning chain:
{reasoning_chain}

Sensor status: {sensor_status}
Sensor summary: {sensor_summary}

Please explain:
1. The diagnosed fault (system > subsystem)
2. Key symptoms that led to this diagnosis
3. What the evidence shows (KG matches + sensor data)
4. Recommended diagnostic steps from the knowledge graph"""


# ─── Main Generator ────────────────────────────────────────────────

def generate_explanation(
    decision: DiagnosticDecision,
    evidence: Dict,
    llm_provider=None
) -> str:
    """
    Generate a natural language explanation from a DiagnosticDecision.

    Args:
        decision: Output from reasoning_engine.reason()
        evidence: Output from evidence_fusion.fuse()
        llm_provider: Optional LLMProvider instance

    Returns:
        Natural language explanation string
    """

    # ── Build context from evidence ───────────────────────────────

    context = _build_context(decision, evidence)

    # ── Try LLM generation ────────────────────────────────────────

    if llm_provider and llm_provider.is_available():
        prompt = _build_prompt(decision.mode, context)
        response = llm_provider.generate(prompt, SYSTEM_PROMPT)
        if response:
            return response

    # ── Fallback to template ──────────────────────────────────────

    return _template_explanation(decision, context)


# ─── Context Building ──────────────────────────────────────────────

def _build_context(decision: DiagnosticDecision, evidence: Dict) -> Dict:
    """Build a context dict for prompt generation."""
    candidates = evidence.get("candidate_faults", [])
    matched_symptoms = evidence.get("matched_symptoms", [])
    missing_symptoms = evidence.get("missing_symptoms", [])

    # Format candidates
    candidates_text = ""
    for i, c in enumerate(candidates[:3], 1):
        system = c.get("system", "Unknown")
        subsystem = c.get("subsystem", "Unknown")
        score = c.get("final_score", 0.0)
        matched = c.get("matched_symptoms", [])
        candidates_text += (
            f"  {i}. {system} > {subsystem} "
            f"(confidence: {score:.2f})\n"
            f"     Matched symptoms: {', '.join(matched)}\n"
        )

    # Format confirmed faults
    confirmed_text = ""
    for f in decision.confirmed_faults:
        confirmed_text += (
            f"  - {f.get('system', 'Unknown')} > "
            f"{f.get('subsystem', 'Unknown')}\n"
        )

    # Format reasoning chain
    chain_text = ""
    for i, step in enumerate(decision.reasoning_chain, 1):
        chain_text += f"  {i}. {step['step']}: {step['detail']}\n"

    # Sensor info
    sensor_summary = get_sensor_summary(evidence.get("sensor_evidence"))

    return {
        "mode": decision.mode,
        "mode_reason": decision.mode_reason,
        "predicted_system": decision.predicted_system or "Unknown",
        "predicted_subsystem": decision.predicted_subsystem or "Unknown",
        "candidates_text": candidates_text,
        "confirmed_text": confirmed_text,
        "reasoning_chain": chain_text,
        "sensor_status": evidence.get("sensor_status", "NOT AVAILABLE"),
        "sensor_summary": sensor_summary,
        "symptoms": ", ".join(matched_symptoms + missing_symptoms),
        "missing_symptoms": ", ".join(missing_symptoms) if missing_symptoms else "None"
    }


# ─── Prompt Building ───────────────────────────────────────────────

def _build_prompt(mode: str, context: Dict) -> str:
    """Build the LLM prompt based on mode."""
    template = {
        "AMBIGUOUS": AMBIGUOUS_TEMPLATE,
        "INFERRED": INFERRED_TEMPLATE,
        "EXTRACTED": EXTRACTED_TEMPLATE
    }.get(mode, AMBIGUOUS_TEMPLATE)

    return template.format(**context)


# ─── Template Fallback ─────────────────────────────────────────────

def _template_explanation(decision: DiagnosticDecision, context: Dict) -> str:
    """Generate explanation using templates (no LLM)."""
    mode = decision.mode

    if mode == "AMBIGUOUS":
        return _template_ambiguous(context)
    elif mode == "INFERRED":
        return _template_inferred(context)
    elif mode == "EXTRACTED":
        return _template_extracted(context)
    return "Unable to generate explanation."


def _template_ambiguous(context: Dict) -> str:
    """Template for AMBIGUOUS mode."""
    return f"""## Diagnosis Status: Need More Information

**Symptoms analyzed:** {context['symptoms']}

**What we found:**
The system searched the knowledge graph but could not identify a clear match
for the provided symptoms. This could mean:
- The symptoms are too general or vague
- The fault may not be in the current knowledge base
- Multiple systems could be involved

**Missing information:**
{context['missing_symptoms'] if context['missing_symptoms'] != 'None' else 'No specific symptoms were identified as missing.'}

**Recommendation:**
Please provide more specific symptoms or describe the issue in more detail.
For example:
- What component is affected?
- When does the issue occur (start-up, driving, braking)?
- Are there any warning lights or unusual sounds?"""


def _template_inferred(context: Dict) -> str:
    """Template for INFERRED mode."""
    return f"""## Diagnosis: Best Guess (Needs Confirmation)

**Most likely fault:** {context['predicted_system']} > {context['predicted_subsystem']}

**Why this is uncertain:**
{context['mode_reason']}

**Evidence:**
{context['candidates_text']}

**Sensor validation:** {context['sensor_status']}
{context['sensor_summary']}

**Reasoning:**
{context['reasoning_chain']}

**Recommendation:**
This is a best-guess diagnosis based on available evidence. To confirm:
1. Verify the specific symptoms against the predicted fault
2. If sensor data is available, check the readings
3. Perform the diagnostic steps listed above
4. If uncertain, provide additional symptoms for re-analysis"""


def _template_extracted(context: Dict) -> str:
    """Template for EXTRACTED mode."""
    return f"""## Diagnosis: {context['predicted_system']} > {context['predicted_subsystem']}

**Status:** High confidence diagnosis

**Reason:** {context['mode_reason']}

**Matched Symptoms:**
{context['candidates_text']}

**Sensor Validation:** {context['sensor_status']}
{context['sensor_summary']}

**Reasoning Chain:**
{context['reasoning_chain']}

**Confirmed by sensor data:** {'Yes' if context['sensor_status'] == 'CONFIRMS' else 'No sensor data available — diagnosis based on knowledge graph evidence only'}

**Recommended Actions:**
1. Verify the predicted fault visually or with diagnostic tools
2. Check the specific subsystem mentioned above
3. Follow the diagnosis steps from the knowledge graph
4. If sensor data is available, cross-reference with the readings"""


# ─── Summary Generator ─────────────────────────────────────────────

def generate_brief_summary(
    decision: DiagnosticDecision,
    evidence: Dict
) -> str:
    """
    Generate a one-line summary of the diagnosis.
    """
    if decision.mode == "AMBIGUOUS":
        return "Need more information — please provide additional symptoms."

    system = decision.predicted_system or "Unknown"
    subsystem = decision.predicted_subsystem or "Unknown"

    if decision.mode == "INFERRED":
        return f"Best guess: {system} > {subsystem} (needs confirmation)"
    else:
        return f"Diagnosed: {system} > {subsystem}"


# ─── AI-Assisted Analysis (AMBIGUOUS Skip Path) ────────────────────

AI_ASSISTED_SYSTEM_PROMPT = """You are an automotive diagnostic assistant providing
general guidance based on your automotive knowledge. You are NOT performing a
knowledge-graph-based diagnosis.

IMPORTANT RULES:
- You are providing GENERAL AUTOMOTIVE GUIDANCE, not a confirmed diagnosis
- Be transparent that this is AI-generated guidance based on general knowledge
- Clearly state that the knowledge graph could not confidently identify a fault
- Present possibilities as general guidance, not confirmed faults
- Recommend professional verification before any action
- Never claim certainty about a specific fault
- Suggest what a technician should check based on common automotive knowledge"""

AI_ASSISTED_TEMPLATE = """A technician has described the following vehicle symptoms, but the
knowledge graph-based diagnostic system could not confidently identify a fault from its
database of known issues.

The system needs additional information or the symptoms may not match known patterns
in the knowledge base.

**Original symptoms described:**
{symptoms}

**What the knowledge graph found:**
- {candidate_count} partial matches were found, but none with sufficient confidence
- Top partial matches: {partial_matches}

Based on your general automotive knowledge, please provide helpful guidance:

1. What are the most COMMON causes of these symptoms in general automotive repair?
2. What additional symptoms or details would help narrow down the diagnosis?
3. What should a technician check first when encountering these symptoms?
4. Are there any quick checks or tests that could help identify the issue?

Remember: This is GENERAL GUIDANCE, not a confirmed diagnosis from the knowledge graph.
Always recommend professional verification."""


def generate_ai_assisted_analysis(
    symptoms: str,
    evidence: Dict,
    llm_provider=None
) -> str:
    """
    Generate an AI-Assisted Analysis for AMBIGUOUS mode skip path.

    This function is called when:
    1. The pipeline reaches AMBIGUOUS mode
    2. The user chooses to skip clarification (or provides insufficient info)
    3. We need to provide helpful guidance instead of asking again

    The output is clearly labeled as AI-generated guidance based on general
    automotive knowledge, NOT a knowledge-graph-derived diagnosis.

    Args:
        symptoms: Original symptoms string
        evidence: Partial evidence from the pipeline
        llm_provider: LLMProvider instance (required for this function)

    Returns:
        AI-generated guidance string, or template fallback if LLM unavailable
    """
    # Build context for the AI-assisted prompt
    candidates = evidence.get("candidate_faults", [])
    partial_matches = ""
    if candidates:
        matches = []
        for c in candidates[:3]:
            system = c.get("system", "Unknown")
            subsystem = c.get("subsystem", "Unknown")
            matches.append(f"{system} > {subsystem}")
        partial_matches = "; ".join(matches)
    else:
        partial_matches = "None found"

    context = {
        "symptoms": symptoms,
        "candidate_count": len(candidates),
        "partial_matches": partial_matches
    }

    # Try LLM generation
    if llm_provider and llm_provider.is_available():
        prompt = AI_ASSISTED_TEMPLATE.format(**context)
        response = llm_provider.generate(prompt, AI_ASSISTED_SYSTEM_PROMPT)
        if response:
            # Prepend the AI-generated disclaimer
            disclaimer = _get_ai_assisted_disclaimer()
            return f"{disclaimer}\n\n{response}"

    # Fallback to template
    return _template_ai_assisted_analysis(context)


def _get_ai_assisted_disclaimer() -> str:
    """Get the standard AI-assisted analysis disclaimer."""
    return """## AI-Assisted Analysis

> **Important:** The following is **AI-generated guidance** based on general automotive
> knowledge. It is **NOT** a knowledge-graph-derived diagnosis.
>
> The diagnostic knowledge graph could not confidently identify a fault from the
> available symptoms. This guidance is provided to help direct your investigation,
> but should **not** be treated as a confirmed diagnosis.
>
> **Always verify with professional diagnostic tools and expertise before taking action.**"""


def _template_ai_assisted_analysis(context: Dict) -> str:
    """Template-based AI-assisted analysis (no LLM)."""
    return f"""## AI-Assisted Analysis

> **Important:** The following is **AI-generated guidance** based on general automotive
> knowledge. It is **NOT** a knowledge-graph-derived diagnosis.
>
> The diagnostic knowledge graph could not confidently identify a fault from the
> available symptoms. This guidance is provided to help direct your investigation,
> but should **not** be treated as a confirmed diagnosis.
>
> **Always verify with professional diagnostic tools and expertise before taking action.**

---

**Symptoms described:** {context['symptoms']}

**Knowledge Graph Status:** Could not confidently identify a fault
- {context['candidate_count']} partial matches found, but none with sufficient confidence
- Top partial matches: {context['partial_matches']}

---

### General Automotive Guidance

Based on general automotive knowledge, here are common areas to investigate for these symptoms:

**Common causes to check:**
1. **Sensor or electrical issues** — Faulty sensors, loose connections, or wiring problems
2. **Fluid levels and condition** — Low, contaminated, or degraded fluids
3. **Mechanical wear** — Worn components, loose fittings, or degraded seals
4. **Vacuum or pressure issues** — Leaks, blockages, or pressure irregularities
5. **Control module problems** — Software issues, calibration needs, or module faults

**Recommended first steps:**
1. Check for any diagnostic trouble codes (DTCs) with an OBD-II scanner
2. Visually inspect the relevant system for obvious issues (leaks, loose connections, damage)
3. Verify fluid levels and conditions
4. Test basic system functionality before deeper investigation

**What additional information would help:**
- When exactly do the symptoms occur? (cold start, under load, at specific speeds)
- Are there any warning lights illuminated?
- Has any recent work been done on the vehicle?
- Are there any unusual sounds, smells, or vibrations?

---

*This guidance is AI-generated and based on general automotive knowledge.
For a definitive diagnosis, please provide more specific symptoms or consult
a qualified technician with diagnostic equipment.*"""
