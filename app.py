"""
app.py — Streamlit Vehicle Fault Diagnosis Assistant

Three-mode diagnostic assistant:
- AMBIGUOUS: needs more information
- INFERRED: best guess, needs confirmation
- EXTRACTED: high confidence diagnosis

Run with: streamlit run app.py
"""

import sys
import os
from pathlib import Path

# ── Add project root to path ───────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
NUM_PIPELINE_DIR = PROJECT_ROOT / "num_pipeline"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(NUM_PIPELINE_DIR / "scripts") not in sys.path:
    sys.path.insert(0, str(NUM_PIPELINE_DIR / "scripts"))

import streamlit as st
import pandas as pd
from num_pipeline.scripts.decision_engine.engine import DiagnosticReport
from num_pipeline.scripts.run_diagnostic import run_diagnostic


# ─── Page Config ───────────────────────────────────────────────────

st.set_page_config(
    page_title="Vehicle Fault Diagnosis",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── Custom CSS ────────────────────────────────────────────────────

st.markdown("""
<style>
    .stAlert > div {
        padding: 10px 15px;
    }
    .mode-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 0.9em;
    }
    .mode-AMBIGUOUS { background: #fff3cd; color: #856404; }
    .mode-INFERRED { background: #cce5ff; color: #004085; }
    .mode-EXTRACTED { background: #d4edda; color: #155724; }
    .kg-chain {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 4px;
        padding: 4px 0 6px 0;
        font-size: 0.85em;
    }
    .kg-node {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 10px;
        font-weight: 500;
        white-space: nowrap;
    }
    .kg-cat { background: #e8daef; color: #6c3483; }
    .kg-subcat { background: #d6eaf8; color: #2471a3; }
    .kg-symptom { background: #d5f5e3; color: #1e8449; }
    .kg-candidate { background: #fdebd0; color: #ca6f1e; }
    .kg-arrow { color: #aaa; font-size: 0.9em; }
    .badge-top {
        display: inline-block;
        padding: 1px 8px;
        border-radius: 8px;
        font-size: 0.75em;
        font-weight: bold;
        background: #d4edda;
        color: #155724;
    }
    .badge-related {
        display: inline-block;
        padding: 1px 8px;
        border-radius: 8px;
        font-size: 0.75em;
        font-weight: bold;
        background: #e2e3e5;
        color: #383d41;
    }
</style>
""", unsafe_allow_html=True)


# ─── Session State ─────────────────────────────────────────────────

if "report" not in st.session_state:
    st.session_state.report = None
if "original_report" not in st.session_state:
    st.session_state.original_report = None
if "interaction_history" not in st.session_state:
    st.session_state.interaction_history = []
if "show_ai_analysis" not in st.session_state:
    st.session_state.show_ai_analysis = False
if "ai_assisted_text" not in st.session_state:
    st.session_state.ai_assisted_text = ""
if "engine_speed" not in st.session_state:
    st.session_state.engine_speed = 1000


# ─── Main App ──────────────────────────────────────────────────────

def main():
    st.title("Vehicle Fault Diagnosis Assistant")
    st.markdown(
        "Describe the symptoms and I'll help diagnose the fault."
    )

    # ── Input Area ────────────────────────────────────────────────

    st.markdown("**Quick Examples:**")
    example_symptoms = [
        "Brake pedal feels spongy",
        "ABS warning light on, brake pedal pulsation",
        "Engine overheating, coolant loss",
        "Steering pulls to the left",
        "Check engine light, rough idle"
    ]
    example_cols = st.columns(len(example_symptoms))
    for i, example in enumerate(example_symptoms):
        with example_cols[i]:
            if st.button(example, key=f"ex_{i}"):
                st.session_state["symptoms_prefill"] = example
                st.rerun()

    prefill = st.session_state.get("symptoms_prefill", "")
    if prefill:
        st.session_state["symptoms_prefill"] = ""

    # ── Sensor Data (sidebar) ──────────────────────────────────

    with st.sidebar:
        st.markdown("### Sensor Data")
        st.session_state.engine_speed = st.selectbox(
            "Engine speed (RPM)",
            options=[1000, 1200, 1400, 1600],
            index=0,
        )
        sensor_mode = st.radio(
            "Sensor data source",
            ["Simulated (from knowledge graph)", "Upload ECU CSV", "None (skip sensor analysis)"],
            index=0,
        )
        uploaded_csv = None
        if sensor_mode == "Upload ECU CSV":
            uploaded_csv = st.file_uploader("Upload ECU data CSV", type=["csv"])

    with st.form("symptom_form"):
        symptoms_input = st.text_area(
            "Describe the symptoms:",
            value=prefill,
            placeholder="e.g., Brake pedal feels spongy, ABS warning light is on",
            height=100
        )

        analyze_clicked = st.form_submit_button(
            "Analyze Symptoms",
            type="primary",
            use_container_width=True
        )

    # ── Run Diagnosis ─────────────────────────────────────────────

    if analyze_clicked and symptoms_input:
        # Resolve current_sample from sidebar state
        if sensor_mode == "Simulated (from knowledge graph)":
            current_sample = "simulated"
        elif sensor_mode == "Upload ECU CSV" and uploaded_csv is not None:
            current_sample = pd.read_csv(uploaded_csv).iloc[0].to_dict()
        else:
            current_sample = None

        with st.spinner("Analyzing symptoms..."):
            try:
                report = run_diagnostic(
                    symptoms_text=symptoms_input,
                    current_sample=current_sample,
                    speed=st.session_state.engine_speed,
                    verbose=False
                )
                st.session_state.report = report
                st.session_state.original_report = None
                st.session_state.show_ai_analysis = False
                st.session_state.interaction_history.append({
                    "input": symptoms_input,
                    "mode": report.mode,
                    "summary": report.summary,
                })
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                return

    # ── Display Results ───────────────────────────────────────────

    report = st.session_state.report

    if report:
        display_diagnosis(report)

    # ── Display AI-Assisted Analysis if available ─────────────────

    if st.session_state.show_ai_analysis and st.session_state.ai_assisted_text:
        st.markdown("---")
        st.markdown(st.session_state.ai_assisted_text)
        st.markdown("---")
        if st.button("Start New Diagnosis"):
            _clear_session()
            st.rerun()


# ─── Display Diagnosis ─────────────────────────────────────────────

def display_diagnosis(report: DiagnosticReport):
    """Render the diagnosis report.

    Order: Mode badge → Symptoms → Diagnosis summary → Explanation →
    Inspection steps → Candidates → Developer Details → Follow-up.
    """
    mode = report.mode
    top = report.top_candidate

    # ── Mode Badge ────────────────────────────────────────────────

    mode_labels = {
        "AMBIGUOUS": "NEED MORE INFO",
        "INFERRED": "BEST GUESS",
        "EXTRACTED": "DIAGNOSIS",
    }

    st.markdown("---")
    mode_badge = f'<span class="mode-badge mode-{mode}">{mode}</span>'
    st.markdown(
        f"### {mode_labels.get(mode, mode)} {mode_badge}",
        unsafe_allow_html=True,
    )
    st.caption(report.summary)

    # ── Original symptoms ─────────────────────────────────────────
    if report.original_symptoms:
        pills = " · ".join(f"`{s}`" for s in report.original_symptoms[:5])
        st.markdown(f"**Symptoms:** {pills}")

    # ── Plain-English diagnosis summary (from engine) ─────────────
    if report.diagnosis_summary:
        st.markdown(report.diagnosis_summary)

    # ── Explanation (from engine) ─────────────────────────────────
    if report.explanation:
        st.markdown(report.explanation)

    # ── Recommended inspection steps (from engine) ────────────────
    if report.inspection_steps:
        st.markdown("**Recommended inspection steps:**")
        for step in report.inspection_steps:
            st.markdown(f"- {step}")

    # ── Candidates ────────────────────────────────────────────────
    _display_candidates(report)

    # ── Developer Details (collapsed) ─────────────────────────────
    with st.expander("Developer Details", expanded=False):
        if report.reasoning_chain:
            st.markdown("**Reasoning chain:**")
            for i, step in enumerate(report.reasoning_chain, 1):
                st.markdown(f"**{i}. {step['step']}**")
                st.write(step["detail"])
                metrics = step.get("metrics", {})
                if metrics:
                    metric_items = []
                    for k, v in metrics.items():
                        if k == "provenance":
                            continue
                        if isinstance(v, float):
                            metric_items.append(f"{k}: {v:.3f}")
                        elif isinstance(v, dict) and len(str(v)) < 100:
                            metric_items.append(f"{k}: {v}")
                        elif isinstance(v, (str, int)):
                            metric_items.append(f"{k}: {v}")
                    if metric_items:
                        st.caption(" · ".join(metric_items))

        # ── Sensor Validation Debug (temporary) ──────────────────
        sd = getattr(report, "sensor_debug", {})
        if sd:
            st.markdown("**Step 6: Sensor Validation (debug)**")
            ran = sd.get("ran", False)
            input_type = sd.get("input_type", "none")
            st.markdown(f"- Ran: **{'Yes' if ran else 'No'}**")
            st.markdown(f"- Source: **{input_type}**")
            st.markdown(f"- Speed: **{sd.get('speed', '')}** RPM")
            nf = sd.get("navic_fault", "")
            if nf:
                st.markdown(f"- Top navic_fault: **{nf}**")
            csv = sd.get("csv_path", "")
            if csv:
                st.markdown(f"- CSV loaded: **{csv}**")
            cols = sd.get("sample_columns", [])
            if cols:
                st.markdown(f"- Sample columns: `{cols}`")

    # ── Mode-specific follow-up ───────────────────────────────────
    st.markdown("---")
    if mode == "AMBIGUOUS":
        _display_ambiguous_followup(report)
    elif mode == "INFERRED":
        _display_inferred_followup(report)
    elif mode == "EXTRACTED":
        _display_extracted_followup(report)


# ─── KG Context Chain ─────────────────────────────────────────────

def _build_kg_chain(candidate: dict) -> str:
    """Build a colored KG context chain as HTML node pills.

    Node colors match the knowledge graph palette:
    - Category:     purple
    - Subcategory:  blue
    - Symptom:      green
    - Candidate:    orange
    """
    label = candidate.get("label", "")
    subcategory = candidate.get("subcategory", "")
    category = candidate.get("category", "")

    nodes = []
    if category:
        nodes.append(("kg-cat", category))
    if subcategory and subcategory != category:
        nodes.append(("kg-subcat", subcategory))
    if label:
        nodes.append(("kg-candidate", label))

    if len(nodes) < 2:
        return ""

    pills = []
    for i, (cls, text) in enumerate(nodes):
        if i > 0:
            pills.append('<span class="kg-arrow">→</span>')
        pills.append(f'<span class="kg-node {cls}">{text}</span>')

    return f'<div class="kg-chain">{"".join(pills)}</div>'


# ─── Candidate Display ─────────────────────────────────────────────

def _display_candidates(report: DiagnosticReport):
    """Display all candidates in the mode's threshold band."""
    candidates = report.display_candidates
    if not candidates:
        st.info("No candidates found.")
        return

    has_sensor = any(
        ev.get("status") not in (None, "No Evidence")
        for ev in report.sensor_evidence.values()
    ) if report.sensor_evidence else False

    for i, c in enumerate(candidates):
        label = c.get("label", "Unknown")
        sensor_status = c.get("sensor_status", "No Evidence")

        # Rank and badge
        rank = f"#{i+1}"
        badge_html = (
            '<span class="badge-top">Top Match</span>' if i == 0
            else '<span class="badge-related">Related</span>'
        )

        # Sensor badge (only when sensor data was provided)
        sensor_badge = _format_sensor_badge(sensor_status) if has_sensor else ""

        # Candidate header
        st.markdown(
            f"**{rank}** {label} {badge_html} {sensor_badge}",
            unsafe_allow_html=True,
        )

        # KG context chain
        chain_html = _build_kg_chain(c)
        if chain_html:
            st.markdown(chain_html, unsafe_allow_html=True)

        # Expandable sensor detail
        fault = c.get("navic_fault", "")
        if has_sensor and fault and fault in report.sensor_evidence:
            se = report.sensor_evidence[fault]
            if se.get("critical") or se.get("warning"):
                with st.expander(f"Sensor detail — {label}", expanded=False):
                    for s in se.get("critical", []):
                        st.markdown(f"🔴 **{s}**: CRITICAL")
                    for s in se.get("warning", []):
                        st.markdown(f"🟡 **{s}**: WARNING")
                    for s in se.get("normal", []):
                        st.markdown(f"⚪ {s}: Normal")

    if not has_sensor:
        st.info("No sensor data provided — sensor validation skipped.")

# ─── Sensor Badge Formatting ───────────────────────────────────────

def _format_sensor_badge(status: str) -> str:
    """Format sensor status as an inline badge."""
    badges = {
        "Supported": "🟢 Sensor Supported",
        "Contradicted": "🟡 Sensor Contradicted",
        "No Evidence": "⚪ No Sensor Data",
    }
    return badges.get(status, "")


# ─── AMBIGUOUS Follow-up ──────────────────────────────────────────

def _display_ambiguous_followup(report: DiagnosticReport):
    """Clarification form for AMBIGUOUS mode."""
    st.markdown("### Clarify Symptoms")
    st.markdown(
        "Please provide more details about the issue."
    )

    with st.form("clarify_form"):
        clarification = st.text_area(
            "Additional details:",
            placeholder="e.g., The issue happens at highway speed, error code P0301",
            height=80,
        )
        col1, col2 = st.columns(2)
        with col1:
            submit_clarify = st.form_submit_button(
                "Submit Clarification",
                type="primary",
                use_container_width=True,
            )
        with col2:
            pass  # empty — skip button is outside form

    # Skip button (outside form, per Streamlit API rules)
    if st.button("Skip — Get AI Analysis", use_container_width=True):
        _show_ai_analysis(report)

    if submit_clarify and clarification:
        combined = f"{report.query_text}, {clarification}"
        with st.spinner("Re-analyzing with additional details..."):
            try:
                new_report = run_diagnostic(
                    symptoms_text=combined,
                    current_sample="simulated",
                    speed=st.session_state.engine_speed,
                    verbose=False,
                )
                st.session_state.report = new_report
                st.session_state.original_report = None
                st.session_state.interaction_history.append({
                    "input": combined,
                    "mode": new_report.mode,
                    "summary": new_report.summary,
                })
                st.rerun()
            except Exception as e:
                st.error(f"Re-analysis failed: {e}")


# ─── INFERRED Follow-up ───────────────────────────────────────────

def _display_inferred_followup(report: DiagnosticReport):
    """Refinement form for INFERRED mode."""
    st.markdown("### Refine Diagnosis")
    st.markdown(
        "Provide additional symptoms to narrow the diagnosis, "
        "or use the current best guess as-is."
    )

    # Build checkbox options from the top candidate's neighborhood
    # (placeholder — in a real scenario, these would come from the graph)
    common_symptoms = [
        "Brake noise when stopping",
        "Vibration during braking",
        "Vehicle pulls to one side",
        "Dashboard warning light",
        "Issue worsens at speed",
    ]

    with st.form("refine_form"):
        st.markdown("**Additional symptoms (check all that apply):**")
        checked = []
        for sym in common_symptoms:
            if st.checkbox(sym, key=f"refine_{sym}"):
                checked.append(sym)

        extra = st.text_area(
            "Other symptoms (optional):",
            placeholder="Describe any other symptoms",
            height=60,
        )
        refine_submit = st.form_submit_button(
            "Refine Diagnosis",
            type="primary",
            use_container_width=True,
        )

    # Skip button (outside form)
    if st.button("Use Current Diagnosis", use_container_width=True):
        st.session_state.original_report = None
        st.rerun()

    if refine_submit:
        new_symptoms = list(checked)
        if extra:
            new_symptoms.append(extra)

        if not new_symptoms:
            st.info("No additional symptoms selected. Current diagnosis unchanged.")
            return

        combined = f"{report.query_text}, {', '.join(new_symptoms)}"
        with st.spinner("Refining diagnosis..."):
            try:
                original_conf = report.confidence
                new_report = run_diagnostic(
                    symptoms_text=combined,
                    current_sample="simulated",
                    speed=st.session_state.engine_speed,
                    verbose=False,
                )

                if new_report.confidence > original_conf:
                    # Improved — show updated result
                    st.session_state.report = new_report
                    st.session_state.original_report = None
                else:
                    # Not improved — keep original, show message
                    st.session_state.original_report = report
                    st.info(
                        "The additional symptoms did not strengthen the "
                        "diagnosis. Your original assessment is shown below."
                    )

                st.session_state.interaction_history.append({
                    "input": combined,
                    "mode": new_report.mode,
                    "summary": new_report.summary,
                })
                st.rerun()
            except Exception as e:
                st.error(f"Refinement failed: {e}")


# ─── EXTRACTED Follow-up ──────────────────────────────────────────

def _display_extracted_followup(report: DiagnosticReport):
    """Post-diagnosis actions for EXTRACTED mode."""
    st.markdown("**Diagnosis complete.**")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start New Diagnosis"):
            _clear_session()
            st.rerun()
    with col2:
        if st.button("View Full Report"):
            with st.expander("Full Report", expanded=True):
                st.json({
                    "mode": report.mode,
                    "confidence": report.confidence,
                    "confidence_components": report.confidence_components,
                    "top_candidate": report.top_candidate,
                    "display_candidates": [
                        {
                            "label": c.get("label"),
                            "confidence": c.get("confidence"),
                            "navic_fault": c.get("navic_fault"),
                            "mapping_type": c.get("mapping_type"),
                            "sensor_status": c.get("sensor_status"),
                        }
                        for c in report.display_candidates
                    ],
                    "sensor_evidence": report.sensor_evidence,
                    "original_symptoms": report.original_symptoms,
                    "query_text": report.query_text,
                })


# ─── AI-Assisted Analysis ─────────────────────────────────────────

def _show_ai_analysis(report: DiagnosticReport):
    """Generate and display AI-assisted analysis."""
    with st.spinner("Generating AI-Assisted Analysis..."):
        from num_pipeline.scripts.decision_engine.explanation import (
            generate_ai_assisted_analysis,
        )
        from num_pipeline.scripts.pipeline.llm_provider import (
            get_llm_provider,
        )

        llm_provider = get_llm_provider()
        text = generate_ai_assisted_analysis(
            symptoms=report.query_text,
            candidates=report.display_candidates,
            llm_provider=llm_provider,
        )
        st.session_state.ai_assisted_text = text
        st.session_state.show_ai_analysis = True
        st.rerun()


# ─── Interaction History ───────────────────────────────────────────

def display_history():
    """Display interaction history in sidebar."""
    with st.sidebar:
        st.markdown("### History")
        if st.session_state.interaction_history:
            for i, entry in enumerate(
                reversed(st.session_state.interaction_history), 1
            ):
                st.markdown(
                    f"**{i}.** {entry['mode']}: {entry['summary']}"
                )
        else:
            st.write("No diagnoses yet.")


# ─── Helpers ───────────────────────────────────────────────────────

def _clear_session():
    """Reset all session state."""
    st.session_state.report = None
    st.session_state.original_report = None
    st.session_state.show_ai_analysis = False
    st.session_state.ai_assisted_text = ""


if __name__ == "__main__":
    display_history()
    main()
