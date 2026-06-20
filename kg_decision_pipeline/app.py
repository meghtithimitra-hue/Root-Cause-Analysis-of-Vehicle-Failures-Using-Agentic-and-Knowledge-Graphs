"""Streamlit UI for the KG decision pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
from kg_decision_pipeline.pipeline import run_pipeline

MODE_COLORS = {"EXTRACTED": "green", "INFERRED": "orange", "AMBIGUOUS": "red"}


def _run(query: str, skip: bool) -> dict:
    with st.spinner("Running pipeline..."):
        return run_pipeline(query, skip_clarification=skip)


def main():
    st.set_page_config(page_title="KG Decision Pipeline", layout="wide")
    st.title("🔧 Root-Cause Analysis — KG Decision Pipeline")

    if "result" not in st.session_state:
        st.session_state.result = None
    if "skip" not in st.session_state:
        st.session_state.skip = False

    query = st.text_input(
        "Describe the vehicle fault",
        placeholder="e.g. brake pedal feels spongy when pressed",
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        submitted = st.button("Submit", type="primary", use_container_width=True)
    with col2:
        pass

    if submitted and query:
        st.session_state.result = _run(query, skip=False)
        st.session_state.skip = False

    result = st.session_state.result
    if result is None:
        st.info("Enter a fault description and press Submit to start.")
        return

    mode = result.get("mode", "AMBIGUOUS")
    color = MODE_COLORS.get(mode, "red")
    st.markdown(f"## Mode: :{color}[{mode}]")

    if mode == "AMBIGUOUS":
        cq = result.get("clarifying_question") or result.get("answer", "")
        st.warning(cq)

        follow_up = st.text_input(
            "Your answer",
            placeholder="e.g. the noise happens when turning left",
            key="follow_up_input",
        )

        if st.button("Submit follow-up", key="follow_up_btn") and follow_up:
            st.session_state.result = _run(
                f"{result.get('query', '')} — {follow_up}", skip=False,
            )
            st.rerun()

        if st.button("Skip — use LLM fallback", key="skip_btn"):
            st.session_state.result = _run(
                result.get("query", ""), skip=True,
            )
            st.rerun()

    else:
        cat = result.get("top_category")
        sub = result.get("top_subcategory")
        if cat or sub:
            c1, c2 = st.columns(2)
            if cat:
                c1.metric("System", cat)
            if sub:
                c2.metric("Subsystem", sub)

        symptoms = result.get("matched_symptoms", [])
        if symptoms:
            st.subheader("Matched Symptoms")
            for s in symptoms:
                st.markdown(f"- {s}")

        steps = result.get("diagnosis_steps", [])
        if steps:
            st.subheader("Diagnosis Steps")
            for i, s in enumerate(steps, 1):
                st.markdown(f"{i}. {s}")

        chain = result.get("reasoning_chain", [])
        if chain:
            st.subheader("Reasoning Chain")
            for i, c in enumerate(chain, 1):
                st.markdown(f"{i}. {c}")

    with st.expander("Raw answer text"):
        st.text(result.get("answer", ""))


if __name__ == "__main__":
    main()
