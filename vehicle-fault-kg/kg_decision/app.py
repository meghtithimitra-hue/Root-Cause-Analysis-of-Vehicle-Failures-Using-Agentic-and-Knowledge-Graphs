"""Streamlit UI for the KG decision pipeline (vehicle-fault-kg)."""

import importlib
import os
import sys
from pathlib import Path

# Ensure working directory is vehicle-fault-kg/ so relative paths in
# the retrieval layer resolve correctly.
_HERE = Path(__file__).resolve().parent
_VFK_ROOT = _HERE.parent
os.chdir(str(_VFK_ROOT))

sys.path.insert(0, str(_VFK_ROOT))

import streamlit as st

_pipeline = importlib.import_module("kg_decision.pipeline")
_answer_gen = importlib.import_module("kg_decision.04_answer_generator")

run_pipeline = _pipeline.run_pipeline
format_graph_answer = _answer_gen.format_graph_answer

MODE_COLORS = {"EXTRACTED": "green", "INFERRED": "orange", "AMBIGUOUS": "red"}


def _run(query: str, skip: bool) -> dict:
    with st.spinner("Running pipeline..."):
        return run_pipeline(query, skip_clarification=skip)


def main():
    st.set_page_config(page_title="KG Decision Pipeline — vehicle-fault-kg", layout="wide")
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

        if st.button("Skip", key="skip_btn"):
            st.session_state.result = _run(
                result.get("query", ""), skip=True,
            )
            st.rerun()

    elif mode == "INFERRED" and result.get("is_intermediate"):
        cat = result.get("top_category")
        sub = result.get("top_subcategory")
        if cat or sub:
            c1, c2 = st.columns(2)
            if cat:
                c1.metric("System", cat)
            if sub:
                c2.metric("Subsystem", sub)

        unconfirmed = result.get("unconfirmed_symptoms", [])
        st.info(
            "The system matched your issue to this subsystem. "
            "Select any additional symptoms that apply to help "
            "refine the diagnosis:"
        )

        checked = {}
        for sym in unconfirmed:
            checked[sym] = st.checkbox(sym, key=f"unconf_{sym}")

        extra = st.text_input(
            "Additional symptoms (optional)",
            placeholder="e.g. vibration at highway speeds",
            key="inferred_extra",
        )

        col_a, col_b = st.columns(2)
        with col_a:
            selected = [s for s in unconfirmed if checked.get(s)]
            extra_parts = [f" — {', '.join(selected)}"] if selected else []
            if extra.strip():
                extra_parts.append(f" — {extra.strip()}")
            enriched = result.get("query", "") + "".join(extra_parts)
            if st.button("Refine with selected symptoms",
                         type="primary", use_container_width=True) and enriched:
                st.session_state.result = _run(enriched, skip=False)
                st.rerun()
        with col_b:
            if st.button("Skip — show diagnosis",
                         use_container_width=True):
                result["answer"] = format_graph_answer(result)
                result["is_intermediate"] = False
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
