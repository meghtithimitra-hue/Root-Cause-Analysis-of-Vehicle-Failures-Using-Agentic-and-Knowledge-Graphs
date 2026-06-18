"""
Streamlit automotive diagnostic app.

Usage:
    streamlit run app_streamlit.py
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from graph_retriever import retrieve_symptom
from scripts.reasoning import explain_diagnosis

st.set_page_config(
    page_title="Auto GraphRAG Diagnosis",
    page_icon=":car:",
    layout="centered",
)

st.markdown("""
    <style>
    .stApp { max-width: 800px; margin: 0 auto; }
    .report-box { padding: 1.2rem 1.5rem; border-radius: 12px; margin: 1rem 0; }
    .symptom { background: #eef2ff; border-left: 4px solid #4f46e5; }
    .cause   { background: #fef2f2; border-left: 4px solid #dc2626; }
    .repair  { background: #f0fdf4; border-left: 4px solid #16a34a; }
    .conf    { background: #faf5ff; border-left: 4px solid #9333ea; }
    .path-box { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 0.8rem 1rem; margin: 0.5rem 0; font-size: 0.9rem; }
    .score-tag { display: inline-block; background: #e2e8f0; border-radius: 6px; padding: 0.1rem 0.5rem; font-size: 0.75rem; font-weight: 600; margin-right: 0.5rem; }
    .arrow { color: #94a3b8; margin: 0 0.3rem; }
    </style>
""", unsafe_allow_html=True)

st.title(":car: Auto GraphRAG Diagnosis")
st.markdown("Describe the vehicle symptom below to get a ranked diagnostic report.")

symptom = st.text_input(
    "Vehicle symptom",
    placeholder="e.g. Engine Overheating, Brake pedal pulsation",
    label_visibility="collapsed",
)

if not symptom:
    st.info("Enter a symptom to begin diagnosis.")
    st.stop()

retrieval = None
diagnosis = None
error = None

with st.spinner("Retrieving from knowledge graph..."):
    try:
        retrieval = retrieve_symptom(symptom)
    except ValueError as e:
        error = str(e)
    except Exception as e:
        error = f"Graph retrieval failed: {e}"

if error:
    st.error(error)
    st.stop()

with st.spinner("Analyzing reasoning paths..."):
    try:
        diagnosis = explain_diagnosis(retrieval)
    except Exception as e:
        st.error(f"Reasoning failed: {e}")
        st.stop()

if not diagnosis.get("top_reasoning_paths"):
    st.warning(f"No diagnostic data found for **{symptom}**.")
    st.stop()

st.divider()
st.markdown("### Diagnosis Report")

cols = st.columns([1, 1, 1, 1])

with cols[0]:
    st.markdown(
        f'<div class="report-box symptom">'
        f'<div style="font-size:0.75rem;color:#4f46e5;font-weight:600;">SYMPTOM</div>'
        f'<div style="font-weight:600;">{diagnosis["symptom"]}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

with cols[1]:
    st.markdown(
        f'<div class="report-box cause">'
        f'<div style="font-size:0.75rem;color:#dc2626;font-weight:600;">MOST LIKELY CAUSE</div>'
        f'<div style="font-weight:600;">{diagnosis["most_likely_cause"]}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

with cols[2]:
    st.markdown(
        f'<div class="report-box repair">'
        f'<div style="font-size:0.75rem;color:#16a34a;font-weight:600;">RECOMMENDED REPAIR</div>'
        f'<div style="font-weight:600;">{diagnosis["recommended_repair"]}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

with cols[3]:
    conf = diagnosis.get("confidence_score", 0)
    st.markdown(
        f'<div class="report-box conf">'
        f'<div style="font-size:0.75rem;color:#9333ea;font-weight:600;">CONFIDENCE</div>'
        f'<div style="font-weight:600;font-size:1.3rem;">{conf:.0f}%</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

st.markdown("#### Reasoning Paths")
for sp in diagnosis["top_reasoning_paths"]:
    parts = " ".join(
        f'<span style="font-weight:500;">{p}</span>'
        if i == len(sp.path) - 1
        else f"<span>{p}</span>"
        for i, p in enumerate(sp.path)
    )
    arrows = f'<span class="arrow">→</span>'.join(
        f'<span>{p}</span>' for p in sp.path
    )
    st.markdown(
        f'<div class="path-box">'
        f'<span class="score-tag">{sp.score}</span>'
        f'{arrows}'
        f'</div>',
        unsafe_allow_html=True,
    )

if retrieval.get("diagnostic_tests"):
    with st.expander("All Diagnostic Tests"):
        for t in retrieval["diagnostic_tests"]:
            st.markdown(f"- {t}")

if retrieval.get("results"):
    with st.expander("All Possible Results"):
        for r in retrieval["results"]:
            st.markdown(f"- {r}")

if retrieval.get("repair_actions"):
    with st.expander("All Repair Actions"):
        for r in retrieval["repair_actions"]:
            st.markdown(f"- {r}")

st.divider()
st.caption("Powered by Neo4j knowledge graph & GraphRAG reasoning")
