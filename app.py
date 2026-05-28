import streamlit as st
import streamlit.components.v1 as components

from scripts.graphrag import graph_rag

st.set_page_config(
    page_title="Vehicle GraphRAG AI",
    layout="wide"
)

st.title("🚗 Vehicle GraphRAG Diagnostic AI")

st.markdown(
    "Knowledge Graph + Llama3 + Graph Retrieval"
)

query = st.text_input(
    "Describe vehicle issue:"
)

if st.button("Analyze Vehicle"):

    with st.spinner("Analyzing graph relationships..."):

        response = graph_rag(query)

    st.subheader("AI Diagnosis")

    st.write(response)

    st.subheader("Hierarchical Knowledge Graph")

    HtmlFile = open(
        "graph/hierarchical_graph.html",
        "r",
        encoding="utf-8"
    )

    source_code = HtmlFile.read()

    components.html(
        source_code,
        height=900
    )