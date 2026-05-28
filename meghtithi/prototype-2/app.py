import streamlit as st
from graph_retriever import GraphRetriever
from llm_predictor import predict_failures

st.set_page_config(
    page_title="Automotive Failure Predictor",
    page_icon="🔧",
    layout="wide"
)

st.title("🔧 Automotive Failure Prediction")
st.caption("Graph RAG — powered by Neo4j + Groq LLM")

@st.cache_resource
def get_retriever():
    return GraphRetriever()

retriever = get_retriever()

# ── Input ──────────────────────────────────────────────────────────────────
col1, col2 = st.columns([3, 1])
with col1:
    user_query = st.text_input(
        "Describe the symptom:",
        placeholder="e.g. engine won't start, battery drains overnight"
    )
with col2:
    st.write("")
    st.write("")
    analyze = st.button("🔍 Analyze", use_container_width=True)

# ── Quick examples ─────────────────────────────────────────────────────────
st.write("Try:")
example_cols = st.columns(4)
examples = [
    "engine won't start",
    "engine misfires",
    "check engine light",
    "engine overheats"
]
for i, ex in enumerate(examples):
    if example_cols[i].button(ex, key=f"ex_{i}"):
        user_query = ex
        analyze = True

# ── Analysis ───────────────────────────────────────────────────────────────
if analyze and user_query:
    with st.spinner("Traversing knowledge graph..."):

        # Step 1: extract entities
        matched_symptoms = retriever.extract_entities(user_query)

        # Step 2: traverse graph
        subgraph = retriever.retrieve_subgraph(matched_symptoms)

        # Step 3: serialize for LLM
        subgraph_context = retriever.serialize_subgraph_for_prompt(subgraph)

    # Show graph data in expander
    with st.expander("📊 Retrieved knowledge subgraph", expanded=True):
        if not matched_symptoms:
            st.warning("No matching symptoms found in the knowledge graph.")
        else:
            st.write(f"**Matched symptoms:** {', '.join(matched_symptoms)}")
            st.code(subgraph_context, language="text")

    # Step 4: LLM prediction
    if subgraph["paths"]:
        with st.spinner("LLM reasoning over subgraph..."):
            prediction = predict_failures(user_query, subgraph_context)

        st.subheader("🧠 Failure Prediction")
        st.markdown(prediction)

        # Show ranked table
        st.subheader("📋 Graph traversal summary")
        import pandas as pd
        df = pd.DataFrame(subgraph["paths"])
        df["confidence"] = df["confidence"].apply(lambda x: f"{x:.0%}")
        df.columns = ["Symptom", "Failure Mode", "Component", "Repair Action", "Confidence"]
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.warning("Graph found no paths for this symptom. Try adding it to the knowledge graph.")

elif analyze and not user_query:
    st.error("Please enter a symptom first.")

    st.divider()
st.subheader("🔬 Knowledge graph tools")
col_a, col_b = st.columns(2)

with col_a:
    if st.button("🕸️ Visualize graph in browser", use_container_width=True):
        import subprocess
        subprocess.Popen(["python", "visualize_graph.py"])
        st.success("Opening graph visualizer in your browser...")

with col_b:
    uploaded_pdf = st.file_uploader("📄 Ingest new PDF into graph", type="pdf")
    if uploaded_pdf and st.button("⚙️ Process PDF", use_container_width=True):
        import tempfile
        import pdf_to_graph
        from importlib import reload
        reload(pdf_to_graph)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_pdf.read())
            tmp_path = tmp.name
        with st.spinner(f"Extracting knowledge from {uploaded_pdf.name}..."):
            pdf_to_graph.process_pdf(tmp_path)
        st.cache_resource.clear()
        st.success(f"✅ Graph updated from {uploaded_pdf.name}! Refresh to use new knowledge.")
