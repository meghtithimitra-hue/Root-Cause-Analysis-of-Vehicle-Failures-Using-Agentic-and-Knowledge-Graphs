import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Automotive KG Q&A",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Automotive Document Q&A")
st.caption("Powered by Hierarchical Knowledge Graph + Groq (LLaMA3-70B)")

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("📄 Upload Your PDF")
    uploaded = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded and st.button("🚀 Ingest PDF"):
        with st.spinner("Building knowledge graph... this takes 2-4 mins"):
            resp = requests.post(
                f"{API_URL}/upload-pdf",
                files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}
            )
            if resp.status_code == 200:
                data = resp.json()
                st.success(f"✅ Done! {data['chunks']} chunks indexed.")
            else:
                st.error(f"Error: {resp.text}")

    st.markdown("---")
    st.markdown("**Stack:**")
    st.markdown("- 🧠 LLM: Groq LLaMA3-70B")
    st.markdown("- 🕸️ Graph: Neo4j")
    st.markdown("- 🔍 Vectors: ChromaDB")
    st.markdown("- ⚡ API: FastAPI")

# ── Chat ──────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("pages"):
            st.caption(f"📖 Source pages: {msg['pages']}")

if question := st.chat_input("Ask anything about the automotive document..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge graph..."):
            resp = requests.post(
                f"{API_URL}/query",
                json={"question": question, "top_k": 8}
            )
            if resp.status_code == 200:
                data = resp.json()
                st.markdown(data["answer"])
                st.caption(f"📖 Pages: {data['source_pages']} | Chunks used: {data['chunks_used']}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": data["answer"],
                    "pages": data["source_pages"]
                })
            else:
                st.error("Something went wrong. Is the backend running?")
