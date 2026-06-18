from scripts.reasoning import search_graph
from langchain_ollama import OllamaLLM

# Load local Llama3 model
llm = OllamaLLM(model="llama3")


def graph_rag(query):

    # Retrieve graph relationships
    results = search_graph(query)

    if not results:
        return "No relevant relationships found in knowledge graph."

    # Build context from graph
    context = ""

    for r in results:

        context += (
            f"{r['subject']} "
            f"{r['relation']} "
            f"{r['object']}\n"
        )

    # Prompt for LLM
    prompt = f"""
You are an expert automotive diagnostic AI.

Use ONLY the following knowledge graph context:

{context}

Diagnose the vehicle issue:

{query}

Explain:

1. Likely causes
2. Severity
3. Recommended fixes
4. Preventive maintenance

Keep the response professional and structured.
"""

    # Generate AI response
    response = llm.invoke(prompt)

    return response