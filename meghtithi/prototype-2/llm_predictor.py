from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

# To use Ollama instead, comment out Groq lines and uncomment Ollama block below.

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are an expert automotive diagnostics AI.
You are given a structured knowledge graph subgraph showing:
  Symptom → Failure Mode → Component → Repair Action → Confidence

Your job:
1. Rank the most likely failure causes based on confidence and symptom match
2. Explain WHY each cause is likely (briefly)
3. Suggest the repair actions in priority order
4. If multiple symptoms point to the same root cause, highlight that

Rules:
- Only use information from the provided subgraph
- Do NOT invent failure modes not present in the graph
- Format your response clearly with numbered causes
- Include confidence percentages
"""

def predict_failures(user_query: str, subgraph_context: str) -> str:
    """Send subgraph + user query to LLM and return prediction."""

    user_message = f"""User query: "{user_query}"

{subgraph_context}

Based on the knowledge graph above, provide your automotive failure prediction."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message}
        ],
        temperature=0.2,   # low temp = more deterministic for diagnostics
        max_tokens=800
    )

    return response.choices[0].message.content


# ── Ollama alternative (uncomment to use instead of Groq) ──────────────────
# import requests, json
#
# def predict_failures(user_query: str, subgraph_context: str) -> str:
#     user_message = f'User query: "{user_query}"\n\n{subgraph_context}\n\nProvide your automotive failure prediction.'
#     payload = {
#         "model": "mistral",     # run: ollama pull mistral
#         "messages": [
#             {"role": "system", "content": SYSTEM_PROMPT},
#             {"role": "user",   "content": user_message}
#         ],
#         "stream": False
#     }
#     response = requests.post("http://localhost:11434/api/chat", json=payload)
#     return response.json()["message"]["content"]
