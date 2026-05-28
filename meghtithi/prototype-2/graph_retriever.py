from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

load_dotenv()

class GraphRetriever:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
        )

    def close(self):
        self.driver.close()

    def extract_entities(self, query: str) -> list:
        # Synonym map — user phrasing → graph node keywords
        SYNONYMS = {
            "tyre":       "tire",
            "blowout":    "tire failure",
            "puncture":   "tire failure",
            "flat tyre":  "tire failure",
            "flat tire":  "tire failure",
            "vibration":  "shock absorber",
            "bouncing":   "shock absorber",
            "judder":     "clutch",
            "slipping":   "clutch",
            "spongy":     "master cylinder",
            "soft pedal": "master cylinder",
            "belt snap":  "drive belt",
            "misfiring":  "spark plug",
            "rough idle": "spark plug",
        }

        # Apply synonyms to query before matching
        query_lower = query.lower()
        for user_term, graph_term in SYNONYMS.items():
            if user_term in query_lower:
                query_lower = query_lower + " " + graph_term

        with self.driver.session() as session:
            result = session.run("MATCH (s:Symptom) RETURN s.name AS name")
            symptoms = [r["name"] for r in result]

        matched = []
        for symptom in symptoms:
            symptom_lower = symptom.lower()
            symptom_words = set(symptom_lower.split())
            query_words = set(query_lower.split())

            # Exact substring match
            if symptom_lower in query_lower or query_lower in symptom_lower:
                matched.append(symptom)
                continue

            # Word overlap match
            if symptom_words & query_words:
                matched.append(symptom)
                continue

            # Fuzzy partial match — catches "tyre blowout" → "tire failure"
            for s_word in symptom_words:
                for q_word in query_words:
                    if len(s_word) > 4 and len(q_word) > 4:
                        if s_word in q_word or q_word in s_word:
                            matched.append(symptom)
                            break

        return list(set(matched))

    def retrieve_subgraph(self, symptoms: list) -> dict:
        if not symptoms:
            return {"symptoms": [], "paths": []}

        with self.driver.session() as session:
            result = session.run("""
                MATCH (s:Symptom)-[i:INDICATES]->(f:FailureMode)
                      -[:AFFECTS]->(c:Component)
                      -[:REQUIRES]->(r:RepairAction)
                WHERE s.name IN $symptoms
                RETURN s.name AS symptom,
                       f.name AS failure_mode,
                       c.name AS component,
                       r.name AS repair_action,
                       i.confidence AS confidence
                ORDER BY i.confidence DESC
            """, symptoms=symptoms)

            paths = []
            for record in result:
                paths.append({
                    "symptom":      record["symptom"],
                    "failure_mode": record["failure_mode"],
                    "component":    record["component"],
                    "repair_action":record["repair_action"],
                    "confidence":   record["confidence"]
                })

        return {
            "matched_symptoms": symptoms,
            "paths": paths
        }

    def serialize_subgraph_for_prompt(self, subgraph: dict) -> str:
        if not subgraph["paths"]:
            return "No relevant subgraph found for this query."

        lines = ["KNOWLEDGE GRAPH SUBGRAPH (traversed relationships):"]
        lines.append(f"Matched symptoms: {', '.join(subgraph['matched_symptoms'])}")
        lines.append("")
        lines.append("Symptom → Failure Mode → Component → Repair Action [confidence]")
        lines.append("-" * 65)

        for p in subgraph["paths"]:
            lines.append(
                f"  {p['symptom']} "
                f"→ [{p['failure_mode']}] "
                f"→ {p['component']} "
                f"→ {p['repair_action']} "
                f"[{p['confidence']:.0%}]"
            )

        return "\n".join(lines)
