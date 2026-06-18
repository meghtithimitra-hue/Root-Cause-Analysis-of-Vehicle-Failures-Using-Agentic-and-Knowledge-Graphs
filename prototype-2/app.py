"""
Automotive GraphRAG CLI — integrates graph retrieval with diagnostic reasoning.

Usage:
    python app.py "Engine Overheating"
"""

import sys
from graph_retriever import retrieve_symptom
from scripts.reasoning import explain_diagnosis


def main():
    if len(sys.argv) < 2:
        print("Usage: python app.py <symptom>")
        sys.exit(1)

    symptom = " ".join(sys.argv[1:])
    retrieval = retrieve_symptom(symptom)
    diagnosis = explain_diagnosis(retrieval)
    print(diagnosis["explanation"])


if __name__ == "__main__":
    main()
