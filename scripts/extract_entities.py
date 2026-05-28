import spacy
import json
import re

# Load better spaCy model
nlp = spacy.load("en_core_web_sm")

# Read extracted text
with open("graph/full_text.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Clean text
text = re.sub(r"\s+", " ", text)
text = re.sub(r"[^a-zA-Z0-9.,:/()%\- ]", " ", text)

# Process document
doc = nlp(text)

triples = []

# Valid technical relations
valid_relations = [
    "cause",
    "affect",
    "damage",
    "produce",
    "result",
    "indicate",
    "lead",
    "create",
    "increase",
    "reduce",
    "prevent",
    "block",
    "fail",
    "overheat",
    "leak",
    "burn",
    "corrode",
    "wear",
    "generate"
]

# Ignore useless words
ignore_words = [
    "page",
    "figure",
    "table",
    "chapter",
    "section",
    "system",
    "vehicle"
]

# -----------------------------
# Extract relation triples
# -----------------------------

for sent in doc.sents:

    sentence = sent.text.strip()

    # Skip short/noisy lines
    if len(sentence) < 20:
        continue

    subject = None
    obj = None
    relation = None

    # Find main verb/relation
    for token in sent:

        if token.dep_ == "ROOT":

            lemma = token.lemma_.lower()

            if lemma in valid_relations:
                relation = lemma

    if not relation:
        continue

    # Extract noun chunks
    noun_chunks = list(sent.noun_chunks)

    for chunk in noun_chunks:

        chunk_text = chunk.text.lower().strip()

        # Remove noisy chunks
        if len(chunk_text) < 3:
            continue

        if chunk_text in ignore_words:
            continue

        # Subject
        if chunk.root.dep_ in ["nsubj", "nsubjpass"]:

            subject = chunk_text

        # Object
        if chunk.root.dep_ in ["dobj", "pobj", "attr", "dative"]:

            obj = chunk_text

    # Save triple
    if subject and obj and subject != obj:

        triples.append({
            "subject": subject,
            "relation": relation,
            "object": obj,
            "sentence": sentence
        })

# -----------------------------
# Deduplicate triples
# -----------------------------

seen = set()
unique = []

for t in triples:

    key = (
        t["subject"],
        t["relation"],
        t["object"]
    )

    if key not in seen:

        seen.add(key)
        unique.append(t)

# -----------------------------
# Save triples
# -----------------------------

with open("graph/triples.json", "w", encoding="utf-8") as f:

    json.dump(unique, f, indent=4)

# -----------------------------
# Extract unique entities
# -----------------------------

entities = set()

for t in unique:

    entities.add(t["subject"])
    entities.add(t["object"])

with open("graph/entities.json", "w", encoding="utf-8") as f:

    json.dump(sorted(list(entities)), f, indent=4)

# -----------------------------
# Statistics
# -----------------------------

print(f"\nExtracted {len(unique)} unique triples")
print(f"Extracted {len(entities)} entities")

print("\nSaved files:")
print("graph/triples.json")
print("graph/entities.json")