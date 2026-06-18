import json
import csv

INPUT = "automotive_faults_aktc_obike_et_al.json"
OUTPUT = "triples.csv"

with open(INPUT, encoding="utf-8") as f:
    data = json.load(f)

with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["subject", "predicate", "object"])

    for entry in data:
        system = entry["category"]
        component = entry["subcategory"]
        symptoms = entry["symptoms"]
        diagnosis_steps = entry["diagnosis_steps"]

        w.writerow([system, "HAS_COMPONENT", component])

        for symptom in symptoms:
            w.writerow([component, "SHOWS_SYMPTOM", symptom])

        for ds in diagnosis_steps:
            step = ds["step"]
            w.writerow([component, "DIAGNOSED_BY", step])
            for result in ds["result"]:
                w.writerow([step, "HAS_RESULT", result])

        repair = f"Replace {component}"
        w.writerow([component, "REQUIRES_FIX", repair])

print(f"triples.csv generated")
