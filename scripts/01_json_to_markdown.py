import json
import re
import os
from collections import defaultdict

JSON_PATH = "automotive_faults_aktc_obike_et_al.json"
OUTPUT_DIR = "data/corpus"

def slugify(name):
    s = name.lower().strip()
    s = re.sub(r'[^a-z0-9]+', '_', s)
    s = s.strip('_')
    return s

def main():
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Group records by category
    groups = defaultdict(list)
    for r in records:
        groups[r["category"]].append(r)

    # Write one file per category
    for cat, items in sorted(groups.items()):
        slug = slugify(cat)
        filepath = os.path.join(OUTPUT_DIR, f"{slug}.md")
        lines = []
        lines.append(f"# {cat}")
        lines.append("")

        # Group by subcategory within this category
        sub_map = defaultdict(list)
        for item in items:
            sub_map[item["subcategory"]].append(item)

        for subcat, sub_items in sorted(sub_map.items()):
            lines.append(f"## {subcat}")
            lines.append("")

            # Explicit linking sentence
            lines.append(f"This subcategory belongs to category {cat}.")
            lines.append("")

            # Symptoms
            lines.append("### Symptoms")
            lines.append("")
            all_symptoms = []
            for item in sub_items:
                for s in item.get("symptoms", []):
                    all_symptoms.append(s)
            # Deduplicate while preserving order
            seen_s = set()
            unique_symptoms = []
            for s in all_symptoms:
                if s not in seen_s:
                    seen_s.add(s)
                    unique_symptoms.append(s)

            for s in unique_symptoms:
                lines.append(f"- Symptom: {s}")
            lines.append("")

            # Explicit symptom linking sentences
            for s in unique_symptoms:
                lines.append(f"{s} is a symptom of {subcat}.")
            lines.append("")

            # Diagnosis Steps
            lines.append("### Diagnosis Steps")
            lines.append("")
            step_num = 1
            for item in sub_items:
                for ds in item.get("diagnosis_steps", []):
                    step = ds["step"]
                    res = ds.get("result", ["", ""])
                    result_a = res[0] if len(res) > 0 else ""
                    result_b = res[1] if len(res) > 1 else ""
                    lines.append(f"{step_num}. Step: {step} -> Result A: {result_a} | Result B: {result_b}")
                    step_num += 1
            lines.append("")

            # Explicit diagnosis step linking sentences
            step_num = 1
            for item in sub_items:
                for ds in item.get("diagnosis_steps", []):
                    step = ds["step"]
                    lines.append(f"{step} is a diagnosis step for {subcat}.")
                    step_num += 1
            lines.append("")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"Wrote {filepath}")

    # Write _index.md
    index_lines = []
    index_lines.append("# Automotive Fault Knowledge Base")
    index_lines.append("")
    for cat in sorted(groups.keys()):
        index_lines.append(f"## {cat}")
        index_lines.append("")
    index_path = os.path.join(OUTPUT_DIR, "_index.md")
    with open(index_path, "w", encoding="utf-8") as f:
        f.write("\n".join(index_lines))
    print(f"Wrote {index_path}")

    # Summary
    print("\n--- Summary ---")
    print(f"Total records: {len(records)}")
    print(f"Categories: {len(groups)}")
    total_steps = sum(len(r.get("diagnosis_steps", [])) for r in records)
    total_symptoms = sum(len(r.get("symptoms", [])) for r in records)
    print(f"Total subcategory entries: {len(records)}")
    print(f"Total diagnosis steps: {total_steps}")
    print(f"Total symptoms: {total_symptoms}")

    for cat, items in sorted(groups.items()):
        subcats = set(r["subcategory"] for r in items)
        steps = sum(len(r.get("diagnosis_steps", [])) for r in items)
        symps = sum(len(r.get("symptoms", [])) for r in items)
        print(f"  {cat}: {len(items)} records, {len(subcats)} subcategories, {symps} symptoms, {steps} steps")

if __name__ == "__main__":
    main()
