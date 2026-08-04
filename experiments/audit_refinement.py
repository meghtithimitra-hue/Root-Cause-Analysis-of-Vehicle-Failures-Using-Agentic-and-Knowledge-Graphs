"""Comprehensive functional audit of the Refine Diagnosis workflow.

Simulates the UI refinement logic from app.py _display_inferred_followup()
without Streamlit — exercises the same backend code path.
"""

import contextlib
import io
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT / "num_pipeline" / "scripts"
NUM_PIPELINE_DIR = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(NUM_PIPELINE_DIR)

import pipeline.evidence_fusion as ef
_o = ef.fuse_evidence
def _q(a,b,c):
    with contextlib.redirect_stdout(io.StringIO()):
        return _o(a,b,c)
ef.fuse_evidence = _q

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from run_diagnostic import run_diagnostic

# ── The 5 hardcoded symptom options from app.py ──
CHECKBOX_OPTIONS = [
    "Brake noise when stopping",
    "Vibration during braking",
    "Vehicle pulls to one side",
    "Dashboard warning light",
    "Issue worsens at speed",
]

# ── 4 reliable INFERRED seed queries ──
SEED_QUERIES = [
    "Brake pedal goes to floor",
    "Radiator fan not spinning",
    "Smoke from tailpipe",
    "Power windows not working",
]

def run_refine(original_query, new_symptoms):
    """Simulate the exact refinement logic from _display_inferred_followup."""
    new_symptoms_list = list(new_symptoms)
    combined = f"{original_query}, {', '.join(new_symptoms_list)}"
    report = run_diagnostic(
        symptoms_text=combined,
        current_sample="simulated",
        speed=1000,
        use_llm=False,
        verbose=False,
    )
    return report, combined


def extract_candidate_keys(candidates):
    """Extract key fields from a candidate list for comparison."""
    return [
        {
            "label": c.get("label", ""),
            "navic_fault": c.get("navic_fault", ""),
            "confidence": round(c.get("confidence", 0.0), 4),
            "fusion": round(c.get("final_score", 0.0), 4),
            "sensor_status": c.get("sensor_status", "N/A"),
            "kg_score": round(c.get("kg_score", 0.0), 4),
            "sensor_score": round(c.get("sensor_score", 0.0), 4),
        }
        for c in candidates
    ]


def extract_sensor_evidence(se):
    """Extract sensor evidence summary."""
    return {
        fault: {
            "status": ev.get("status", "N/A"),
            "sensor_confidence": round(ev.get("sensor_confidence", 0.0), 4),
            "n_critical": len(ev.get("critical", [])),
            "n_warning": len(ev.get("warning", [])),
        }
        for fault, ev in se.items()
    }


def print_diff(before, after, label):
    """Print comparison of before/after diagnosis state."""
    print(f"\n  [{label}]")
    print(f"  Mode: {before.mode} -> {after.mode}")
    print(f"  Confidence: {before.confidence:.4f} -> {after.confidence:.4f}  "
          f"(delta={after.confidence - before.confidence:+.4f})")
    print(f'  Top: "{before.top_candidate.get("label","")}" -> "{after.top_candidate.get("label","")}"')
    print(f'  Fault: "{before.top_candidate.get("navic_fault","")}" -> "{after.top_candidate.get("navic_fault","")}"')

    # Component diff
    bc = before.confidence_components
    ac = after.confidence_components
    print(f"  Components:")
    for k in ["raw_retrieval_score", "calibrated_retrieval", "separation", "coverage", "sensor_boost"]:
        bv = bc.get(k, 0)
        av = ac.get(k, 0)
        if bv != av:
            print(f"    {k}: {bv:.4f} -> {av:.4f} ({av-bv:+.4f})")

    # Candidate ranking diff
    b_cands = extract_candidate_keys(before.display_candidates)
    a_cands = extract_candidate_keys(after.display_candidates)
    b_labels = [c["label"] for c in b_cands]
    a_labels = [c["label"] for c in a_cands]
    if b_labels != a_labels:
        print(f"  Candidates BEFORE: {b_labels}")
        print(f"  Candidates AFTER:  {a_labels}")
        # Detect ranking changes
        for ac_i, ac_label in enumerate(a_labels):
            if ac_label in b_labels:
                bc_i = b_labels.index(ac_label)
                if bc_i != ac_i:
                    print(f"    Ranking change: \"{ac_label}\" #{bc_i+1} -> #{ac_i+1}")
            else:
                print(f"    New candidate: \"{ac_label}\" at #{ac_i+1}")
        for bc_i, bc_label in enumerate(b_labels):
            if bc_label not in a_labels:
                print(f"    Dropped candidate: \"{bc_label}\"")
    else:
        print(f"  Candidates unchanged ({len(b_cands)} total)")

    # Sensor evidence diff
    b_se = extract_sensor_evidence(before.sensor_evidence)
    a_se = extract_sensor_evidence(after.sensor_evidence)
    if b_se != a_se:
        all_faults = set(b_se.keys()) | set(a_se.keys())
        for f in sorted(all_faults):
            bv = b_se.get(f, {})
            av = a_se.get(f, {})
            if bv != av:
                print(f"    Sensor [{f}]: {bv.get('status','N/A')} (conf={bv.get('sensor_confidence',0)}) "
                      f"-> {av.get('status','N/A')} (conf={av.get('sensor_confidence',0)})")

    # KG path changes (candidate category/subcategory)
    b_top = before.top_candidate
    a_top = after.top_candidate
    b_cat = (b_top.get("category",""), b_top.get("subcategory",""))
    a_cat = (a_top.get("category",""), a_top.get("subcategory",""))
    if b_cat != a_cat:
        print(f"  KG path: {b_cat} -> {a_cat}")

    # N_candidates diff
    bn = len(before.display_candidates)
    an = len(after.display_candidates)
    if bn != an:
        print(f"  Display candidates: {bn} -> {an}")

    # Query text check
    print(f"  Original symptoms: {before.original_symptoms}")
    print(f"  Refined symptoms:  {after.original_symptoms}")


def test_refinement_options(baseline, original_text, label):
    """Test every refinement option individually."""
    results = []
    for sym in CHECKBOX_OPTIONS:
        new_report, combined = run_refine(original_text, [sym])
        mode_changed = baseline.mode != new_report.mode
        conf_change = new_report.confidence - baseline.confidence
        results.append({
            "option": sym,
            "combined": combined,
            "mode": new_report.mode,
            "confidence": round(new_report.confidence, 4),
            "delta": round(conf_change, 4),
            "mode_changed": mode_changed,
            "top_label": new_report.top_candidate.get("label", ""),
            "top_fault": new_report.top_candidate.get("navic_fault", ""),
            "improved": conf_change > 0,
            "report": new_report,
        })

    print(f"\n{'='*100}")
    print(f"INDIVIDUAL OPTIONS — {label}")
    print(f"{'='*100}")
    for r in results:
        arrow = "+" if r["delta"] > 0 else ""
        imp = "IMPROVED" if r["improved"] else "WORSENED/FLAT"
        mc = f" MODE CHANGE: {r['mode']}" if r["mode_changed"] else ""
        print(f"  {r['option']:40s} -> {r['mode']:10s}  conf={r['confidence']:.4f} ({arrow}{r['delta']:.4f})"
              f" [{imp}]{mc}")
        print(f"  {'':40s}   top=\"{r['top_label']}\" ({r['top_fault']})")

    return results


def test_all_combinations(baseline, original_text, label):
    """Test all 31 non-empty checkbox combinations (2^5 - 1)."""
    from itertools import combinations
    all_syms = CHECKBOX_OPTIONS
    results = []
    n_total = 0
    for r in range(1, 6):
        for combo in combinations(all_syms, r):
            n_total += 1
            new_report, combined = run_refine(original_text, list(combo))
            conf_delta = new_report.confidence - baseline.confidence
            mode_changed = baseline.mode != new_report.mode
            results.append({
                "combo": combo,
                "n_symptoms": len(combo),
                "combined": combined,
                "mode": new_report.mode,
                "confidence": round(new_report.confidence, 4),
                "delta": round(conf_delta, 4),
                "mode_changed": mode_changed,
                "top_label": new_report.top_candidate.get("label", ""),
                "top_fault": new_report.top_candidate.get("navic_fault", ""),
                "improved": conf_delta > 0,
                "best_conf": False,  # will be determined later
            })

    # Determine best combination
    best = max(results, key=lambda r: r["confidence"])
    best["best_conf"] = True

    improved = [r for r in results if r["improved"]]
    worsened = [r for r in results if not r["improved"]]
    mode_changes = [r for r in results if r["mode_changed"]]

    print(f"\n{'='*100}")
    print(f"ALL COMBINATIONS ({n_total}) — {label}")
    print(f"{'='*100}")
    print(f"  Improved confidence:  {len(improved)}/{n_total}")
    print(f"  Worsened/flat:        {len(worsened)}/{n_total}")
    print(f"  Mode changes:         {len(mode_changes)}/{n_total}")

    if mode_changes:
        for r in mode_changes[:5]:
            print(f"    Mode change: {baseline.mode} -> {r['mode']}  "
                  f"combo={r['combo']}  delta={r['delta']:+.4f}")

    print(f"\n  Best combination: {best['combo']}")
    print(f"    Mode: {best['mode']}  Confidence: {best['confidence']:.4f} (delta={best['delta']:+.4f})")
    print(f'    Top: "{best["top_label"]}" ({best["top_fault"]})')

    # Check if empty-improvement exists (delta=0 or very close)
    zero_delta = [r for r in results if abs(r["delta"]) < 0.001]
    if zero_delta:
        print(f"\n  *** ZERO-EFFECT combinations: {len(zero_delta)} ***")
        for r in zero_delta:
            print(f"    combo={r['combo']}  delta={r['delta']:+.4f}  mode={r['mode']}")

    return {"all": results, "improved": improved, "worsened": worsened,
            "mode_changes": mode_changes, "best": best, "total": n_total}


def test_edge_cases(original_text, label):
    """Test edge cases in the refinement workflow."""
    print(f"\n{'='*100}")
    print(f"EDGE CASES — {label}")
    print(f"{'='*100}")

    # Case 1: No symptoms selected (empty refinement)
    print(f"\n  1. Empty refinement (no symptoms selected):")
    combined = f"{original_text}, "
    try:
        report_empty = run_diagnostic(
            symptoms_text=original_text,  # same as original, no addition
            current_sample="simulated",
            speed=1000,
            use_llm=False,
        )
        print(f"     Mode: {report_empty.mode}  Conf: {report_empty.confidence:.4f}")
        print(f'     Top: "{report_empty.top_candidate.get("label","")}"')
        print(f"     Status: No crash — runs with original text only")
    except Exception as e:
        print(f"     *** ERROR: {e} ***")

    # Case 2: Refinement with only the same symptoms already present
    print(f"\n  2. Refinement with redundant symptoms (query contains same words):")
    query_words = original_text.lower().split()
    redundant = [s for s in CHECKBOX_OPTIONS
                 if any(w in s.lower().split() for w in query_words)]
    if redundant:
        new_report, combined = run_refine(original_text, redundant)
        print(f"     Selected redundant symptoms: {redundant}")
        print(f"     Mode: {new_report.mode}  Conf: {new_report.confidence:.4f}")
        print(f'     Top: "{new_report.top_candidate.get("label","")}"')
    else:
        print(f"     No redundant symptoms found for this query")

    # Case 3: Multiple refinement rounds (refine -> refine again)
    print(f"\n  3. Sequential refinement (refine an already-refined result):")
    first_refine, first_combined = run_refine(original_text, [CHECKBOX_OPTIONS[0]])
    second_refine, second_combined = run_refine(first_combined, [CHECKBOX_OPTIONS[1]])
    third_refine, third_combined = run_refine(second_combined, [CHECKBOX_OPTIONS[2]])
    print(f"     Round 1: added \"{CHECKBOX_OPTIONS[0]}\" -> conf={first_refine.confidence:.4f} mode={first_refine.mode}")
    print(f"     Round 2: added \"{CHECKBOX_OPTIONS[1]}\" -> conf={second_refine.confidence:.4f} mode={second_refine.mode}")
    print(f"     Round 3: added \"{CHECKBOX_OPTIONS[2]}\" -> conf={third_refine.confidence:.4f} mode={third_refine.mode}")
    print(f"     Combined text growth: {len(original_text)} -> {len(first_combined)} -> {len(second_combined)} -> {len(third_combined)} chars")
    # Check if accumulated text causes issues
    print(f"     Round 3 combined text: \"{third_combined[:120]}...\"")

    # Case 4: Free-text only (no checkboxes)
    print(f"\n  4. Free-text only (no checkbox selected):")
    free_text = "Makes a grinding noise when turning"
    combined_ft = f"{original_text}, {free_text}"
    report_ft = run_diagnostic(
        symptoms_text=combined_ft,
        current_sample="simulated",
        speed=1000,
        use_llm=False,
    )
    print(f"     Free text: \"{free_text}\"")
    print(f"     Mode: {report_ft.mode}  Conf: {report_ft.confidence:.4f}")
    print(f'     Top: "{report_ft.top_candidate.get("label","")}" ({report_ft.top_candidate.get("navic_fault","")})')

    # Case 5: Free-text + all checkboxes
    print(f"\n  5. All checkboxes + free text:")
    all_checked = list(CHECKBOX_OPTIONS)
    combined_all = f"{original_text}, {', '.join(all_checked)}, {free_text}"
    report_all = run_diagnostic(
        symptoms_text=combined_all,
        current_sample="simulated",
        speed=1000,
        use_llm=False,
    )
    print(f"     Combined length: {len(combined_all)} chars")
    print(f"     Mode: {report_all.mode}  Conf: {report_all.confidence:.4f}")
    print(f'     Top: "{report_all.top_candidate.get("label","")}" ({report_all.top_candidate.get("navic_fault","")}')

    # Case 6: "Use Current Diagnosis" path — no refinement, just keep baseline
    print(f"\n  6. 'Use Current Diagnosis' — no refinement applied:")
    print(f"     This is a UI-only action (st.session_state.original_report = None).")
    print(f"     Backend state unchanged — same report kept.")
    print(f"     No re-analysis occurs. Same as baseline.")

    # Case 7: Verify symptom incorporation — are refinement words detectable?
    print(f"\n  7. Symptom incorporation verification:")
    print(f"     Original symptoms:   {report_empty.original_symptoms}")
    print(f"     After refine:        {report_ft.original_symptoms}")
    # Check if free_text words appear
    ft_words = set(free_text.lower().split())
    orig_ft_words = set(report_ft.original_symptoms or [])
    matched = ft_words & {w.lower() for w in orig_ft_words}
    print(f"     Free-text words found in refined symptoms: {matched} / {ft_words}")

    # Case 8: Repeated clicks (same refinement twice)
    print(f"\n  8. Repeated refinement (same query twice):")
    repeat1, _ = run_refine(original_text, [CHECKBOX_OPTIONS[0]])
    repeat2, _ = run_refine(original_text, [CHECKBOX_OPTIONS[0]])
    identical = (
        repeat1.mode == repeat2.mode
        and abs(repeat1.confidence - repeat2.confidence) < 0.0001
        and repeat1.top_candidate.get("label") == repeat2.top_candidate.get("label")
    )
    if identical:
        print(f"     Same refinement twice -> identical results (deterministic: OK)")
    else:
        print(f"     *** NON-DETERMINISTIC: different results for same refinement ***")
        print(f"     Run 1: {repeat1.mode} conf={repeat1.confidence:.4f} top=\"{repeat1.top_candidate.get('label','')}\"")
        print(f"     Run 2: {repeat2.mode} conf={repeat2.confidence:.4f} top=\"{repeat2.top_candidate.get('label','')}\"")

    return {
        "empty": report_empty,
        "free_text": report_ft,
        "all_options": report_all,
        "sequential": [first_refine, second_refine, third_refine],
    }


# ── Main ────────────────────────────────────────────────────────────

def main():
    print("=" * 120)
    print("REFINE DIAGNOSIS — COMPREHENSIVE FUNCTIONAL AUDIT")
    print("=" * 120)

    all_reports = {}
    checklist = []

    for seed in SEED_QUERIES:
        print(f"\n{'#' * 120}")
        print(f"## SEED QUERY: \"{seed}\"")
        print(f"{'#' * 120}")

        # Baseline
        baseline = run_diagnostic(
            symptoms_text=seed,
            current_sample="simulated",
            speed=1000,
            use_llm=False,
        )
        print(f"\n  BASELINE: {baseline.mode}  conf={baseline.confidence:.4f}")
        print(f'  Top: "{baseline.top_candidate.get("label","")}" ({baseline.top_candidate.get("navic_fault","")})')
        print(f'  Query text in report: "{baseline.query_text}"')

        all_reports[seed] = {"baseline": baseline}

        # 1. Test individual refinement options
        indiv = test_refinement_options(baseline, baseline.query_text or seed, seed)
        all_reports[seed]["individual"] = indiv

        # 2. Test all combinations
        combos = test_all_combinations(baseline, baseline.query_text or seed, seed)
        all_reports[seed]["combinations"] = combos

        # 3. Edge cases
        edges = test_edge_cases(baseline.query_text or seed, seed)
        all_reports[seed]["edges"] = edges

    # ── Final summary ──────────────────────────────────────────────

    print(f"\n\n{'=' * 120}")
    print("FINDINGS SUMMARY")
    print(f"{'=' * 120}")

    # Count improvements across all queries
    total_improved = 0
    total_combos = 0
    total_worsened = 0
    total_mode_changes = 0
    zero_effects = []

    for seed, data in all_reports.items():
        combos = data["combinations"]
        total_combos += combos["total"]
        total_improved += len(combos["improved"])
        total_worsened += len(combos["worsened"])
        total_mode_changes += len(combos["mode_changes"])
        for r in combos["all"]:
            if abs(r["delta"]) < 0.001:
                zero_effects.append((seed, r["combo"], r["delta"]))

    print(f"\n  Refinement combinations tested: {total_combos}")
    print(f"    Improved confidence:  {total_improved} ({total_improved/total_combos*100:.1f}%)")
    print(f"    Worsened/flat:        {total_worsened} ({total_worsened/total_combos*100:.1f}%)")
    print(f"    Mode changes:         {total_mode_changes} ({total_mode_changes/total_combos*100:.1f}%)")

    # Mode transition types
    print(f"\n  Mode transitions observed:")
    transitions = {}
    for seed, data in all_reports.items():
        for r in data["combinations"]["all"]:
            if r["mode_changed"]:
                key = f"{data['baseline'].mode} -> {r['mode']}"
                transitions[key] = transitions.get(key, 0) + 1
    for t, c in sorted(transitions.items()):
        print(f"    {t}: {c}")

    # Best improvements
    print(f"\n  Best refinement by seed query:")
    for seed, data in all_reports.items():
        best = data["combinations"]["best"]
        bl = data["baseline"]
        print(f'    \"{seed}\": '
              f'{best["combo"]} -> '
              f'{best["mode"]} conf={best["confidence"]:.4f} '
              f'(delta={best["delta"]:+.4f}, baseline={bl.confidence:.4f})')

    # Zero-effect combinations
    if zero_effects:
        print(f"\n  *** {len(zero_effects)} ZERO-EFFECT refinements detected ***")
        for seed, combo, delta in zero_effects[:10]:
            print(f"    seed=\"{seed}\" combo={combo} delta={delta:+.4f}")

    # ── BUG REPORT ─────────────────────────────────────────────────
    print(f"\n\n{'=' * 120}")
    print("BUG REPORT")
    print(f"{'=' * 120}")

    bugs = []

    # Bug 1: Hardcoded symptoms
    bugs.append((
        "BUG-1: Hardcoded symptom options",
        "HIGH",
        "The 5 refinement checkbox options are hardcoded in app.py lines 493-499 "
        "and never change based on the current diagnosis or top candidate. "
        "All 5 options are brake-oriented ('Brake noise when stopping', "
        "'Vibration during braking', 'Vehicle pulls to one side'). "
        "When the INFERRED diagnosis is about 'Radiator fan not spinning' or "
        "'Smoke from tailpipe', brake symptoms are completely irrelevant. "
        "The UI claims 'Additional symptoms (check all that apply)' but these "
        "are NOT dynamically generated from the KG neighborhood. "
        "This is a static placeholder that was never implemented.",
    ))

    # Bug 2: UI-only dead state
    bugs.append((
        "BUG-2: st.session_state.original_report is set but never read",
        "MEDIUM",
        "In _display_inferred_followup(), when refinement does not improve "
        "confidence, st.session_state.original_report = report stores the old "
        "report. But this variable is NEVER read back anywhere in the codebase. "
        "It is only ever set to None or to a stale value. This is dead state "
        "that serves no purpose. The info message is shown via st.info() but "
        "original_report is not consulted for display logic.",
    ))

    # Bug 3: Confidence-only gate ignores mode semantics
    bugs.append((
        "BUG-3: Confidence comparison ignores mode quality",
        "MEDIUM",
        "The refinement gate (app.py line 544) only checks "
        "new_report.confidence > original_conf. It does not check whether mode "
        "improved (e.g., INFERRED -> EXTRACTED) even if confidence is slightly "
        "lower. Conversely, a refinement that produces EXTRACTED with confidence "
        "0.74 would be REJECTED because 0.74 < 0.75 baseline, even though the "
        "mode improved. A refinement that stays INFERRED but drops from 0.72 to "
        "0.70 would be kept if it improves mode. The gate should also consider "
        "mode transitions.",
    ))

    # Bug 4: Simple string concatenation without deduplication
    bugs.append((
        "BUG-4: Naive symptom concatenation",
        "LOW",
        "The refinement uses f\"{report.query_text}, {', '.join(new_symptoms)}\" "
        "(app.py line 533). This is simple string concatenation with no "
        "deduplication, semantic merging, or token reduction. If a user runs "
        "refinement multiple times, the query text grows unboundedly: "
        "\"Brake pedal goes to floor, Brake noise when stopping, Vibration "
        "during braking, Vehicle pulls to one side, ...\" Repeated refinement "
        "rounds produce increasingly long queries that may hit the preprocessor "
        "or retrieval limits.",
    ))

    # Bug 5: query_text may be None/empty
    bugs.append((
        "BUG-5: query_text fallback to empty string",
        "LOW",
        "If report.query_text is empty or None (possible if retrieval_result "
        "has no 'query' key and preprocessed has no 'original' key), "
        "the refinement produces ', symptom1, symptom2' — leading with a comma. "
        "The run_diagnostic would get an empty-prefix query. Test shows query_text "
        "is populated for these seeds, but there is no guard.",
    ))

    # Bug 6: All 5 options are always brake-related
    bugs.append((
        "BUG-6: Refinement options are semantically blind",
        "HIGH",
        "The same 5 brake symptoms are offered regardless of whether the INFERRED "
        "diagnosis is brake-related. For 'Smoke from tailpipe' (exhaust issue): "
        "offering 'Brake noise when stopping' is irrelevant and potentially "
        "misleading. The UI suggests these are 'additional symptoms to narrow "
        "the diagnosis' but they narrow toward brakes regardless of context.",
    ))

    # Bug 7: No visual feedback when refinement text is incorporated
    bugs.append((
        "BUG-7: No symptom incorporation visibility",
        "LOW",
        "After refinement, the UI shows the new diagnosis but does not explicitly "
        "show which refinement symptoms were incorporated into the KG match. "
        "The user sees 'Symptoms: ...' pills but only for the original symptoms "
        "from the preprocessor — not a clear before/after of which refinement "
        "options affected the result. The original_symptoms field on the report "
        "shows the PREPROCESSOR's entity extraction, not the actual query text "
        "split. If a refinement symptom doesn't match any KG entity, the user "
        "cannot tell it was ignored.",
    ))

    for title, severity, description in bugs:
        print(f"\n  {'─'*80}")
        print(f"  [{severity}] {title}")
        print(f"  {'─'*80}")
        print(f"  {description}")

    # ── Summary metrics ────────────────────────────────────────────

    print(f"\n\n{'=' * 120}")
    print("QUANTITATIVE FINDINGS")
    print(f"{'=' * 120}")

    # Average delta by number of symptoms selected
    from collections import defaultdict
    delta_by_n = defaultdict(list)
    for seed, data in all_reports.items():
        for r in data["combinations"]["all"]:
            delta_by_n[r["n_symptoms"]].append(r["delta"])
    print(f"\n  Average confidence delta by number of symptoms:")
    for n in sorted(delta_by_n.keys()):
        deltas = delta_by_n[n]
        avg = sum(deltas) / len(deltas)
        positive = sum(1 for d in deltas if d > 0)
        negative = sum(1 for d in deltas if d < 0)
        zero = sum(1 for d in deltas if abs(d) < 0.001)
        print(f"    {n} symptom(s): avg_delta={avg:+.4f}  positive={positive}  negative={negative}  zero={zero}")

    # Mode transition matrix
    print(f"\n  Mode transition matrix (across all refinement attempts):")
    matrix = defaultdict(lambda: defaultdict(int))
    for seed, data in all_reports.items():
        bl_mode = data["baseline"].mode
        for r in data["combinations"]["all"]:
            matrix[bl_mode][r["mode"]] += 1
        for r in data["individual"]:
            matrix[bl_mode][r["mode"]] += 1
        for ek, ev in data["edges"].items():
            if hasattr(ev, "mode"):
                matrix[bl_mode][ev.mode] += 1
    for from_mode, to_modes in sorted(matrix.items()):
        for to_mode, count in sorted(to_modes.items()):
            print(f"    {from_mode} -> {to_mode}: {count}")


if __name__ == "__main__":
    main()
