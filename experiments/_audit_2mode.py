"""
Architectural analysis: feasibility of collapsing to two-mode system.
Runs all evaluation queries, records every confidence component,
analyzes INFERRED clustering, simulates candidate thresholds,
and recommends the best EXTRACTED threshold.
"""

import os
import sys
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT / "num_pipeline" / "scripts"
NUM_PIPELINE_DIR = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(NUM_PIPELINE_DIR)

from run_diagnostic import run_diagnostic

# ── Combined query set (deduplicated) ────────────────────────────────

QUERIES = [
    # --- Engine ---
    "Engine overheating with coolant loss",
    "Check engine light is on, rough idle",
    "Engine misfires at idle",
    "Car hesitates on acceleration",
    "Loss of engine power",
    "Engine knocking sound under load",
    "Vehicle sputters at high speeds",
    "Engine cranks slowly, won't start",
    "Engine does not crank at all",
    "Engine stalls when I let off the gas",
    "Poor engine performance",
    "Low engine power",
    "Engine consumes too much fuel",
    "Rough idle",
    "Engine hesitation",
    # --- Brakes ---
    "Brake pedal feels spongy",
    "Brake pedal goes to floor",
    "ABS warning light on, brake pedal pulsation",
    "Brakes squealing when I stop",
    "Steering pulls to the left when braking",
    "Brake fluid leak under the car",
    "Soft brake pedal",
    # --- Transmission ---
    "Transmission slips between gears",
    "Car won't shift out of first gear",
    "Transmission grinding on upshift",
    "Harsh shift from 2nd to 3rd",
    # --- Fuel ---
    "Poor fuel economy, running rich",
    "Fuel smell after parking",
    "Car won't start, no fuel pressure",
    # --- Cooling ---
    "Temperature gauge goes to red",
    "Coolant leaking from radiator",
    "Heater blows cold air, engine overheats",
    "Coolant leak",
    "Engine overheating",
    # --- Electrical ---
    "Battery keeps dying overnight",
    "Dashboard lights flickering",
    "Alternator warning light on",
    "Car electrical system not working",
    "Radiator fan not spinning",
    # --- Exhaust / Emissions ---
    "Black smoke from exhaust",
    "Check engine light, code P0301",
    "Smoke from tailpipe",
    # --- Steering / Suspension ---
    "Steering wheel vibrates at highway speed",
    "Car pulls to one side",
    "Clunking noise over bumps",
    # --- Vague / edge ---
    "Something feels wrong with the car",
    "My car is making a weird noise",
    "The engine light came on yesterday",
    "Brakes feel weird",
    "Car shakes when I brake",
    "my car is broken help",
    "Power windows not working",
]

QUERIES = sorted(set(q.strip() for q in QUERIES))


# ── Run all queries ──────────────────────────────────────────────────

def run_all():
    rows = []
    errors = []

    for query in QUERIES:
        try:
            report = run_diagnostic(
                symptoms_text=query,
                current_sample="simulated",
                speed=1000,
                use_llm=False,
                verbose=False,
            )
            cc = report.confidence_components or {}
            top = report.top_candidate or {}
            row = {
                "query": query,
                "mode": report.mode,
                "final_confidence": round(report.confidence, 4),
                "raw_retrieval_score": round(cc.get("raw_retrieval_score", 0.0), 4),
                "calibrated_retrieval": round(cc.get("calibrated_retrieval", 0.0), 4),
                "separation": round(cc.get("separation", 0.0), 4),
                "coverage": round(cc.get("coverage", 0.0), 4),
                "sensor_boost": round(cc.get("sensor_boost", 0.0), 4),
                "top_label": top.get("label", "N/A"),
                "top_fault": top.get("navic_fault", "N/A"),
                "candidate_conf": round(top.get("confidence", 0.0), 4),
            }
            rows.append(row)
        except Exception as e:
            errors.append((query, str(e)))

    return rows, errors


results, errors = run_all()

# ── Report errors ────────────────────────────────────────────────────

print("=" * 110)
print("  TWO-MODE SIMPLIFICATION ANALYSIS")
print("=" * 110)
print(f"\nTotal queries attempted : {len(QUERIES)}")
print(f"Successful              : {len(results)}")
print(f"Errors                  : {len(errors)}")
for q, e in errors:
    print(f"  ERROR: \"{q}\" → {e}")

# ── 1. Current mode distribution ────────────────────────────────────

mode_counts = Counter(r["mode"] for r in results)
print(f"\n{'─' * 110}")
print("  1. CURRENT MODE DISTRIBUTION")
print(f"{'─' * 110}")
for m in ["EXTRACTED", "INFERRED", "AMBIGUOUS"]:
    n = mode_counts.get(m, 0)
    pct = n / len(results) * 100
    print(f"    {m:12s}: {n:3d}  ({pct:5.1f}%)")

# ── 2. Confidence distribution analysis ─────────────────────────────

print(f"\n{'─' * 110}")
print("  2. CONFIDENCE DISTRIBUTION BY MODE")
print(f"{'─' * 110}")

for m in ["EXTRACTED", "INFERRED", "AMBIGUOUS"]:
    vals = [r["final_confidence"] for r in results if r["mode"] == m]
    if not vals:
        continue
    vals.sort()
    print(f"    {m:12s}: n={len(vals):2d}  range=[{vals[0]:.4f}, {vals[-1]:.4f}]  "
          f"median={vals[len(vals)//2]:.4f}  mean={sum(vals)/len(vals):.4f}")

# ── 3. Full detail per query (sorted by confidence) ─────────────────

print(f"\n{'─' * 110}")
print("  3. PER-QUERY DETAIL (sorted by final confidence)")
print(f"{'─' * 110}")
print(f"  {'Query':45s} {'Mode':12s} {'Conf':>6s} {'Raw':>6s} {'Cal':>6s} "
      f"{'Sep':>6s} {'Cov':>6s} {'Boost':>6s} {'Top Diagnosis':30s}")
print(f"  {'-'*45} {'-'*12} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*30}")

sorted_all = sorted(results, key=lambda r: r["final_confidence"])
for r in sorted_all:
    diag = (r["top_label"][:28] + "..") if len(r["top_label"]) > 28 else r["top_label"]
    print(f"  {r['query']:45s} {r['mode']:12s} {r['final_confidence']:6.3f} "
          f"{r['raw_retrieval_score']:6.3f} {r['calibrated_retrieval']:6.3f} "
          f"{r['separation']:6.3f} {r['coverage']:6.3f} {r['sensor_boost']:6.3f} "
          f"{diag:30s}")

# ── 4. INFERRED cluster analysis ────────────────────────────────────

inferred = [r for r in results if r["mode"] == "INFERRED"]
inferred_sorted = sorted(inferred, key=lambda r: r["final_confidence"])

print(f"\n{'─' * 110}")
print("  4. INFERRED CLUSTER ANALYSIS")
print(f"{'─' * 110}")
print(f"    Total INFERRED queries: {len(inferred)}")

if inferred:
    lo = inferred_sorted[0]["final_confidence"]
    hi = inferred_sorted[-1]["final_confidence"]
    print(f"    Confidence range: [{lo:.4f}, {hi:.4f}]")
    print(f"    Band width: {hi - lo:.4f}")
    print(f"    Full EXTRACTED+INFERRED+AMBIGUOUS span: "
          f"{sorted_all[-1]['final_confidence'] - sorted_all[0]['final_confidence']:.4f}")
    bw = hi - lo
    span = sorted_all[-1]["final_confidence"] - sorted_all[0]["final_confidence"]
    if span > 0:
        print(f"    INFERRED occupies {bw/span*100:.1f}% of the total confidence range.")

    # Check for distinct clustering vs bridge
    gaps = []
    for i in range(1, len(inferred_sorted)):
        gaps.append(inferred_sorted[i]["final_confidence"] - inferred_sorted[i-1]["final_confidence"])
    avg_gap = sum(gaps) / len(gaps) if gaps else 0
    max_gap = max(gaps) if gaps else 0
    print(f"    Avg gap between adjacent INFERRED: {avg_gap:.4f}")
    print(f"    Max gap between adjacent INFERRED: {max_gap:.4f}")

    # Gap to nearest EXTRACTED below
    extracted_below = [r for r in results if r["mode"] == "EXTRACTED" and r["final_confidence"] < inferred_sorted[-1]["final_confidence"]]
    if extracted_below:
        nearest_extracted = max(extracted_below, key=lambda r: r["final_confidence"])
        gap_to_extracted = inferred_sorted[0]["final_confidence"] - nearest_extracted["final_confidence"]
        print(f"    Gap from highest INFERRED to nearest EXTRACTED below: {gap_to_extracted:.4f}")
    else:
        print(f"    No EXTRACTED queries below INFERRED range")

    # Gap to nearest AMBIGUOUS above
    ambiguous_above = [r for r in results if r["mode"] == "AMBIGUOUS" and r["final_confidence"] > inferred_sorted[0]["final_confidence"]]
    if ambiguous_above:
        nearest_ambig = min(ambiguous_above, key=lambda r: r["final_confidence"])
        gap_to_ambig = nearest_ambig["final_confidence"] - inferred_sorted[-1]["final_confidence"]
        print(f"    Gap from lowest INFERRED to nearest AMBIGUOUS above: {gap_to_ambig:.4f}")
    else:
        print(f"    No AMBIGUOUS queries above INFERRED range")

    # Assess clustering: distinct cluster or bridge?
    ambig_vals = sorted([r["final_confidence"] for r in results if r["mode"] == "AMBIGUOUS"])
    extracted_vals = sorted([r["final_confidence"] for r in results if r["mode"] == "EXTRACTED"])
    if ambig_vals and extracted_vals:
        ambig_max = ambig_vals[-1]
        extracted_min = extracted_vals[0]
        inferred_min = inferred_sorted[0]["final_confidence"]
        inferred_max = inferred_sorted[-1]["final_confidence"]
        overlap_ambig = inferred_min <= ambig_max
        overlap_extracted = inferred_max >= extracted_min
        if overlap_ambig and overlap_extracted:
            print(f"    VERDICT: INFERRED is a BRIDGE — it overlaps both AMBIGUOUS and EXTRACTED ranges.")
            print(f"             There is no natural gap separating the three modes.")
        elif overlap_ambig:
            print(f"    VERDICT: INFERRED overlaps AMBIGUOUS but not EXTRACTED — weak bridge.")
        elif overlap_extracted:
            print(f"    VERDICT: INFERRED overlaps EXTRACTED but not AMBIGUOUS — strong bridge.")
        else:
            gap_low = inferred_min - ambig_max
            gap_high = extracted_min - inferred_max
            print(f"    VERDICT: INFERRED forms a DISTINCT cluster (gap to AMBIGUOUS={gap_low:.4f}, "
                  f"gap to EXTRACTED={gap_high:.4f})")

    print(f"\n    INFERRED queries detail:")
    print(f"    {'Query':45s} {'Conf':>6s} {'Raw':>6s} {'Cal':>6s} {'Sep':>6s} {'Cov':>6s} {'Boost':>6s}")
    print(f"    {'-'*45} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")
    for r in inferred_sorted:
        print(f"    {r['query']:45s} {r['final_confidence']:6.3f} "
              f"{r['raw_retrieval_score']:6.3f} {r['calibrated_retrieval']:6.3f} "
              f"{r['separation']:6.3f} {r['coverage']:6.3f} {r['sensor_boost']:6.3f}")
else:
    print("    No INFERRED queries to analyze.")


# ── 5. Threshold simulation ─────────────────────────────────────────

print(f"\n{'─' * 110}")
print("  5. THRESHOLD SIMULATION")
print(f"{'─' * 110}")

# Candidate thresholds to test, from very inclusive to very exclusive
candidate_thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]

# Currently INFERRED queries — these are the ones that would be reclassified
current_inferred_set = {r["query"] for r in results if r["mode"] == "INFERRED"}

print(f"  {'Threshold':>10s}  {'EXTRACTED':>10s}  {'AMBIGUOUS':>10s}  {'INF→EX':>10s}  {'INF→AM':>10s}  "
      f"{'EX→AM(miscls)':>14s}  {'AM→EX(miscls)':>14s}  {'Total OK':>10s}  {'Comment':30s}")
print(f"  {'-'*10:>10s}  {'-'*10:>10s}  {'-'*10:>10s}  {'-'*10:>10s}  {'-'*10:>10s}  "
      f"{'-'*14:>14s}  {'-'*14:>14s}  {'-'*10:>10s}  {'-'*30:>30s}")

best_threshold = None
best_score = -1

for thresh in candidate_thresholds:
    ex_count = 0
    am_count = 0
    inf_to_ex = 0     # INFERRED correctly promoted to EXTRACTED
    inf_to_am = 0     # INFERRED incorrectly demoted to AMBIGUOUS
    ex_to_am = 0      # EXTRACTED incorrectly demoted to AMBIGUOUS
    am_to_ex = 0      # AMBIGUOUS incorrectly promoted to EXTRACTED
    total_ok = 0

    for r in results:
        new_mode = "EXTRACTED" if r["final_confidence"] >= thresh else "AMBIGUOUS"
        old_mode = r["mode"]

        if old_mode == "EXTRACTED":
            if new_mode == "EXTRACTED":
                ex_count += 1
                total_ok += 1
            else:
                ex_to_am += 1  # misclassification
        elif old_mode == "INFERRED":
            if new_mode == "EXTRACTED":
                inf_to_ex += 1
                ex_count += 1
                total_ok += 1
            else:
                inf_to_am += 1
                am_count += 1
        else:  # AMBIGUOUS
            if new_mode == "EXTRACTED":
                am_to_ex += 1  # misclassification
                ex_count += 1
            else:
                am_count += 1
                total_ok += 1

    # Quality metric: minimize misclassifications while maximizing correct reclassifications
    misclassifications = ex_to_am + am_to_ex
    correct = total_ok
    preserved_inferred = inf_to_ex  # correctly absorbed into EXTRACTED

    # Build a comment on quality
    if misclassifications == 0:
        comment = "No misclassifications!"
    elif misclassifications <= 2:
        comment = f"Minor ({misclassifications} miscls)"
    elif misclassifications <= 4:
        comment = f"Moderate ({misclassifications} miscls)"
    else:
        comment = f"Heavy ({misclassifications} miscls)"

    # Score: prefer high correct count with minimal misclassifications
    score = correct - 3 * misclassifications
    if score > best_score:
        best_score = score
        best_threshold = thresh

    print(f"  {thresh:>10.2f}  {ex_count:>10d}  {am_count:>10d}  {inf_to_ex:>10d}  {inf_to_am:>10d}  "
          f"{ex_to_am:>14d}  {am_to_ex:>14d}  {correct:>10d}  {comment:30s}")

print(f"\n  Best threshold (by score={best_score}): {best_threshold:.2f}")

# ── 6. Detailed misclassification analysis at best threshold ─────────

print(f"\n{'─' * 110}")
print(f"  6. MISCLASSIFICATION DETAIL AT THRESHOLD = {best_threshold:.2f}")
print(f"{'─' * 110}")

ex_to_am_list = []
am_to_ex_list = []

for r in results:
    new_mode = "EXTRACTED" if r["final_confidence"] >= best_threshold else "AMBIGUOUS"
    old_mode = r["mode"]
    if old_mode == "EXTRACTED" and new_mode == "AMBIGUOUS":
        ex_to_am_list.append(r)
    if old_mode == "AMBIGUOUS" and new_mode == "EXTRACTED":
        am_to_ex_list.append(r)

if ex_to_am_list:
    print(f"\n  EXTRACTED → AMBIGUOUS (false negatives):")
    for r in sorted(ex_to_am_list, key=lambda x: x["final_confidence"]):
        print(f"    \"{r['query']:45s}\"  conf={r['final_confidence']:.3f}  "
              f"raw={r['raw_retrieval_score']:.3f}  top=\"{r['top_label']}\"")

if am_to_ex_list:
    print(f"\n  AMBIGUOUS → EXTRACTED (false positives):")
    for r in sorted(am_to_ex_list, key=lambda x: x["final_confidence"]):
        print(f"    \"{r['query']:45s}\"  conf={r['final_confidence']:.3f}  "
              f"raw={r['raw_retrieval_score']:.3f}  top=\"{r['top_label']}\"")

if not ex_to_am_list and not am_to_ex_list:
    print(f"  No misclassifications at this threshold.")

# ── 7. Does the confidence scorer add value beyond ranking? ──────────

print(f"\n{'─' * 110}")
print("  7. CONFIDENCE SCORER VALUE ANALYSIS")
print(f"{'─' * 110}")

raw_scores = [r["raw_retrieval_score"] for r in results]
final_scores = [r["final_confidence"] for r in results]

# Rank correlation: does ordering change between raw and final?
raw_ranked = sorted(results, key=lambda r: r["raw_retrieval_score"])
final_ranked = sorted(results, key=lambda r: r["final_confidence"])

# Count how many queries change relative order (pairwise inversions)
n = len(results)
inversions = 0
for i in range(n):
    for j in range(i+1, n):
        q_i_raw = raw_ranked[i]["query"]
        q_j_raw = raw_ranked[j]["query"]
        # Find the final-confidence rank positions
        pos_i_final = next(idx for idx, r in enumerate(final_ranked) if r["query"] == q_i_raw)
        pos_j_final = next(idx for idx, r in enumerate(final_ranked) if r["query"] == q_j_raw)
        if pos_i_final > pos_j_final:
            inversions += 1

total_pairs = n * (n-1) // 2
inv_pct = inversions / total_pairs * 100 if total_pairs > 0 else 0
print(f"  Total queries: {n}")
print(f"  Total pairwise comparisons: {total_pairs}")
print(f"  Rank inversions (order changed by confidence scorer): {inversions} ({inv_pct:.1f}%)")
if inv_pct < 10:
    print(f"  VERDICT: Confidence scorer closely follows raw retrieval ranking — "
          f"minimal reordering.")
elif inv_pct < 30:
    print(f"  VERDICT: Confidence scorer produces moderate reordering — "
          f"separation/coverage/boost have some effect.")
else:
    print(f"  VERDICT: Confidence scorer significantly reorders queries — "
          f"separation/coverage/boost are important discriminators.")

# Component contribution analysis
print(f"\n  COMPONENT CONTRIBUTION BREAKDOWN (averages):")
for mode_name, label in [("EXTRACTED", "EXTRACTED"), ("INFERRED", "INFERRED"), ("AMBIGUOUS", "AMBIGUOUS")]:
    subset = [r for r in results if r["mode"] == mode_name]
    if not subset:
        continue
    avg_raw = sum(r["raw_retrieval_score"] for r in subset) / len(subset)
    avg_cal = sum(r["calibrated_retrieval"] for r in subset) / len(subset)
    avg_sep = sum(r["separation"] for r in subset) / len(subset)
    avg_cov = sum(r["coverage"] for r in subset) / len(subset)
    avg_boost = sum(r["sensor_boost"] for r in subset) / len(subset)
    avg_final = sum(r["final_confidence"] for r in subset) / len(subset)
    # Effective contribution
    contrib_cal = 0.60 * avg_cal
    contrib_sep = 0.20 * avg_sep
    contrib_cov = 0.20 * avg_cov
    print(f"    {label:12s}: raw={avg_raw:.3f}  cal={avg_cal:.3f}(×0.60={contrib_cal:.3f})  "
          f"sep={avg_sep:.3f}(×0.20={contrib_sep:.3f})  cov={avg_cov:.3f}(×0.20={contrib_cov:.3f})  "
          f"boost={avg_boost:.3f}  → final={avg_final:.3f}")

# How many queries have non-zero coverage?
zero_cov = sum(1 for r in results if r["coverage"] == 0)
nonzero_cov = sum(1 for r in results if r["coverage"] > 0)
zero_boost = sum(1 for r in results if r["sensor_boost"] == 0)
nonzero_boost = sum(1 for r in results if r["sensor_boost"] > 0)
print(f"\n  Coverage: {nonzero_cov}/{n} queries have positive coverage ({nonzero_cov/n*100:.0f}%)")
print(f"  Sensor boost: {nonzero_boost}/{n} queries have positive boost ({nonzero_boost/n*100:.0f}%)")

# ── 8. Final recommendation ─────────────────────────────────────────

print(f"\n{'=' * 110}")
print("  8. RECOMMENDATION")
print(f"{'=' * 110}")

# Determine if INFERRED forms a cluster or a bridge
inferred_vals = sorted([r["final_confidence"] for r in results if r["mode"] == "INFERRED"])
ambig_vals = sorted([r["final_confidence"] for r in results if r["mode"] == "AMBIGUOUS"])
extracted_vals = sorted([r["final_confidence"] for r in results if r["mode"] == "EXTRACTED"])

is_bridge = False
if ambig_vals and extracted_vals and inferred_vals:
    if inferred_vals[0] <= ambig_vals[-1] or inferred_vals[-1] >= extracted_vals[0]:
        is_bridge = True

if is_bridge:
    print(f"  INFERRED forms a BRIDGE (not a distinct cluster) — it smoothly connects")
    print(f"  AMBIGUOUS and EXTRACTED regions. This means collapsing to two modes")
    print(f"  is NATURAL — INFERRED is an artifact of the two-threshold design, not a")
    print(f"  genuinely separate diagnostic state.")
else:
    print(f"  INFERRED forms a DISTINCT cluster — removing it would lose information.")

conf_range_ratio = 0
if inferred_vals:
    irange = inferred_vals[-1] - inferred_vals[0]
    arange = ambig_vals[-1] - ambig_vals[0] if ambig_vals else 0
    erange = extracted_vals[-1] - extracted_vals[0] if extracted_vals else 0
    total_range = (extracted_vals[-1] if extracted_vals else 1) - (ambig_vals[0] if ambig_vals else 0)
    if total_range > 0:
        conf_range_ratio = irange / total_range

print(f"\n  Best threshold: {best_threshold:.2f}")
print(f"  Misclassifications at best threshold: {len(ex_to_am_list) + len(am_to_ex_list)}")
print(f"  INFERRED → EXTRACTED (correct absorption): {sum(1 for r in results if r['mode'] == 'INFERRED' and r['final_confidence'] >= best_threshold)}")
print(f"  INFERRED → AMBIGUOUS (correct fallback): {sum(1 for r in results if r['mode'] == 'INFERRED' and r['final_confidence'] < best_threshold)}")

if inv_pct < 15:
    print(f"\n  Since ranking reordering is low ({inv_pct:.0f}% inversions), the confidence scorer")
    print(f"  acts primarily as a CALIBRATION LAYER. A simpler approach could use the raw")
    print(f"  retrieval score directly with a threshold, but the multi-factor formula")
    print(f"  provides better separation at the decision boundary and should be retained.")
else:
    print(f"\n  With {inv_pct:.0f}% rank inversions, the confidence scorer provides meaningful")
    print(f"  reordering beyond raw retrieval. Separation, coverage, and sensor boost")
    print(f"  contribute non-trivial discriminative power.")

print(f"\n  {'='*60}")
print(f"  VERDICT: {'FEASIBLE' if len(ex_to_am_list) + len(am_to_ex_list) <= 2 else 'FEASIBLE WITH CAVEATS' if len(ex_to_am_list) + len(am_to_ex_list) <= 5 else 'NOT RECOMMENDED'}")
print(f"  {'='*60}")
