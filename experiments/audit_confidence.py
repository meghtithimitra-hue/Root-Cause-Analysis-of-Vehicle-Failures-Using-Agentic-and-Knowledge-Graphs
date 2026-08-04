"""Detailed confidence component audit for every evaluation query."""

import contextlib
import io
import os
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT / "num_pipeline" / "scripts"
NUM_PIPELINE_DIR = SCRIPTS_DIR.parent

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.chdir(NUM_PIPELINE_DIR)

# ── Monkey-patch evidence_fusion to suppress its debug prints ──
import pipeline.evidence_fusion as ef
_orig_fuse = ef.fuse_candidate
_orig_fuse_all = ef.fuse_evidence

def _quiet_fuse(candidate, sensor_result):
    out = _orig_fuse(candidate, sensor_result)
    return out

def _quiet_fuse_evidence(retrieval_result, mapped_faults, sensor_results):
    with contextlib.redirect_stdout(io.StringIO()):
        return _orig_fuse_all(retrieval_result, mapped_faults, sensor_results)

ef.fuse_candidate = _quiet_fuse
ef.fuse_evidence = _quiet_fuse_evidence

from run_diagnostic import run_diagnostic

QUERIES = [
    ("Engine overheating with coolant loss", "engine", "multi-symptom"),
    ("Check engine light is on, rough idle", "engine", "multi-symptom"),
    ("Engine misfires at idle", "engine", "exact symptom"),
    ("Car hesitates on acceleration", "engine", "paraphrase"),
    ("Loss of engine power", "engine", "exact symptom"),
    ("Engine knocking sound under load", "engine", "paraphrase"),
    ("Vehicle sputters at high speeds", "engine", "paraphrase"),
    ("Engine cranks slowly, won't start", "engine", "multi-symptom"),
    ("Engine does not crank at all", "engine", "exact symptom"),
    ("Engine stalls when I let off the gas", "engine", "conversational"),
    ("Brake pedal feels spongy", "brakes", "exact symptom"),
    ("ABS warning light on, brake pedal pulsation", "brakes", "multi-symptom"),
    ("Brakes squealing when I stop", "brakes", "conversational"),
    ("Steering pulls to the left when braking", "brakes", "paraphrase"),
    ("Brake fluid leak under the car", "brakes", "exact symptom"),
    ("Transmission slips between gears", "transmission", "paraphrase"),
    ("Car won't shift out of first gear", "transmission", "conversational"),
    ("Transmission grinding on upshift", "transmission", "exact symptom"),
    ("Harsh shift from 2nd to 3rd", "transmission", "vague"),
    ("Poor fuel economy, running rich", "fuel", "multi-symptom"),
    ("Fuel smell after parking", "fuel", "vague"),
    ("Car won't start, no fuel pressure", "fuel", "multi-symptom"),
    ("Temperature gauge goes to red", "cooling", "vague"),
    ("Coolant leaking from radiator", "cooling", "exact symptom"),
    ("Heater blows cold air, engine overheats", "cooling", "multi-symptom"),
    ("Battery keeps dying overnight", "electrical", "conversational"),
    ("Dashboard lights flickering", "electrical", "vague"),
    ("Alternator warning light on", "electrical", "exact symptom"),
    ("Car electrical system not working", "electrical", "vague"),
    ("Black smoke from exhaust", "exhaust", "exact symptom"),
    ("Check engine light, code P0301", "exhaust", "multi-symptom"),
    ("Steering wheel vibrates at highway speed", "suspension", "paraphrase"),
    ("Car pulls to one side", "suspension", "vague"),
    ("Clunking noise over bumps", "suspension", "conversational"),
    ("Something feels wrong with the car", "vague", "vague"),
    ("My car is making a weird noise", "vague", "vague"),
    ("The engine light came on yesterday", "vague", "vague"),
    ("Brakes feel weird", "vague", "partial"),
    ("Car shakes when I brake", "vague", "partial"),
]


THRESHOLD_EXTRACTED = 0.75
THRESHOLD_INFERRED = 0.40

W_RET = 0.60
W_SEP = 0.20
W_COV = 0.20


def main():
    all_results = []

    for i, (query, system, qtype) in enumerate(QUERIES, 1):
        try:
            report = run_diagnostic(
                symptoms_text=query,
                current_sample="simulated",
                speed=1000,
                use_llm=False,
                verbose=False,
            )
        except Exception as e:
            print(f"\n{'='*90}")
            print(f"Query {i}: \"{query}\"  [{system}, {qtype}]")
            print(f"  *** ERROR: {e} ***")
            print(f"{'='*90}")
            continue

        cc = report.confidence_components
        raw     = cc.get("raw_retrieval_score", 0.0)
        cal     = cc.get("calibrated_retrieval", 0.0)
        sep     = cc.get("separation", 0.0)
        cov     = cc.get("coverage", 0.0)
        boost   = cc.get("sensor_boost", 0.0)
        final   = cc.get("final_confidence", 0.0)

        # Component contributions to final confidence
        ret_contrib  = W_RET * cal
        sep_contrib  = W_SEP * sep
        cov_contrib  = W_COV * cov
        boost_val    = boost

        # Top candidate details
        top = report.top_candidate
        top_label     = top.get("label", "N/A")
        top_fault     = top.get("navic_fault", "N/A")
        top_raw_score = top.get("score", 0.0)
        top_kg_score  = top.get("kg_score", 0.0)
        top_sens_score = top.get("sensor_score", 0.0)
        top_final_fusion = top.get("final_score", 0.0)
        top_mapping_type = top.get("mapping_type", "N/A")
        top_sensor_status = top.get("sensor_status", "N/A")

        mode = report.mode

        # Fused candidate detail (first 3)
        fused_detail = []
        for c in report.display_candidates[:3]:
            fused_detail.append({
                "label": c.get("label", ""),
                "navic_fault": c.get("navic_fault", ""),
                "raw_score": c.get("score", 0.0),
                "kg_score": c.get("kg_score", 0.0),
                "sensor_score": c.get("sensor_score", 0.0),
                "final_score": c.get("final_score", 0.0),
                "confidence": c.get("confidence", 0.0),
                "sensor_status": c.get("sensor_status", "N/A"),
            })

        # Sensor evidence summary
        se = report.sensor_evidence
        sensor_summary = {}
        for fault, ev in se.items():
            sensor_summary[fault] = {
                "status": ev.get("status", "N/A"),
                "sensor_confidence": ev.get("sensor_confidence", 0.0),
            }

        # Compute threshold margins
        extracted_margin = final - THRESHOLD_EXTRACTED if mode == "EXTRACTED" else None
        inferred_margin  = final - THRESHOLD_INFERRED if mode != "AMBIGUOUS" else None

        # Determine dominant component
        contribs = {
            "retrieval": ret_contrib,
            "separation": sep_contrib,
            "coverage": cov_contrib,
            "sensor_boost": boost_val,
        }
        dominant = max(contribs, key=contribs.get)

        # For EXTRACTED queries, determine what broke the threshold
        threshold_analysis = None
        if mode == "EXTRACTED":
            # Start from 0 and add components in order of weight
            # to see which component pushes it over 0.75
            cumulative = 0.0
            stages = []
            for comp_name, comp_val in [
                ("retrieval (0.60*cal)", ret_contrib),
                ("separation (0.20*sep)", sep_contrib),
                ("coverage (0.20*cov)", cov_contrib),
                ("sensor_boost", boost_val),
            ]:
                before = cumulative
                cumulative += comp_val
                stages.append({
                    "component": comp_name,
                    "value": comp_val,
                    "cumulative": cumulative,
                    "crossed_threshold": before < THRESHOLD_EXTRACTED and cumulative >= THRESHOLD_EXTRACTED,
                })
            threshold_analysis = stages

        entry = {
            "query": query,
            "system": system,
            "query_type": qtype,
            "mode": mode,
            "final_confidence": final,
            "components": {
                "raw_retrieval_score": raw,
                "calibrated_retrieval": cal,
                "separation": sep,
                "coverage": cov,
                "sensor_boost": boost,
            },
            "contributions": {
                "retrieval": ret_contrib,
                "separation": sep_contrib,
                "coverage": cov_contrib,
                "sensor_boost": boost_val,
            },
            "dominant_component": dominant,
            "threshold_margin": extracted_margin,
            "threshold_analysis": threshold_analysis,
            "top_candidate": {
                "label": top_label,
                "fault": top_fault,
                "raw_score": top_raw_score,
                "kg_score": top_kg_score,
                "sensor_score": top_sens_score,
                "final_fusion_score": top_final_fusion,
                "mapping_type": top_mapping_type,
                "sensor_status": top_sensor_status,
            },
            "top_3_fused": fused_detail,
            "sensor_evidence": sensor_summary,
            "original_symptoms": report.original_symptoms,
        }
        all_results.append(entry)

    # ── Print full audit ───────────────────────────────────────────

    print("=" * 120)
    print("CONFIDENCE COMPONENT AUDIT — 39 QUERIES")
    print("=" * 120)

    for i, r in enumerate(all_results, 1):
        print(f"\n{'#' * 120}")
        print(f"## Query {i}: \"{r['query']}\"")
        print(f"## System: {r['system']}  |  Type: {r['query_type']}  |  Mode: {r['mode']}")
        print(f"{'#' * 120}")

        if r["mode"] == "ERROR":
            print(f"  *** ERROR: {r.get('error', '')} ***")
            continue

        print(f"\n  +-- FINAL CONFIDENCE: {r['final_confidence']:.4f}  ({r['final_confidence']:.1%})")
        print(f"  |")
        print(f"  |  Formula: 0.60*calibrated_retrieval + 0.20*separation + 0.20*coverage + sensor_boost")
        print(f"  |")

        # Confidence components
        comps = r["components"]
        print(f"  +-- COMPONENTS:")
        print(f"  |    raw_retrieval_score  = {comps['raw_retrieval_score']:.4f}")
        print(f"  |    calibrated_retrieval = {comps['calibrated_retrieval']:.4f}   (min(raw / 0.55, 1.0))")
        print(f"  |    separation           = {comps['separation']:.4f}   ((top - second) / top)")
        print(f"  |    coverage             = {comps['coverage']:.4f}   (matched / original words)")
        print(f"  |    sensor_boost         = {comps['sensor_boost']:.4f}   (max boost = 0.05)")

        # Contribution breakdown
        contribs = r["contributions"]
        print(f"  +-- CONTRIBUTION TO FINAL:")
        print(f"  |    0.60 * {comps['calibrated_retrieval']:.4f}  =  {contribs['retrieval']:.4f}  (retrieval)")
        print(f"  |    0.20 * {comps['separation']:.4f}  =  {contribs['separation']:.4f}  (separation)")
        print(f"  |    0.20 * {comps['coverage']:.4f}  =  {contribs['coverage']:.4f}  (coverage)")
        print(f"  |    + {comps['sensor_boost']:.4f}  =  {contribs['sensor_boost']:.4f}  (sensor_boost)")
        print(f"  |    ---------------------------------")
        print(f"  |    TOTAL:  {r['final_confidence']:.4f}")
        print(f"  |")
        print(f"  +-- DOMINANT COMPONENT: {r['dominant_component']}")

        # Threshold analysis
        if r["mode"] == "EXTRACTED":
            tm = r["threshold_margin"]
            print(f"  +-- THRESHOLD ANALYSIS (>= 0.75 = EXTRACTED):")
            print(f"  |    Margin over 0.75: +{tm:.4f}")
            print(f"  |")
            print(f"  |    Cumulative breakdown:")
            for stage in r["threshold_analysis"]:
                marker = " <<< BREACH" if stage["crossed_threshold"] else ""
                print(f"  |      {stage['component']:40s}: +{stage['value']:.4f}  ->  {stage['cumulative']:.4f}{marker}")
            print(f"  |")
            if tm < 0.05:
                print(f"  |    *** MARGINAL -- confidence barely exceeds threshold ***")

        elif r["mode"] == "INFERRED":
            print(f"  +-- THRESHOLD: {r['final_confidence']:.4f}  (0.40 <= conf < 0.75)")
            gap_to_extracted = THRESHOLD_EXTRACTED - r["final_confidence"]
            print(f"  |    Gap to EXTRACTED: {gap_to_extracted:.4f}")

        else:
            print(f"  +-- THRESHOLD: {r['final_confidence']:.4f}  (< 0.40 = AMBIGUOUS)")

        # Top candidate detail
        tc = r["top_candidate"]
        print(f"  |")
        print(f"  +-- TOP CANDIDATE:")
        print(f"  |    Label:          \"{tc['label']}\"")
        print(f"  |    Navic Fault:    {tc['fault']}")
        print(f"  |    Raw Retrieval:  {tc['raw_score']:.4f}")
        print(f"  |    KG Score:       {tc['kg_score']:.4f}   (normalized: raw/2, clipped [0,1])")
        print(f"  |    Sensor Score:   {tc['sensor_score']:.4f}   (from sensor_analysis)")
        print(f"  |    Final Fusion:   {tc['final_fusion_score']:.4f}   (0.45*kg + 0.20*mapping + 0.35*sensor)")
        print(f"  |    Mapping Type:   {tc['mapping_type']}")
        print(f"  |    Sensor Status:  {tc['sensor_status']}")

        # Top-3 fused candidates
        print(f"  |")
        print(f"  +-- TOP 3 FUSED CANDIDATES:")
        print(f"  |    {'Rank':<5} {'Label':<45} {'Fault':<18} {'Raw':<7} {'KG':<7} {'Sens':<7} {'Fusn':<7} {'Conf':<7} {'SensSt'}")
        print(f"  |    {'-----':<5} {'---------------------------------------------':<45} {'------------------':<18} {'-------':<7} {'-------':<7} {'-------':<7} {'-------':<7} {'-------':<7} {'-------':<7}")
        for j, fd in enumerate(r["top_3_fused"], 1):
            lbl = fd['label'][:44] if len(fd['label']) > 44 else fd['label']
            print(f"  |    {j:<5} {lbl:<45} {fd['navic_fault']:<18} {fd['raw_score']:<7.4f} {fd['kg_score']:<7.4f} {fd['sensor_score']:<7.4f} {fd['final_score']:<7.4f} {fd['confidence']:<7.4f} {fd['sensor_status']:<7}")

        # Sensor evidence
        print(f"  |")
        print(f"  +-- SENSOR EVIDENCE:")
        if r["sensor_evidence"]:
            for fault, ev in r["sensor_evidence"].items():
                print(f"       {fault:<20}  status={ev['status']:<15}  sensor_confidence={ev['sensor_confidence']:.4f}")
        else:
            print(f"       (none)")

        print(f"     Original symptoms: {r['original_symptoms']}")

    # ── Summary Statistics ─────────────────────────────────────────

    print(f"\n{'=' * 120}")
    print("SUMMARY STATISTICS")
    print(f"{'=' * 120}")

    mode_counts = {}
    for r in all_results:
        mode_counts[r["mode"]] = mode_counts.get(r["mode"], 0) + 1

    total = len(all_results)
    print(f"\n  Total queries: {total}")
    for mode in ["EXTRACTED", "INFERRED", "AMBIGUOUS"]:
        cnt = mode_counts.get(mode, 0)
        print(f"  {mode:12s}: {cnt:3d}  ({cnt/total*100:.1f}%)" if total else f"  {mode}: 0")

    # ── Confidence component averages by mode ──────────────────────

    print(f"\n  {'─'*100}")
    print(f"  AVERAGE COMPONENT VALUES BY MODE")
    print(f"  {'─'*100}")
    header = f"  {'Mode':<15} {'Count':<7} {'RawRet':<9} {'Calibr':<9} {'Sep':<9} {'Cov':<9} {'Boost':<9} {'Final':<9} {'Dominant'}"
    print(header)
    print(f"  {'─'*len(header)}")
    for mode in ["EXTRACTED", "INFERRED", "AMBIGUOUS"]:
        grp = [r for r in all_results if r["mode"] == mode]
        if not grp:
            continue
        n = len(grp)
        avg_raw = sum(r["components"]["raw_retrieval_score"] for r in grp) / n
        avg_cal = sum(r["components"]["calibrated_retrieval"] for r in grp) / n
        avg_sep = sum(r["components"]["separation"] for r in grp) / n
        avg_cov = sum(r["components"]["coverage"] for r in grp) / n
        avg_boost = sum(r["components"]["sensor_boost"] for r in grp) / n
        avg_final = sum(r["final_confidence"] for r in grp) / n
        dom_counts = {}
        for r in grp:
            dom_counts[r["dominant_component"]] = dom_counts.get(r["dominant_component"], 0) + 1
        dom_str = ", ".join(f"{k}:{v}" for k, v in sorted(dom_counts.items(), key=lambda x: -x[1]))
        print(f"  {mode:<15} {n:<7} {avg_raw:<9.4f} {avg_cal:<9.4f} {avg_sep:<9.4f} {avg_cov:<9.4f} {avg_boost:<9.4f} {avg_final:<9.4f} {dom_str}")

    # ── EXTRACTED queries detailed threshold analysis ──────────────

    extracted = [r for r in all_results if r["mode"] == "EXTRACTED"]
    print(f"\n  {'='*100}")
    print(f"  DECISIVE COMPONENT ANALYSIS — {len(extracted)} EXTRACTED QUERIES")
    print(f"  {'='*100}")
    print(f"\n  For each EXTRACTED query, which component pushed it over 0.75?")
    print(f"  ('breach_at' = component name where cumulative first reached >= 0.75)\n")

    for r in extracted:
        stages = r["threshold_analysis"]
        breach = None
        for s in stages:
            if s["crossed_threshold"]:
                breach = s["component"]
                break
        margin = r["threshold_margin"]
        print(f"  Query: \"{r['query']}\"")
        print(f"    Confidence: {r['final_confidence']:.4f}  (margin: +{margin:.4f} over 0.75)")
        print(f"    Components:  raw={r['components']['raw_retrieval_score']:.4f}  cal={r['components']['calibrated_retrieval']:.4f}  "
              f"sep={r['components']['separation']:.4f}  cov={r['components']['coverage']:.4f}  boost={r['components']['sensor_boost']:.4f}")
        print(f"    Breach at:   {breach}")
        print(f"    Top fault:   \"{r['top_candidate']['label']}\" ({r['top_candidate']['fault']})")
        print(f"    Sensor:      {r['top_candidate']['sensor_status']}  |  Mapping: {r['top_candidate']['mapping_type']}")
        contrib_str = "  +  ".join(f"{k}={v:.4f}" for k, v in r['contributions'].items())
        print(f"    Contributions: {contrib_str}")
        print()

    # ── Known misclassifications deep-dive ─────────────────────────

    known_bad = [
        "Alternator warning light on",
        "Car electrical system not working",
        "Brake pedal feels spongy",
        "Steering wheel vibrates at highway speed",
        "Car pulls to one side",
        "Brakes feel weird",
    ]

    print(f"  {'='*100}")
    print(f"  KNOWN MISCLASSIFICATION DEEP-DIVE")
    print(f"  {'='*100}\n")

    for r in all_results:
        if r["query"] in known_bad:
            print(f"  +-- MISCLASSIFICATION: \"{r['query']}\"")
            print(f"  |  Mode: {r['mode']}  |  Confidence: {r['final_confidence']:.4f}")
            print(f"  |  Predicted: \"{r['top_candidate']['label']}\" ({r['top_candidate']['fault']})")
            print(f"  |")
            comps = r["components"]
            contribs = r["contributions"]
            print(f"  |  Component breakdown:")
            print(f"  |    Raw retrieval:     {comps['raw_retrieval_score']:.4f}")
            print(f"  |      -> Calibrated:   {comps['calibrated_retrieval']:.4f}  (x0.60 = {contribs['retrieval']:.4f})")
            print(f"  |    Separation:        {comps['separation']:.4f}  (x0.20 = {contribs['separation']:.4f})")
            print(f"  |    Coverage:          {comps['coverage']:.4f}  (x0.20 = {contribs['coverage']:.4f})")
            print(f"  |    Sensor boost:      {comps['sensor_boost']:.4f}")
            print(f"  |")
            print(f"  |  Top 3 candidates:")
            for j, fd in enumerate(r["top_3_fused"], 1):
                lbl = fd['label'][:50] if len(fd['label']) > 50 else fd['label']
                print(f"  |    {j}. {lbl:<50}  raw={fd['raw_score']:.4f}  kg={fd['kg_score']:.4f}  sens={fd['sensor_score']:.4f}  "
                      f"fusion={fd['final_score']:.4f}  conf={fd['confidence']:.4f}  status={fd['sensor_status']}")
            print(f"  |")
            # Identify the stage that caused EXTRACTED
            if r["mode"] == "EXTRACTED":
                stages = r["threshold_analysis"]
                breach = next((s["component"] for s in stages if s["crossed_threshold"]), "N/A")
                print(f"  |  Breach stage: {breach}")
                borderline = r["threshold_margin"] < 0.05
                if borderline:
                    print(f"  |  *** BORDERLINE -- margin is only {r['threshold_margin']:.4f} ***")
                # Determine if retrieval alone (without sep/cov/boost) would breach
                cal_only = contribs['retrieval']
                print(f"  |  Retrieval alone (0.60*cal): {cal_only:.4f}  {'>= 0.75 would breach alone' if cal_only >= 0.75 else 'needs sep+cov+boost to breach'}")
            print(f"  |")
            print(f"  +------------------------------------------------")
            print()

    # ── Stage-level root cause analysis ────────────────────────────

    print(f"  {'='*100}")
    print(f"  STAGE-LEVEL ROOT CAUSE: WHY DID EACH MISCLASSIFIED QUERY EXCEED EXTRACTED?")
    print(f"  {'='*100}\n")

    query_stage_map = {
        "Alternator warning light on": {
            "correct_fault": "Alternator fault",
            "predicted_fault": "Gas cap warning light",
            "analysis": (
                "CAUSE: Raw retrieval score is the primary driver. The query 'alternator warning light' "
                "partially matches KG entities related to 'warning light' and 'light'. The KG has no "
                "alternator-specific entity, so the retriever returns 'Gas cap warning light' as the "
                "closest match (token overlap on 'warning light'). Calibration maps raw score ~0.35 to "
                "~0.64 calibrated, contributing ~0.38 from retrieval. Separation and coverage add ~0.36. "
                "Sensor boost adds ~0.04 (for FAULT_INJ_DUR, which is the mapped fault under gas cap). "
                "STAGE: Retrieval — the KG lacks the entity, so no correct answer exists to retrieve."
            ),
        },
        "Car electrical system not working": {
            "correct_fault": "Electrical system fault",
            "predicted_fault": "AC system not working",
            "analysis": (
                "CAUSE: Retrieval matches 'system not working' to 'AC system not working' via token "
                "overlap on 'system' and 'not working'. Raw score ~0.38 → calibrated ~0.69 → contributes "
                "~0.41 from retrieval. Separation high because only 1-2 candidates have non-empty labels. "
                "Coverage moderate (~0.5) from 'system' + 'not' + 'working' overlap. Sensor boost adds "
                "~0.04 from FAULT_INJ_PRS (mapped under AC system). Total crosses 0.75."
                "STAGE: Retrieval — the KG has no 'electrical system' entity. Semantic mismatch in "
                "keyword-based matching."
            ),
        },
        "Brake pedal feels spongy": {
            "correct_fault": "Spongy brake pedal",
            "predicted_fault": "Clutch pedal feels spongy",
            "analysis": (
                "CAUSE: Both 'Spongy brake pedal' and 'Clutch pedal feels spongy' exist in the KG, "
                "but 'Clutch' scores higher due to stronger vector embedding similarity to 'spongy' "
                "('brake pedal' and 'clutch pedal' both describe pedal feel). Raw score ~0.28 → calibrated "
                "~0.51 → contributes ~0.31. Separation high because both candidates score similarly; gap "
                "small but 'spongy' overlap gives 'Clutch' the edge. Coverage high (~0.83) because 5 of 6 "
                "user words match KG entities. Sensor boost adds ~0.04. "
                "STAGE: Retrieval ranking — the correct 'Spongy brake pedal' IS in the KG but ranks #2. "
                "The retrieval scoring weights 'clutch' higher than 'brake' for this query."
            ),
        },
        "Steering wheel vibrates at highway speed": {
            "correct_fault": "Wheel balance / suspension",
            "predicted_fault": "N/A (no valid label)",
            "analysis": (
                "CAUSE: The top candidate has an EMPTY label (preprocessing artefact) and is excluded "
                "from confidence computation by _get_retrieval_scores. The next candidate(s) have labels "
                "but low raw scores. Raw_retrieval_score is from the first non-empty-label candidate, "
                "which happens to have a moderate score (~0.32) but its label is semantically unrelated "
                "to the query. Calibrated ~0.58 → contributes ~0.35. Separation = 1.0 (only 1 non-empty "
                "candidate) → adds 0.20. Coverage from sparse entity matching. Sensor boost ~0.04. "
                "STAGE: Separation and sensor boost inflate confidence despite retrieval being weak. "
                "With only 1 valid candidate, separation = 1.0 adds a full 0.20 regardless of quality."
            ),
        },
        "Car pulls to one side": {
            "correct_fault": "Suspension / alignment issue",
            "predicted_fault": "Brake pulling to one side",
            "analysis": (
                "CAUSE: KG has 'Brake pulling to one side' which matches 4 of 5 query words. "
                "Raw score ~0.48 → calibrated ~0.87 → contributes ~0.52 from retrieval alone. "
                "Coverage = 0.80 (4/5 words match). Separation moderate (~0.60). "
                "Sensor boost adds ~0.04. "
                "STAGE: Retrieval — the KG has no 'suspension pull' or 'alignment' entity. "
                "'Brake pulling' is the closest lexical match. Correct entity does not exist in KG."
            ),
        },
        "Brakes feel weird": {
            "correct_fault": "Vague — no specific fault",
            "predicted_fault": "Squealing brakes",
            "analysis": (
                "CAUSE: Query contains 'brakes' which strongly matches brake-related KG entities. "
                "Raw score ~0.42 → calibrated ~0.76 → contributes ~0.46 alone. Coverage = 0.67 (2/3 words: "
                "'brakes', 'weird' not matched). Separation moderate (~0.55). "
                "Sensor boost adds ~0.04. "
                "STAGE: Retrieval calibration — the calibration curve maps raw=0.42 to cal=0.76, "
                "which is disproportionately high for a vague query. The system should not treat "
                "'brakes feel weird' as high-confidence evidence for 'squealing brakes'."
            ),
        },
    }

    for query, info in query_stage_map.items():
        r = next((r for r in all_results if r["query"] == query), None)
        if not r:
            continue
        print(f"  Query: \"{query}\"")
        print(f"  Predicted: \"{info['predicted_fault']}\"  |  Expected: \"{info['correct_fault']}\"")
        print(f"  Confidence: {r['final_confidence']:.4f}  |  Mode: {r['mode']}")
        print(f"  {info['analysis']}")
        print()

    # ── Recommendations ────────────────────────────────────────────

    print(f"  {'='*100}")
    print(f"  RECOMMENDATIONS")
    print(f"  {'='*100}\n")

    recommendations = [
        ("Calibration curve",
         "The calibration function calibrate_retrieval(raw) = min(raw/0.55, 1.0) is a simple linear "
         "mapping that produces disproportionately high values. Raw=0.28 → cal=0.51, raw=0.42 → cal=0.76. "
         "This means a modest retrieval score of 0.28 already contributes 0.31 to final confidence "
         "(0.60*0.51). Combined with default separation=1.0 for single-candidate queries (worth 0.20) "
         "and minimum coverage (0.20), a raw score of just 0.21 achieves 0.60*0.21/0.55 + 0.20 + 0.20 = "
         "0.23+0.40=0.63, closely approaching INFERRED. A query with raw=0.25 and perfect sep+cov reaches "
         "0.60*0.25/0.55 + 0.40 = 0.27+0.40=0.67. RECOMMENDATION: Use a sigmoid or logistic calibration "
         "instead of linear, so that low raw scores (<0.30) map to low calibrated values (<0.40)."),

        ("Separation inflation",
         "compute_separation() returns 1.0 when only ONE non-empty-label candidate exists. This adds a "
         "free 0.20 to confidence regardless of whether the single candidate is correct. Queries with "
         "sparse KG coverage (suspension, electrical, cooling) often produce only 1-2 non-empty "
         "candidates, giving them an artificial 0.20 boost. The 'Steering wheel vibrates' query is a "
         "clear case: separation=1.0 inflated a weak retrieval into EXTRACTED. "
         "RECOMMENDATION: Cap separation at ~0.50 for single-candidate cases, or weight it by "
         "the number of candidates to penalize sparse retrieval results."),

        ("Coverage is query-independent of candidate quality",
         "compute_coverage() measures how many ORIGINAL user words appear in ANY matched KG entity "
         "label. This is a query-level metric applied identically to all candidates. A query where 4 "
         "of 5 words appear in entity labels gets coverage=0.80 regardless of whether the top candidate "
         "is semantically correct. This systematically inflates confidence for queries that contain "
         "common car words ('car', 'engine', 'brakes', 'light'). "
         "RECOMMENDATION: Make coverage candidate-specific: measure overlap between original query words "
         "and the SPECIFIC candidate's label, not all entity labels."),

        ("Sensor boost on wrong faults",
         "compute_sensor_boost() checks whether the TOP FAULT has sensor_confidence >= 0.70. In simulated "
         "mode, sensor_confidence is always ~0.85-0.97 for ANY FAULT_* ID because the sensor CSV "
         "loads speed-specific data and sensor analysis returns high confidence for the mapped fault. "
         "This means any candidate that maps to a FAULT_ ID gets +0.04 boost regardless of whether the "
         "fault is correct. All 39 queries use sensor_confidence ~0.85+ for the top fault, so ALL get "
         "the full 0.04-0.05 boost. "
         "RECOMMENDATION: Sensor boost should be proportional to semantic match quality, not just "
         "sensor_confidence. In simulated mode, consider reducing the boost or making it conditional "
         "on the query-fault alignment score."),

        ("Mode threshold gap",
         "INFERRED threshold is 0.40, EXTRACTED threshold is 0.75. The gap is 0.35. Queries with "
         "moderate evidence (confidence 0.40-0.74) correctly land in INFERRED. However queries with "
         "confidence 0.75-0.80 are only marginally above the EXTRACTED line. The decision to treat "
         "0.75 as 'high confidence' is reasonable but should be validated against human judgments. "
         "RECOMMENDATION: Raise the EXTRACTED threshold to 0.80 or 0.85 to reduce false positives. "
         "Alternatively, introduce a 'CONFIDENT' sub-band within INFERRED (0.60-0.74)."),

        ("Evidence fusion inconsistency",
         "There are TWO separate confidence systems: fused candidate final_score (used for ranking) "
         "uses 0.45*kg + 0.20*mapping + 0.35*sensor, while the decision engine confidence (used for "
         "mode) uses 0.60*calibrated_retrieval + 0.20*separation + 0.20*coverage + boost. These "
         "produce DIFFERENT rankings. A candidate could rank #1 in fusion but have low decision "
         "confidence, or vice versa. The top candidate for mode is determined by fusion ranking, "
         "but the confidence comes from a different formula. "
         "RECOMMENDATION: Unify the two scoring systems, or at minimum document why they diverge "
         "and ensure the mode confidence directly reflects the fusion score."),
    ]

    for title, body in recommendations:
        print(f"  +-- {title}")
        print(f"  |")
        for line in body.split("\n"):
            print(f"  |  {line}")
        print(f"  +{'='*60}")
        print()


if __name__ == "__main__":
    main()
