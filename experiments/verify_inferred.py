"""Verify the top INFERRED queries are stable."""

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

queries = [
    "Brake pedal goes to floor",
    "Radiator fan not spinning",
    "Smoke from tailpipe",
    "Power windows not working",
]

for q in queries:
    r = run_diagnostic(symptoms_text=q, current_sample="simulated", speed=1000, use_llm=False)
    cc = r.confidence_components
    cal = cc["calibrated_retrieval"]
    sep = cc["separation"]
    cov = cc["coverage"]
    boost = cc["sensor_boost"]
    rc = 0.60 * cal
    sc = 0.20 * sep
    cc_c = 0.20 * cov
    bc = boost
    cum = [rc, rc+sc, rc+sc+cc_c, rc+sc+cc_c+bc]
    labels = ["retrieval", "separation", "coverage", "sensor_boost"]
    breach = None
    for c, l in zip(cum, labels):
        if c >= 0.75:
            breach = l
            break
    print(f"Query: {q}")
    print(f"  Mode: {r.mode}  Conf: {r.confidence:.4f}")
    print(f'  Top: "{r.top_candidate.get("label","")}" ({r.top_candidate.get("navic_fault","")})')
    print(f"  Components: raw={cc['raw_retrieval_score']:.4f} cal={cal:.4f} sep={sep:.4f} cov={cov:.4f} boost={boost:.4f}")
    print(f"  Contributions: ret={rc:.4f} sep={sc:.4f} cov={cc_c:.4f} boost={bc:.4f}")
    print(f"  Cumulative: {[f'{c:.4f}' for c in cum]}")
    print(f"  Breach stage: {breach}")
    print()
