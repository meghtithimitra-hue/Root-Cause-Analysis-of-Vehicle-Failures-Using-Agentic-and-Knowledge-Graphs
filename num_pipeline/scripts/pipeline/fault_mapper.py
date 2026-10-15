"""
fault_mapper.py

Maps Knowledge Graph labels and categories to NavicEngine fault
datasets.  Acts as the bridge between hybrid retrieval (KG) and
sensor validation (numerical profiles).

Design decisions
----------------
1.  **Label mapping first, category mapping second.**
    Label matching uses stemmed-token overlap (Jaccard) instead of
    exact string equality, so "Engine cranks slowly" matches the
    anchor "slow cranking", "Engine misfire" matches "engine
    misfires", and "Poor engine power" matches "loss of engine
    power".

2.  **Minimal, self-contained stemmer.**
    Covers the common English suffixes found in automotive
    diagnostic labels (-ing, -s, -ly, -ed, -es, -tion).  No
    external NLP dependency required.

3.  **Categories taken from the actual KG.**
    KG_TO_NAVIC keys ("Engine Components", "Fuel System", etc.)
    are the real category strings produced by hybrid_retrieval.py.
    The ".Md" metadata suffix is stripped automatically.

4.  **One KG label → at most one Navic fault per label match.**
    When a label matches multiple anchors the highest-scoring
    anchor wins.  Duplicates across candidates are prevented
    with a seen-set keyed on (label, fault).
"""

import re
from typing import Dict, List, Tuple

# ==========================================================
# Configuration
# ==========================================================

_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "in", "on", "of", "and", "or",
    "when", "it", "my", "to", "for", "not", "at", "by", "with",
})

# Generic stems that appear across many automotive fault types.
# Excluded from Jaccard so they cannot inflate scores for unrelated
# label pairs (e.g. "Coolant leaks under the vehicle" vs "visible
# fuel leak under vehicle").
# Minimum token-overlap score for a label match to count
_MATCH_THRESHOLD = 0.15

# ==========================================================
# Minimal English Stemmer
# ==========================================================


def _stem(word):
    """
    Strip common suffixes.  Good enough for the automotive
    diagnostic vocabulary without pulling in NLTK.
    """
    if len(word) <= 3:
        return word
    if word.endswith("ing"):
        base = word[:-3]
        return base
    if word.endswith("tion") and len(word) > 6:
        return word[:-4]
    if word.endswith("ly") and len(word) > 4:
        return word[:-2]
    if word.endswith("ed") and len(word) > 4:
        return word[:-2]
    if word.endswith("s") and len(word) > 3:
        return word[:-1]
    if word.endswith("es") and len(word) > 4:
        return word[:-2]
    return word

# ==========================================================
# Stem Sets for Domain-Aware Matching
# ==========================================================

# Generic stems that appear across many automotive fault types.
# Excluded from Jaccard so they cannot inflate scores for unrelated
# label pairs (e.g. "Coolant leaks under the vehicle" vs "visible
# fuel leak under vehicle").
_GENERIC_STEMS = frozenset({
    _stem(w) for w in [
        "vehicle", "system", "under", "from", "leak", "leaks",
        "warning", "visible", "noise",
    ]
})

# Domain-specific stems that anchor a label to a particular vehicle
# subsystem.  When BOTH sides carry domain anchors, at least one
# must overlap or the match is rejected — this prevents matching
# across unrelated domains (fuel ↔ coolant, brake ↔ battery, etc.).
_DOMAIN_ANCHOR_STEMS = frozenset(
    {_stem(w) for w in [
        "engine", "crank", "cranking", "misfire", "misfires",
        "knock", "knocking", "power", "performance", "stall",
        "stalling", "start", "starting", "hesitate", "sputter",
    ]}
    | {_stem(w) for w in ["fuel", "consumption", "economy"]}
    | {_stem(w) for w in [
        "coolant", "oil", "brake", "battery", "transmission",
    ]}
    | {_stem(w) for w in ["exhaust", "smoke"]}
    | {_stem(w) for w in ["inject", "injection", "timing", "pressure"]}
)

# ==========================================================
# Normalize Label
# ==========================================================


def normalize_label(label):
    """
    Lowercase, strip punctuation, collapse whitespace.

    >>> normalize_label("Engine knocking/pinging")
    'engine knocking pinging'
    """
    text = label.lower().strip()
    text = re.sub(r"[/\\]", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ==========================================================
# Token Match Score
# ==========================================================


def compute_token_match(kg_label, anchor_label):
    """
    Fuzzy similarity between two labels using stemmed tokens.

    Returns
    -------
    float in [0, 1]
        1.0  — exact normalised match
        0.0+ — domain-aware Jaccard index of stemmed token sets

    Generic stems (vehicle, leak, under …) are excluded from the
    Jaccard calculation to prevent false-positive inflation.
    When both sides carry domain-specific anchor stems (coolant,
    fuel, brake …) at least one must overlap or the match is
    rejected, preventing cross-domain false matches.
    """
    kg_norm = normalize_label(kg_label)
    anchor_norm = normalize_label(anchor_label)

    if kg_norm == anchor_norm:
        return 1.0

    kg_stems = {
        _stem(t) for t in kg_norm.split()
        if t not in _STOP_WORDS and len(t) > 1
    }
    anchor_stems = {
        _stem(t) for t in anchor_norm.split()
        if t not in _STOP_WORDS and len(t) > 1
    }

    if not kg_stems or not anchor_stems:
        return 0.0

    # Filter out generic stems that appear across many fault types
    kg_filtered = kg_stems - _GENERIC_STEMS
    anchor_filtered = anchor_stems - _GENERIC_STEMS

    if not kg_filtered or not anchor_filtered:
        return 0.0

    # Domain anchor check: if both sides carry domain stems,
    # at least one must overlap or reject the match
    kg_anchors = kg_filtered & _DOMAIN_ANCHOR_STEMS
    anchor_anchors = anchor_filtered & _DOMAIN_ANCHOR_STEMS
    if kg_anchors and anchor_anchors and not (kg_anchors & anchor_anchors):
        return 0.0

    intersection = kg_filtered & anchor_filtered
    if not intersection:
        return 0.0

    union = kg_filtered | anchor_filtered
    return len(intersection) / len(union)

# ==========================================================
# Label → NavicEngine Fault Mapping
# ==========================================================
#
# Keys are canonical symptom phrases.  Fuzzy token matching
# means the KG label does not need to be an exact string of
# any key — shared stems are enough.

LABEL_TO_NAVIC = {

    # ── Injection Duration (performance & starting) ──

    "poor engine performance":          ["FAULT_INJ_DUR"],
    "decreased engine performance":     ["FAULT_INJ_DUR"],
    "engine performance issues":        ["FAULT_INJ_DUR"],
    "engine hesitates on acceleration": ["FAULT_INJ_DUR"],
    "engine stalls":                    ["FAULT_INJ_DUR"],
    "engine stalls at high speeds":     ["FAULT_INJ_DUR"],
    "stalling at idle":                 ["FAULT_INJ_DUR"],
    "difficulty starting engine":       ["FAULT_INJ_DUR"],
    "engine cranks slowly":             ["FAULT_INJ_DUR"],
    "engine does not crank":            ["FAULT_INJ_DUR"],
    "engine wont start":                ["FAULT_INJ_DUR"],
    "hard starting in cold weather":    ["FAULT_INJ_DUR"],
    "engine sputters at high speeds":   ["FAULT_INJ_DUR"],
    "slow cranking":                    ["FAULT_INJ_DUR"],

    # ── Injection Pressure (fuel & misfire) ──

    "high fuel consumption":     ["FAULT_INJ_PRS"],
    "poor fuel economy":         ["FAULT_INJ_PRS"],
    "engine misfires":           ["FAULT_INJ_PRS"],
    "loss of engine power":      ["FAULT_INJ_PRS"],
    "loss of power":             ["FAULT_INJ_PRS"],
    "engine power loss":         ["FAULT_INJ_PRS"],
    "poor engine power":         ["FAULT_INJ_PRS"],
    "fuel odor from vehicle":    ["FAULT_INJ_PRS"],
    "visible fuel leak under vehicle": ["FAULT_INJ_PRS"],

    # ── Start of Injection (knocking & timing) ──

    "engine knocking":           ["FAULT_SOI"],
    "engine knocking/pinging":   ["FAULT_SOI"],
    "engine knocking noise":     ["FAULT_SOI"],
    "poor engine timing":        ["FAULT_SOI"],
    "black smoke from exhaust":  ["FAULT_SOI"],
    "blue smoke from exhaust":   ["FAULT_SOI"],
}

# ==========================================================
# Category → NavicEngine Fault Mapping  (fallback)
# ==========================================================
#
# Keys are the exact category strings produced by
# hybrid_retrieval.py (derived from the KG node metadata).
# The ".Md" suffix is stripped automatically before lookup.

CATEGORY_MAP = {

    "Engine Components": [
        {"fault": "FAULT_INJ_DUR", "confidence": 0.85},
        {"fault": "FAULT_INJ_PRS", "confidence": 0.75},
        {"fault": "FAULT_SOI",    "confidence": 0.70},
    ],

    "Engine Compartment": [
        {"fault": "FAULT_INJ_DUR", "confidence": 0.80},
        {"fault": "FAULT_INJ_PRS", "confidence": 0.70},
        {"fault": "FAULT_SOI",    "confidence": 0.65},
    ],

    "Fuel System": [
        {"fault": "FAULT_INJ_PRS", "confidence": 0.90},
        {"fault": "FAULT_INJ_DUR", "confidence": 0.70},
    ],

    "Emissions System": [
        {"fault": "FAULT_INJ_PRS", "confidence": 0.85},
        {"fault": "FAULT_SOI",    "confidence": 0.75},
    ],

    "Cooling System":        [],
    "Liquid Systems":        [],
    "Transmission":          [],
    "Drivetrain":            [],
    "ABS System":            [],
    "Electrical System":     [],
    "Steering":              [],
    "Wheels & Tires":        [],
    "Air Conditioning System": [],
}

# ==========================================================
# Normalize Category
# ==========================================================


def _normalize_category(category):
    """
    Strip the '.Md' metadata suffix and whitespace.

    >>> _normalize_category("Engine Components.Md")
    'Engine Components'
    """
    return category.split(".")[0].strip()

# ==========================================================
# Map by Label  (fuzzy — highest priority)
# ==========================================================


def map_by_label(candidate):
    """
    Match a KG candidate's label against every anchor in
    LABEL_TO_NAVIC using stemmed token overlap.

    Returns
    -------
    list of (fault_id, confidence) tuples
        Empty list if nothing matched.
    """
    label = candidate.get("label", "")

    best_score = 0.0
    best_faults: List[str] = []

    for anchor, faults in LABEL_TO_NAVIC.items():
        score = compute_token_match(label, anchor)
        if score > best_score and score >= _MATCH_THRESHOLD:
            best_score = score
            best_faults = faults

    if best_faults:
        return [(f, 1.0) for f in best_faults]

    return []

# ==========================================================
# Map by Category  (fallback)
# ==========================================================


def map_by_category(candidate):
    """
    Look up the candidate's KG category in CATEGORY_MAP.
    Handles '.Md' variants and partial name overlap.

    Returns
    -------
    list of (fault_id, confidence) tuples
        Empty list when the category has no Navic mapping.
    """
    raw_category = candidate.get("category", "").strip()
    cat_clean = _normalize_category(raw_category)

    # Exact match on the full category string
    if raw_category in CATEGORY_MAP:
        entries = CATEGORY_MAP[raw_category]
        return [(e["fault"], e["confidence"]) for e in entries]

    # Exact match after .Md stripping
    if cat_clean in CATEGORY_MAP:
        entries = CATEGORY_MAP[cat_clean]
        return [(e["fault"], e["confidence"]) for e in entries]

    # Partial overlap (handles future category renames)
    cat_lower = cat_clean.lower()
    for known_cat, entries in CATEGORY_MAP.items():
        known_lower = known_cat.lower()
        if cat_lower in known_lower or known_lower in cat_lower:
            return [(e["fault"], e["confidence"]) for e in entries]

    return []

# ==========================================================
# Main Entry Point
# ==========================================================


def map_faults(retrieval_result: Dict) -> List[Dict]:
    """
    Map retrieval candidates to NavicEngine faults.

    Stages
    ------
    1. Label mapping  (fuzzy token match) — highest priority
    2. Category mapping                   — fallback

    Each mapped entry carries every field expected by
    ``sensor_analysis.py`` and ``evidence_fusion.py``:

        kg_category  label  navic_fault  kg_score
        source  node_type  mapping_type  mapping_confidence

    Parameters
    ----------
    retrieval_result : dict
        Output of ``hybrid_retrieve()``.  Must contain ``candidates``.

    Returns
    -------
    list[dict]
    """
    mapped = []
    seen: set = set()

    for candidate in retrieval_result["candidates"]:
        label    = candidate.get("label", "")
        category = candidate.get("category", "")
        kg_score = candidate["score"]

        # -------------------------------------------
        # Stage 1 — Label mapping (fuzzy)
        # -------------------------------------------

        label_matches = map_by_label(candidate)

        if label_matches:
            mapping_type = "label"
            matches = label_matches
        else:
            # ---------------------------------------
            # Stage 2 — Category mapping (fallback)
            # ---------------------------------------

            cat_matches = map_by_category(candidate)

            if cat_matches:
                mapping_type = "category"
                matches = cat_matches
            else:
                continue

        # -------------------------------------------
        # Build mapped entries
        # -------------------------------------------

        for fault, base_confidence in matches:

            key = (label, fault)

            if key in seen:
                continue
            seen.add(key)

            if mapping_type == "label":
                final_confidence = 1.0
            else:
                final_confidence = base_confidence

            # Small reward for high-scoring KG candidates
            if kg_score >= 2.0:
                final_confidence += 0.05
            elif kg_score >= 1.5:
                final_confidence += 0.02

            final_confidence = min(final_confidence, 1.0)

            mapped.append({
                "kg_category":        category,
                "label":              label,
                "navic_fault":        fault,
                "kg_score":           kg_score,
                "source":             candidate["source"],
                "node_type":          candidate["node_type"],
                "mapping_type":       mapping_type,
                "mapping_confidence": round(final_confidence, 3),
            })

    mapped.sort(key=lambda x: x["kg_score"], reverse=True)
    return mapped

# ==========================================================
# Pretty Print
# ==========================================================


def print_mapping(mapped):

    print("\n")
    print("=" * 70)
    print("KG -> NAVIC MAPPING")
    print("=" * 70)
    print(
        f"{'Label':<30}"
        f"{'Fault':<18}"
        f"{'Type':<10}"
        f"{'Conf':<8}"
        f"{'KG Score'}"
    )
    print("-" * 70)

    for m in mapped:
        print(
            f"{m['label']:<30}"
            f"{m['navic_fault']:<18}"
            f"{m['mapping_type']:<10}"
            f"{m['mapping_confidence']:<8.2f}"
            f"{m['kg_score']:.2f}"
        )

# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":

    retrieval = {
        "candidates": [
            {
                "category": "Engine Components",
                "label": "Poor engine performance",
                "node_type": "Symptom",
                "score": 1.82,
                "source": "all",
            },
            {
                "category": "Transmission",
                "label": "Transmission slipping",
                "node_type": "Symptom",
                "score": 1.54,
                "source": "all",
            },
            {
                "category": "Engine Components",
                "label": "Engine cranks slowly",
                "node_type": "Symptom",
                "score": 1.40,
                "source": "vector+graph",
            },
            {
                "category": "Engine Components",
                "label": "Engine misfire",
                "node_type": "Symptom",
                "score": 1.30,
                "source": "vector",
            },
            {
                "category": "Engine Components",
                "label": "Engine knocking noise",
                "node_type": "Symptom",
                "score": 1.20,
                "source": "graph",
            },
            {
                "category": "Fuel System",
                "label": "Poor fuel economy",
                "node_type": "Symptom",
                "score": 1.15,
                "source": "community",
            },
            {
                "category": "Engine Components",
                "label": "Boost Pressure",
                "node_type": "Sensor",
                "score": 0.90,
                "source": "vector",
            },
        ]
    }

    result = map_faults(retrieval)
    print_mapping(result)
