import json
import re

from collections import defaultdict
from difflib import SequenceMatcher

try:
    from rapidfuzz import process, fuzz
    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False


# ==========================================================
# Paths
# ==========================================================

GRAPH_PATH = "data/processed/hierarchical_graph.json"


# ==========================================================
# Minimal English Stemmer
# ==========================================================


def _stem(word):
    if len(word) <= 3:
        return word
    if word.endswith("ation") and len(word) > 6:
        return word[:-3]
    if word.endswith("ing"):
        return word[:-3]
    if word.endswith("tion") and len(word) > 6:
        return word[:-3]
    if word.endswith("ly") and len(word) > 4:
        return word[:-2]
    if word.endswith("ed") and len(word) > 4:
        return word[:-2]
    if word.endswith("es") and len(word) > 5:
        return word[:-2]
    if word.endswith("s") and len(word) > 3:
        return word[:-1]
    if word.endswith("ness") and len(word) > 5:
        return word[:-4]
    if word.endswith("ment") and len(word) > 5:
        return word[:-4]
    if word.endswith("ance") and len(word) > 5:
        return word[:-4]
    if word.endswith("ence") and len(word) > 5:
        return word[:-4]
    return word


# ==========================================================
# Lazy Loaded Objects
# ==========================================================

_graph_nodes = None

_graph_edges = None

_vocab_words = None

_label_lookup = None

_category_index = None

_community_index = None

_neighbor_index = None

_subcategory_siblings = None

_word_document_freq = None

_auto_synonyms = None

_label_stems = None


# ==========================================================
# Manual Synonyms
# ==========================================================

SYNONYM_TABLE = {

    "cel": "check engine light",
    "check engine light": "check engine light",
    "check engine light on": "check engine light on",

    "brakes": "braking system",
    "braking": "braking system",

    "ac": "air conditioning",
    "a/c": "air conditioning",

    "trans": "transmission",
    "gearbox": "transmission",

    "abs": "abs system",

    "won't start": "engine wont start",
    "wont start": "engine wont start",
    "won t start": "engine wont start",
    "hard start": "difficulty starting engine",
    "starts hard": "difficulty starting engine",
    "hard starting": "difficulty starting engine",
    "difficulty starting": "difficulty starting engine",

    "overheating": "engine overheating",
    "overheat": "engine overheating",
    "engine hot": "engine overheating",
    "running hot": "engine overheating",

    "hesitation": "engine hesitates on acceleration",
    "hesitates": "engine hesitates on acceleration",
    "hesitating": "engine hesitates on acceleration",
    "bogging down": "engine hesitates on acceleration",

    "rough idle": "rough idle",
    "erratic idle": "erratic idle speed",
    "idle rough": "rough idle",
    "idle fluctuates": "erratic idle speed",

    "low engine power": "loss of engine power",
    "loses power": "loss of engine power",
    "lost power": "loss of engine power",
    "power loss": "loss of engine power",
    "lack of power": "loss of engine power",
    "no power": "loss of engine power",
    "weak acceleration": "loss of engine power",

    "consumes too much fuel": "high fuel consumption",
    "uses too much fuel": "high fuel consumption",
    "burns too much fuel": "high fuel consumption",
    "gas guzzler": "high fuel consumption",
    "poor gas mileage": "poor fuel economy",
    "bad fuel economy": "poor fuel economy",
    "fuel hungry": "high fuel consumption",

    "misfire": "engine misfires",
    "misfires": "engine misfires",
    "misfiring": "engine misfires",
    "engine misfire": "engine misfires",
    "engine misfires": "engine misfires",

    "knocking": "engine knocking",
    "knock": "engine knocking",
    "pinging": "engine knocking/pinging",

    "coolant leak": "coolant leak",
    "coolant leaks": "coolant leaks under the vehicle",
    "coolant loss": "coolant leak",
    "antifreeze leak": "coolant leak",

    "smoke": "black smoke from exhaust",
    "exhaust smoke": "black smoke from exhaust",
    "blue smoke": "blue smoke from exhaust",
    "black smoke": "black smoke from exhaust",

    "stalls": "engine stalls",
    "stalling": "engine stalls",
    "stall": "engine stalls",
    "engine stall": "engine stalls",
    "dies": "engine stalls",
    "shuts off": "engine stalls",

    "cranks slowly": "engine cranks slowly",
    "slow crank": "slow cranking",
    "slow cranking": "slow cranking",
    "weak crank": "engine cranks slowly",
    "clicking no start": "engine does not crank",
    "no crank": "engine does not crank",
    "wont crank": "engine does not crank",

    "sputters": "engine sputters at high speeds",
    "sputtering": "engine sputters at high speeds",

    "tire pressure": "low tire pressure",
    "tyre pressure": "low tire pressure",
    "flat tire": "low tire pressure",
}


# ==========================================================
# Node Type Weight
# ==========================================================

NODE_TYPE_WEIGHT = {

    "Category": 1.0,

    "Subcategory": 1.2,

    "Symptom": 1.8,

    "DiagnosisStep": 0.8,

    "Cause": 1.5

}


# ==========================================================
# Intent Regex
# ==========================================================

_ERROR_CODE_PATTERN = re.compile(

    r"\b(error|code|dtc|fault)\s*[:#]?\s*([a-z]?\d{2,5})\b",

    re.IGNORECASE

)

_BARE_CODE_PATTERN = re.compile(

    r"\b[pPbBcCuU]\d{4}\b"

)

_QUESTION_WORDS = (

    "why",

    "how",

    "what",

    "when",

    "is my",

    "should i",

    "can i"

)


# ==========================================================
# Load Graph
# ==========================================================

def _load_graph():

    global _graph_nodes
    global _graph_edges
    global _vocab_words
    global _label_lookup
    global _category_index
    global _community_index
    global _neighbor_index
    global _subcategory_siblings
    global _word_document_freq
    global _auto_synonyms
    global _label_stems

    if _graph_nodes is not None:
        return

    with open(GRAPH_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # ------------------------------------------------------
    # Store nodes
    # ------------------------------------------------------

    nodes = {
        n["id"]: n
        for n in data["nodes"]
    }

    _graph_nodes = nodes
    _graph_edges = data.get("edges", [])

    vocab = set()

    label_lookup = {}

    category_index = defaultdict(list)

    community_index = defaultdict(list)

    subcategory_by_category = defaultdict(set)

    neighbor_index = defaultdict(list)

    word_doc_freq = defaultdict(int)

    auto_synonyms = defaultdict(set)

    label_stems = {}

    # ------------------------------------------------------
    # Build node lookup
    # ------------------------------------------------------

    for nid, nd in nodes.items():

        label = nd.get("label", "")

        if not label:
            continue

        label_lower = label.lower()

        label_lookup[label_lower] = nid

        words = set(
            re.findall(
                r"[a-z0-9]+",
                label_lower
            )
        )

        for w in words:

            if len(w) <= 2:
                continue

            vocab.add(w)

            word_doc_freq[w] += 1

            auto_synonyms[w].add(label)

        label_stems[label_lower] = frozenset(
            _stem(w) for w in words if len(w) > 2
        )

        category = nd.get("category", "")

        if category:
            category_index[category.lower()].append(nid)

        subcategory = nd.get("subcategory", "")

        if category and subcategory:
            subcategory_by_category[
                category.lower()
            ].add(subcategory)

        community = nd.get("community")

        if community is not None:
            community_index[community].append(nid)

    # ------------------------------------------------------
    # Build Neighbor Index
    # Store neighbour + relation + weight
    # ------------------------------------------------------

    for edge in _graph_edges:

        src = edge["source"]

        dst = edge["target"]

        relation = edge.get("relation", "")

        weight = edge.get("weight", 1.0)

        neighbor_index[src].append({

            "target": dst,

            "relation": relation,

            "weight": weight

        })

        neighbor_index[dst].append({

            "target": src,

            "relation": relation,

            "weight": weight

        })

    # ------------------------------------------------------
    # Build sibling lookup
    # ------------------------------------------------------

    subcategory_siblings = {}

    for category, subcats in subcategory_by_category.items():

        subcats = sorted(subcats)

        for sc in subcats:

            siblings = [
                s
                for s in subcats
                if s != sc
            ]

            subcategory_siblings[
                sc.lower()
            ] = siblings

    # ------------------------------------------------------
    # Save
    # ------------------------------------------------------

    _vocab_words = vocab

    _label_lookup = label_lookup

    _category_index = category_index

    _community_index = community_index

    _neighbor_index = neighbor_index

    _subcategory_siblings = subcategory_siblings

    _word_document_freq = word_doc_freq

    _auto_synonyms = auto_synonyms

    _label_stems = label_stems
# ==========================================================
# Intent Classification
# ==========================================================

def classify_intent(query: str):

    q = query.strip().lower()

    if _BARE_CODE_PATTERN.search(query):

        return "error_code"

    if _ERROR_CODE_PATTERN.search(query):

        return "error_code"

    if q.endswith("?"):

        return "question"

    if any(

        q.startswith(w)

        for w in _QUESTION_WORDS

    ):

        return "question"

    return "symptom_report"

def filter_entities(entities):
    """
    Keep only strong entity matches.
    Uses both an absolute floor and a relative threshold.
    """

    if not entities:
        return []

    ABSOLUTE_FLOOR = 0.15

    highest = max(e["confidence"] for e in entities)

    filtered = [
        e
        for e in entities
        if e["confidence"] >= ABSOLUTE_FLOOR
        and e["confidence"] >= highest * 0.70
    ]

    return filtered

# ==========================================================
# Error Code Extraction
# ==========================================================

def extract_error_codes(query):

    codes = [

        c.upper()

        for c in _BARE_CODE_PATTERN.findall(query)

    ]

    match = _ERROR_CODE_PATTERN.search(query)

    if match:

        codes.append(

            match.group(2).upper()

        )

    return list(dict.fromkeys(codes))


# ==========================================================
# Fuzzy Match
# ==========================================================

def _best_vocab_match(

    word,

    threshold=82

):

    if _HAS_RAPIDFUZZ:

        result = process.extractOne(

            word,

            _vocab_words,

            scorer=fuzz.ratio

        )

        if result and result[1] >= threshold:

            return result[0]

        return None

    best = None

    best_score = 0

    for candidate in _vocab_words:

        score = (

            SequenceMatcher(

                None,

                word,

                candidate

            ).ratio()

            * 100

        )

        if score > best_score:

            best_score = score

            best = candidate

    if best_score >= threshold:

        return best

    return None


# ==========================================================
# Typo Correction
# ==========================================================

def correct_typos(query):

    _load_graph()

    words = query.split()

    corrected = []

    for w in words:

        clean = re.sub(

            r"[^a-zA-Z0-9]",

            "",

            w

        ).lower()

        if len(clean) <= 2:

            corrected.append(w)

            continue

        if clean in _vocab_words:

            corrected.append(w)

            continue

        match = _best_vocab_match(clean)

        if match:

            corrected.append(match)

        else:

            corrected.append(w)

    return " ".join(corrected)


# ==========================================================
# Entity Detection
# ==========================================================

def find_entities(

    query,

    threshold=0.20

):

    _load_graph()

    query_words = set(

        re.findall(

            r"[a-z0-9]+",

            query.lower()

        )

    )

    query_stems = frozenset(
        _stem(w) for w in query_words if len(w) > 2
    )

    if not query_stems and not query_words:
        return []

    scored = []

    for label_lower, nid in _label_lookup.items():

        label_words = set(

            re.findall(

                r"[a-z0-9]+",

                label_lower

            )

        )

        if not label_words:

            continue

        label_stem_set = _label_stems.get(
            label_lower, frozenset()
        )

        stem_overlap = query_stems & label_stem_set
        word_overlap = query_words & label_words

        if not stem_overlap and not word_overlap:
            continue

        n_label = len(label_words)
        if n_label == 0:
            continue

        stem_ratio = (
            len(stem_overlap) / n_label
            if label_stem_set else 0
        )

        word_ratio = len(word_overlap) / n_label

        overlap_score = 0.6 * stem_ratio + 0.4 * word_ratio

        idf_sum = sum(
            1 / _word_document_freq.get(w, 1)
            for w in word_overlap
        )

        idf_component = min(idf_sum, 1.5)

        raw_score = (
            0.45 * idf_component
            + 0.55 * overlap_score
        )

        node = _graph_nodes[nid]

        node_weight = NODE_TYPE_WEIGHT.get(

            node.get(

                "node_type",

                ""

            ),

            1.0

        )

        raw_score *= node_weight

        confidence = min(

            raw_score,

            1.0

        )

        if confidence < threshold:

            continue

        scored.append({

            "node_id":

            nid,

            "label":

            node.get(

                "label",

                ""

            ),

            "node_type":

            node.get(

                "node_type",

                ""

            ),

            "category":

            node.get(

                "category",

                ""

            ),

            "subcategory":

            node.get(

                "subcategory",

                ""

            ),

            "community":

            node.get(

                "community"

            ),

            "match_ratio":

            round(

                word_ratio,

                2

            ),

            "weighted_score":

            round(

                raw_score,

                3

            ),

            "confidence":

            round(

                confidence,

                3

            )

        })

    scored.sort(

        key=lambda x: (

            x["confidence"],

            x["weighted_score"]

        ),

        reverse=True

    )

    return scored
# ==========================================================
# Graph Neighbor Expansion
# ==========================================================

def expand_by_neighbors(entities):
    """
    Expand query using graph neighbours while
    filtering irrelevant nodes.
    """

    _load_graph()

    expanded = {}

    # -------------------------------------
    # Communities matched by the query
    # -------------------------------------

    matched_communities = {

        e["community"]

        for e in entities

        if e.get("community") is not None

    }

    # -------------------------------------
    # Useful graph relations
    # -------------------------------------

    VALID_RELATIONS = {

        "HAS_COMPONENT",

        "CAUSES",

        "MEASURED_BY",

        "CONNECTED_TO",

        "HAS_SUBSYSTEM"

    }

    # -------------------------------------
    # Expand
    # -------------------------------------

    for ent in entities:

        node_id = ent["node_id"]

        confidence = ent.get("confidence", 1.0)

        # Ignore weak entity matches
        if confidence < 0.70:

            continue

        neighbours = _neighbor_index.get(node_id, [])

        for edge in neighbours:

            # -----------------------------
            # Support both formats
            # -----------------------------

            if isinstance(edge, dict):

                neighbour_id = edge.get("target")

                relation = edge.get("relation", "")

                edge_weight = edge.get("weight", 1.0)

            else:

                neighbour_id = edge

                relation = ""

                edge_weight = 1.0

            node = _graph_nodes.get(neighbour_id)

            if node is None:

                continue

            # -----------------------------
            # Community filtering
            # -----------------------------

            if node.get("community") not in matched_communities:

                if relation != "CONNECTED_TO":

                    continue

            # -----------------------------
            # Relation filtering
            # -----------------------------

            if relation:

                if relation not in VALID_RELATIONS:

                    continue

            label = node.get("label")

            if not label:

                continue

            # -----------------------------
            # Weighted score
            # -----------------------------

            score = confidence * edge_weight * 0.90

            expanded[label] = max(

                expanded.get(label, 0),

                score

            )

    # -------------------------------------
    # Rank expansions
    # -------------------------------------

    ranked = sorted(

        expanded.items(),

        key=lambda x: x[1],

        reverse=True

    )

    MAX_EXPANSIONS = 20

    return [

        label

        for label, _

        in ranked[:MAX_EXPANSIONS]

    ]



def expand_by_category(entities):
    """
    Expand using sibling nodes from the same
    subcategory while filtering weak matches.
    """

    _load_graph()

    expanded = {}
    MAX_SIBLINGS = 5

    for ent in entities:

        confidence = ent.get("confidence", 1.0)

        # Ignore weak entity matches
        if confidence < 0.70:
            continue

        subcat = (ent.get("subcategory") or "").lower()

        if subcat not in _subcategory_siblings:
            continue

        siblings = sorted(_subcategory_siblings[subcat])

        count = 0

        for sibling in siblings:

            # Don't re-add the original entity
            if sibling.lower() == ent["label"].lower():
                continue

            expanded[sibling] = max(
                expanded.get(sibling, 0),
                confidence
            )

            count += 1

            if count >= MAX_SIBLINGS:
                break

    ranked = sorted(
        expanded.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [label for label, _ in ranked]
# ==========================================================
# Automatic Synonym Expansion
# ==========================================================

def expand_queries(query):

    """
    Multi-query expansion using

    1. Manual synonyms

    2. Graph-derived synonyms
    """

    _load_graph()

    variants = {

        query

    }

    q = query.lower()

    # ------------------------------------
    # Manual synonym table
    # ------------------------------------

    for phrase, replacement in SYNONYM_TABLE.items():

        if phrase in q:

            variant = re.sub(

                re.escape(phrase),

                replacement,

                q,

                flags=re.IGNORECASE

            )

            if variant == q:
                continue

            phrase_stems = frozenset(
                _stem(w) for w in phrase.split() if len(w) > 2
            )
            repl_stems = frozenset(
                _stem(w) for w in replacement.split() if len(w) > 2
            )
            q_stems = frozenset(
                _stem(w) for w in q.split() if len(w) > 2
            )
            new_stems = repl_stems - phrase_stems
            if new_stems & q_stems:
                continue

            variants.add(variant)

    # ------------------------------------
    # Automatic graph synonyms
    # ------------------------------------

    words = set(

        re.findall(

            r"[a-z0-9]+",

            q

        )

    )

    MAX_SYNONYMS_PER_WORD = 5

    for w in words:

        if w not in _auto_synonyms:
            continue

        synonyms = sorted(_auto_synonyms[w])

        for phrase in synonyms[:MAX_SYNONYMS_PER_WORD]:

            phrase_lower = phrase.lower()

            if phrase_lower == q:
                continue

            if q in phrase_lower:
                continue

            phrase_words = set(phrase_lower.split())
            q_words = set(q.split())

            if q_words & phrase_words == q_words:
                continue

            variants.add(phrase_lower)

    return sorted(variants)


# ==========================================================
# Retrieval Hints
# ==========================================================

def build_retrieval_hints(entities):

    """
    Information passed directly into
    hybrid_retrieval.py.
    """

    communities = sorted({

        e["community"]

        for e in entities

        if e.get("community") is not None

    })

    categories = sorted({

        e["category"]

        for e in entities

        if e.get("category")

    })

    node_types = sorted({

        e["node_type"]

        for e in entities

        if e.get("node_type")

    })

    return {

        "communities": communities,

        "categories": categories,

        "node_types": node_types

    }


# ==========================================================
# Expected Sensors
# ==========================================================

CATEGORY_SENSOR_MAP = {

    "Cooling System": [

        "coolant_temp",

        "engine_temp",

        "coolant_pressure"

    ],

    "ABS System": [

        "wheel_speed",

        "brake_pressure"

    ],

    "Fuel System": [

        "fuel_pressure",

        "injector_duration"

    ],

    "Air Conditioning System": [

        "ac_pressure",

        "evaporator_temp"

    ],

    "Engine Components": [

        "rpm",

        "coolant_temp",

        "maf",

        "map",

        "throttle_position"

    ],

    "Engine Compartment": [

        "rpm",

        "oil_pressure",

        "coolant_temp",

        "battery_voltage"

    ],

    "Emissions System": [

        "o2_sensor",

        "fuel_trim",

        "catalytic_temp"

    ],

    "Electrical System": [

        "battery_voltage",

        "alternator_output"

    ],

    "Liquid Systems": [

        "coolant_temp",

        "coolant_pressure"

    ],

    "Transmission": [

        "transmission_temp",

        "vehicle_speed"

    ],

    "Drivetrain": [

        "vehicle_speed",

        "wheel_speed"

    ],

    "Steering": [

        "steering_angle",

        "power_steering_pressure"

    ],

    "Wheels & Tires": [

        "wheel_speed",

        "tire_pressure"

    ]

}


def expected_sensors(entities):

    sensors = set()

    for ent in entities:

        category = ent.get("category")

        if category in CATEGORY_SENSOR_MAP:

            sensors.update(

                CATEGORY_SENSOR_MAP[category]

            )

    return sorted(sensors)
# ==========================================================
# Main Entry Point
# ==========================================================

def preprocess_query(raw_query: str):

    """
    Complete preprocessing pipeline.

    Returns
    -------

    {
        original
        processed
        intent
        error_codes
        entities
        expansion_terms
        expanded_queries
        retrieval_hints
        expected_sensors
    }
    """

    _load_graph()

    # --------------------------------------------
    # Step 1 : Typo correction
    # --------------------------------------------

    corrected = correct_typos(raw_query)

    # --------------------------------------------
    # Step 2 : Intent & error codes (from raw)
    # --------------------------------------------

    intent = classify_intent(raw_query)

    error_codes = extract_error_codes(raw_query)

    # --------------------------------------------
    # Step 3 : Manual synonym expansion (for
    #          entity detection)
    # --------------------------------------------

    manual_variants = {corrected}
    q_lower = corrected.lower()
    q_stems = frozenset(
        _stem(w) for w in q_lower.split() if len(w) > 2
    )
    for phrase, replacement in SYNONYM_TABLE.items():
        if phrase in q_lower:
            variant = re.sub(
                re.escape(phrase),
                replacement,
                q_lower,
                flags=re.IGNORECASE
            )
            if variant == q_lower:
                continue
            phrase_stems = frozenset(
                _stem(w) for w in phrase.split() if len(w) > 2
            )
            repl_stems = frozenset(
                _stem(w) for w in replacement.split() if len(w) > 2
            )
            new_stems = repl_stems - phrase_stems
            if new_stems & q_stems:
                continue
            variant_words = variant.split()
            q_words = q_lower.split()
            is_dup = (
                len(variant_words) > len(q_words)
                and variant_words[:len(q_words)] == q_words
            )
            if not is_dup:
                manual_variants.add(variant)

    # --------------------------------------------
    # Step 4 : Entity detection on original +
    #          manual synonym variants only
    # --------------------------------------------

    all_entities = []

    for variant in manual_variants:

        ents = find_entities(variant)

        all_entities.extend(ents)

    # Keep best entity per node_id
    best_by_nid = {}
    for e in all_entities:
        nid = e["node_id"]
        if nid not in best_by_nid or e["confidence"] > best_by_nid[nid]["confidence"]:
            best_by_nid[nid] = e

    entities = filter_entities(list(best_by_nid.values()))

    # --------------------------------------------
    # Step 5 : Graph expansion from entities
    # --------------------------------------------

    neighbor_terms = expand_by_neighbors(
        entities
    )

    category_terms = expand_by_category(
        entities
    )

    expansion_terms = sorted(

        set(

            neighbor_terms +

            category_terms

        )

    )

    # --------------------------------------------
    # Step 6 : Full expansion for retrieval
    #          (includes auto-synonyms)
    # --------------------------------------------

    expanded_queries = expand_queries(
        corrected
    )

    expanded_queries.extend(

        neighbor_terms

    )

    expanded_queries = sorted(

        set(expanded_queries)

    )

    # --------------------------------------------
    # Step 7 : Retrieval hints
    # --------------------------------------------

    retrieval_hints = build_retrieval_hints(

        entities

    )

    # --------------------------------------------
    # Step 8 : Expected sensors
    # --------------------------------------------

    sensors = expected_sensors(

        entities

    )

    # --------------------------------------------
    # Return
    # --------------------------------------------

    return {

        "original":

        raw_query,

        "processed":

        corrected,

        "intent":

        intent,

        "error_codes":

        error_codes,

        "entities":

        entities,

        "expansion_terms":

        expansion_terms,

        "expanded_queries":

        expanded_queries,

        "retrieval_hints":

        retrieval_hints,

        "expected_sensors":

        sensors

    }


# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":

    test_queries = [

        "brke pedal feels spongey when I press it",

        "check engine light on, code P0301",

        "why is my car overheeting",

        "abs warning light stays on",

        "fuel injector pressure low",

        "engine overheating with coolant leak"

    ]

    for q in test_queries:

        print("\n")

        print("=" * 90)

        print("RAW QUERY")

        print("=" * 90)

        print(q)

        print()

        result = preprocess_query(q)

        print("Processed Query")

        print(result["processed"])

        print()

        print("Intent")

        print(result["intent"])

        print()

        print("Entities")

        for e in result["entities"]:

            print(e)

        print()

        print("Expansion Terms")

        print(result["expansion_terms"])

        print()

        print("Expanded Queries")

        for query in result["expanded_queries"]:

            print(" -", query)

        print()

        print("Retrieval Hints")

        print(result["retrieval_hints"])

        print()

        print("Expected Sensors")

        print(result["expected_sensors"])

        print()

        print("Error Codes")

        print(result["error_codes"])