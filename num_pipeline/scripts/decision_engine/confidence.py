"""Calibrated confidence calculation.

Computes a single confidence value [0.0, 1.0] representing how
strongly the available evidence supports the top-ranked diagnosis.

Formula:
    confidence = 0.60 × calibrate_retrieval(score)
               + 0.20 × separation
               + 0.20 × coverage
               + sensor_boost

Where:
    calibrate_retrieval  — linear mapping of raw retrieval score to [0, 1]
    separation           — relative lead of #1 over #2 in retrieval scores
    coverage             — fraction of original user symptoms matched
    sensor_boost         — optional [0, 0.05] bonus for sensor confirmation
"""

from typing import Dict, List

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Reference retrieval score that maps to calibration = 1.0.
# Chosen as the approximate practical maximum observed in the hybrid
# retrieval pipeline (theoretical max ~1.35, practical max ~0.55).
CALIBRATION_REFERENCE = 0.55

# Weights for the three confidence components.
W_RETRIEVAL = 0.60
W_SEPARATION = 0.20
W_COVERAGE = 0.20

# Maximum sensor boost applied when sensor data strongly confirms.
SENSOR_BOOST_MAX = 0.05
# Minimum sensor confidence required to trigger the boost.
SENSOR_BOOST_THRESHOLD = 0.70


# ---------------------------------------------------------------------------
# Component functions
# ---------------------------------------------------------------------------

def calibrate_retrieval(raw_score: float) -> float:
    """Map a raw retrieval score to [0, 1] using linear calibration.

    Parameters
    ----------
    raw_score : float
        The retrieval fusion score (typically in [0.0, 0.55]).

    Returns
    -------
    float
        Calibrated value in [0.0, 1.0].  Values above the reference
        are capped at 1.0.
    """
    if raw_score <= 0.0:
        return 0.0
    return min(raw_score / CALIBRATION_REFERENCE, 1.0)


def compute_separation(retrieval_scores: List[float]) -> float:
    """Compute the relative lead of the top retrieval score over #2.

    Uses the *original* retrieval scores (before evidence fusion) so
    that the separation metric reflects pure retrieval quality.

    Parameters
    ----------
    retrieval_scores : list[float]
        Retrieval scores for all candidates, in descending order.
        Must contain at least one element.

    Returns
    -------
    float
        Separation in [0.0, 1.0].  A single candidate returns 1.0
        (no competition).  Tied scores return 0.0.
    """
    if not retrieval_scores:
        return 0.0

    top = retrieval_scores[0]

    if top <= 0.0:
        return 0.0

    if len(retrieval_scores) < 2:
        return 1.0

    second = retrieval_scores[1]
    gap = top - second

    if gap <= 0.0:
        return 0.0

    return min(gap / top, 1.0)


def compute_coverage(
    original_symptoms: List[str],
    matched_symptoms: List[str],
) -> float:
    """Compute the fraction of original user words found in matched entities.

    Parameters
    ----------
    original_symptoms : list[str]
        Individual words from the raw user query (before preprocessor
        expansion).  Produced by ``_extract_original_symptoms()``.
    matched_symptoms : list[str]
        Individual words from KG entity labels detected from the
        original input.  Produced by ``_extract_matched_symptoms()``.

    Returns
    -------
    float
        Coverage in [0.0, 1.0].  Returns 0.0 if no original symptoms
        were provided.
    """
    if not original_symptoms:
        return 0.0

    # Normalise: lowercase and strip whitespace for comparison.
    orig_set = {s.strip().lower() for s in original_symptoms if s.strip()}
    match_set = {s.strip().lower() for s in matched_symptoms if s.strip()}

    if not orig_set:
        return 0.0

    matched_count = len(orig_set & match_set)
    return matched_count / len(orig_set)


def compute_sensor_boost(
    sensor_results: Dict[str, dict],
    top_fault: str,
) -> float:
    """Return a small confidence boost when sensor data confirms the fault.

    Parameters
    ----------
    sensor_results : dict
        Sensor analysis output keyed by NavicEngine fault ID.
    top_fault : str
        The NavicEngine fault ID of the top-ranked candidate.

    Returns
    -------
    float
        Boost in [0.0, 0.05].  Returns 0.0 when sensor data is
        unavailable or below the confirmation threshold.
    """
    if not sensor_results or top_fault not in sensor_results:
        return 0.0

    sensor_conf = sensor_results[top_fault].get("sensor_confidence", 0.0)

    if sensor_conf < SENSOR_BOOST_THRESHOLD:
        return 0.0

    return min(sensor_conf * SENSOR_BOOST_MAX, SENSOR_BOOST_MAX)


# ---------------------------------------------------------------------------
# Main formula
# ---------------------------------------------------------------------------

def compute_confidence(
    retrieval_scores: List[float],
    original_symptoms: List[str],
    matched_symptoms: List[str],
    sensor_results: Dict[str, dict],
    top_fault: str,
) -> float:
    """Compute the calibrated confidence for the top-ranked candidate.

    Parameters
    ----------
    retrieval_scores : list[float]
        All candidate retrieval scores in descending order.
    original_symptoms : list[str]
        Raw user-provided symptoms (not expanded by the preprocessor).
    matched_symptoms : list[str]
        Symptoms matched by the KG / retrieval.
    sensor_results : dict
        Sensor analysis output keyed by fault ID.
    top_fault : str
        NavicEngine fault ID of the top candidate.

    Returns
    -------
    float
        Confidence in [0.0, 1.0].
    """
    top_score = retrieval_scores[0] if retrieval_scores else 0.0

    cal = calibrate_retrieval(top_score)
    sep = compute_separation(retrieval_scores)
    cov = compute_coverage(original_symptoms, matched_symptoms)
    boost = compute_sensor_boost(sensor_results, top_fault)

    confidence = W_RETRIEVAL * cal + W_SEPARATION * sep + W_COVERAGE * cov + boost

    return min(confidence, 1.0)
