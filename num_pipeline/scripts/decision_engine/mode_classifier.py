"""Mode classification and threshold-band candidate selection.

Determines whether a diagnosis is EXTRACTED, INFERRED, or AMBIGUOUS
based on the top candidate's calibrated confidence, then selects
which candidates to display (all within the mode's threshold band).
"""

from typing import List

# ---------------------------------------------------------------------------
# Mode thresholds
# ---------------------------------------------------------------------------

THRESHOLD_EXTRACTED = 0.75
"""Confidence at or above this value → EXTRACTED."""

THRESHOLD_INFERRED = 0.40
"""Confidence at or above this value (but below EXTRACTED) → INFERRED.

Below this → AMBIGUOUS."""

MAX_DISPLAY = 5
"""Maximum number of candidates shown in the threshold band."""


# ---------------------------------------------------------------------------
# Mode classification
# ---------------------------------------------------------------------------

def classify_mode(confidence: float) -> str:
    """Determine the diagnostic mode from the top candidate's confidence.

    Parameters
    ----------
    confidence : float
        Calibrated confidence of the top-ranked candidate [0.0, 1.0].

    Returns
    -------
    str
        "EXTRACTED", "INFERRED", or "AMBIGUOUS".
    """
    if confidence >= THRESHOLD_EXTRACTED:
        return "EXTRACTED"
    if confidence >= THRESHOLD_INFERRED:
        return "INFERRED"
    return "AMBIGUOUS"


def get_mode_description(mode: str) -> str:
    """Return a short human-readable description of the mode.

    Parameters
    ----------
    mode : str
        One of "EXTRACTED", "INFERRED", "AMBIGUOUS".

    Returns
    -------
    str
        Description string.
    """
    descriptions = {
        "EXTRACTED": (
            "High-confidence diagnosis supported by strong evidence "
            "across retrieval, symptom coverage, and (if available) "
            "sensor data."
        ),
        "INFERRED": (
            "Best-guess diagnosis with moderate evidence. Additional "
            "symptoms may improve confidence."
        ),
        "AMBIGUOUS": (
            "Insufficient evidence to identify a clear diagnosis. "
            "Please provide more details."
        ),
    }
    return descriptions.get(mode, "Unknown mode.")


# ---------------------------------------------------------------------------
# Threshold-band candidate selection
# ---------------------------------------------------------------------------

def _get_band_bounds(mode: str) -> tuple[float, float]:
    """Return (low, high) bounds for the mode's confidence band.

    EXTRACTED band: [0.75, 1.01)   — includes the threshold itself
    INFERRED band:  [0.40, 0.75)
    AMBIGUOUS band: [0.00, 0.40)
    """
    if mode == "EXTRACTED":
        return (THRESHOLD_EXTRACTED, THRESHOLD_EXTRACTED + 0.26)
    if mode == "INFERRED":
        return (THRESHOLD_INFERRED, THRESHOLD_EXTRACTED)
    return (0.0, THRESHOLD_INFERRED)


def select_display_candidates(
    candidates: List[dict],
    mode: str,
    max_display: int = MAX_DISPLAY,
) -> List[dict]:
    """Select candidates within the mode's threshold band for display.

    Parameters
    ----------
    candidates : list[dict]
        Fused candidates, each with a ``"confidence"`` key.
    mode : str
        The determined mode ("EXTRACTED", "INFERRED", "AMBIGUOUS").
    max_display : int
        Maximum number of candidates to return.

    Returns
    -------
    list[dict]
        Filtered and capped list of candidates within the band.
    """
    low, high = _get_band_bounds(mode)

    in_band = [
        c for c in candidates
        if low <= c.get("confidence", 0.0) < high
    ]

    in_band.sort(key=lambda c: c.get("confidence", 0.0), reverse=True)

    return in_band[:max_display]
