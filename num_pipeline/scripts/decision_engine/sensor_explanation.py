"""Sensor evidence explanation layer.

Consumes existing pipeline outputs (sensor_dictionary.json, EDA,
comparisons, sensor_mapping) and enriches raw sensor evidence with
human-readable interpretations.  No re-retrieval — everything here
is a presentation-layer transform on already-computed data.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Sensor dictionary (cached after first load)
# ---------------------------------------------------------------------------

_SENSOR_DICT: Optional[Dict[str, Any]] = None


def _load_sensor_dict() -> Dict[str, Any]:
    """Load ``sensor_dictionary.json``, caching across calls."""
    global _SENSOR_DICT
    if _SENSOR_DICT is not None:
        return _SENSOR_DICT

    here = Path(__file__).resolve().parent
    path = here.parent.parent / "data" / "processed" / "sensor_dictionary.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            _SENSOR_DICT = json.load(f)
    else:
        _SENSOR_DICT = {"sensors": {}}
    return _SENSOR_DICT


# ---------------------------------------------------------------------------
# Public API — enrichment functions
# ---------------------------------------------------------------------------

def enrich_sensor(sensor_name: str) -> Dict[str, str]:
    """Look up a sensor's display metadata.

    Returns
    -------
    dict with keys: ``display_name``, ``description``, ``quantity``,
    ``subsystem``, ``confidence``.  Every key is guaranteed non-None;
    missing keys default to the raw sensor name / empty string.
    """
    sd = _load_sensor_dict()
    sensors = sd.get("sensors", {})
    entry = sensors.get(sensor_name, {})
    return {
        "display_name": entry.get("display_name", sensor_name),
        "description": entry.get("description", ""),
        "quantity": entry.get("quantity", ""),
        "subsystem": entry.get("subsystem", ""),
        "confidence": entry.get("confidence", ""),
    }


def enrich_sensor_list(sensor_names: List[str]) -> List[Dict[str, str]]:
    """Enrich a list of raw sensor names."""
    return [
        {"raw_name": name, **enrich_sensor(name)}
        for name in sensor_names
    ]


def enrich_evidence_badges(sensor_evidence: Dict[str, Any]) -> Dict[str, Any]:
    """Add ``*_enriched`` keys to each fault in a sensor_evidence dict.

    The returned dict has the same structure with additional lists
    (``critical_enriched``, ``warning_enriched``, ``normal_enriched``)
    where each entry is a dict with ``raw_name``, ``display_name``,
    ``description``, etc.
    """
    enriched = {}
    for fault_id, ev in sensor_evidence.items():
        enriched[fault_id] = {
            **ev,
            "critical_enriched": enrich_sensor_list(ev.get("critical", [])),
            "warning_enriched": enrich_sensor_list(ev.get("warning", [])),
            "normal_enriched": enrich_sensor_list(ev.get("normal", [])),
        }
    return enriched


def format_evidence_for_llm(
    fault_id: str,
    enriched_evidence: Dict[str, Any],
) -> str:
    """Build a concise sensor evidence string for LLM context injection.

    Parameters
    ----------
    fault_id : str
        NavicEngine fault ID (e.g. ``"FAULT_INJ_DUR"``).
    enriched_evidence : dict
        Fault-keyed dict with ``*_enriched`` lists (from
        ``enrich_evidence_badges``).

    Returns
    -------
    str
        Human-readable summary (e.g. *"Actual Inj Duration (amp_mes):
        CRITICAL — deviation in injection current duration. Charge Air
        Pressure (prs_cmpr_up): WARNING."*).
    """
    ev = enriched_evidence.get(fault_id, {})
    parts = []

    for level, label in [("critical", "CRITICAL"), ("warning", "WARNING")]:
        items = ev.get(f"{level}_enriched", [])
        if items:
            for item in items:
                desc = item["description"]
                desc_str = f" — {desc}" if desc else ""
                parts.append(
                    f"{item['display_name']} ({item['raw_name']}): "
                    f"{label}{desc_str}"
                )

    if not parts:
        return "All sensor readings are within the normal range."

    return "; ".join(parts)


# ---------------------------------------------------------------------------
# EDA visualisation helpers
# ---------------------------------------------------------------------------

def get_eda_paths(speed: int, condition: str = "NOMINAL") -> Dict[str, Any]:
    """Return paths to pre-rendered EDA visualisations for a speed/condition.

    Parameters
    ----------
    speed : int
        Engine speed (1000, 1200, etc.).
    condition : str
        Dataset condition (default ``"NOMINAL"``).

    Returns
    -------
    dict with keys:
        - ``base_dir``: str | None
        - ``correlation_matrix``: str | None
        - ``boxplots``: dict[str, str] — sensor → absolute path
        - ``histograms``: dict[str, str] — sensor → absolute path
        - ``report``: str | None
        - ``exists``: bool
    """
    here = Path(__file__).resolve().parent
    base = here.parent.parent / "outputs" / "eda" / f"INCA_SPEED_{speed}_{condition}"
    result = {
        "base_dir": str(base) if base.exists() else None,
        "correlation_matrix": None,
        "boxplots": {},
        "histograms": {},
        "report": None,
        "exists": base.exists(),
    }

    if not base.exists():
        return result

    cm = base / "correlation_matrix.png"
    if cm.exists():
        result["correlation_matrix"] = str(cm)

    rpt = base / "report.txt"
    if rpt.exists():
        result["report"] = str(rpt)

    for subdir, key in [("boxplots", "boxplots"), ("histograms", "histograms")]:
        d = base / subdir
        if d.exists():
            for p in sorted(d.glob("*.png")):
                result[key][p.stem] = str(p)

    return result


# ---------------------------------------------------------------------------
# Sensor interpretation engine
# ---------------------------------------------------------------------------

_SENSOR_MAPPING_CACHE: Optional[Dict[str, Any]] = None

_BASE_DIR = Path(__file__).resolve().parent.parent.parent


def _load_sensor_mapping() -> Dict[str, Any]:
    """Load ``sensor_mapping.json``, caching across calls."""
    global _SENSOR_MAPPING_CACHE
    if _SENSOR_MAPPING_CACHE is not None:
        return _SENSOR_MAPPING_CACHE
    path = _BASE_DIR / "data" / "processed" / "sensor_mapping.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            _SENSOR_MAPPING_CACHE = json.load(f)
    else:
        _SENSOR_MAPPING_CACHE = {}
    return _SENSOR_MAPPING_CACHE


def _load_comparison(speed: int, fault_id: str) -> "Optional[pd.DataFrame]":
    """Load a fault-vs-nominal comparison CSV, or None if missing."""
    path = _BASE_DIR / "outputs" / "comparisons" / f"{speed}_{fault_id}_vs_NOMINAL.csv"
    if path.exists():
        import pandas as pd
        return pd.read_csv(path)
    return None


def _describe_z(z: float) -> str:
    """Plain-English description of a z-score magnitude."""
    az = abs(z)
    if az >= 4:
        return "far outside"
    if az >= 3:
        return "well outside"
    if az >= 2:
        return "outside"
    return "at the edge of"


def interpret_flagged_sensor(
    sensor_name: str,
    fault_id: str,
    speed: int,
    sensor_results_raw: Dict[str, Any],
) -> Dict[str, str]:
    """Generate a grounded evidence summary for one flagged sensor.

    Uses only available numerical data and metadata.  When data is
    missing the output explicitly acknowledges the gap rather than
    inventing a diagnosis.

    Parameters
    ----------
    sensor_name : str
        INCA signal name (e.g. ``"amp_mes"``).
    fault_id : str
        NavicEngine fault ID (e.g. ``"FAULT_INJ_DUR"``).
    speed : int
        Engine speed RPM.
    sensor_results_raw : dict
        The per-fault sensor-analysis results dict (field
        ``sensor_results_raw`` on ``DiagnosticReport``).

    Returns
    -------
    dict with keys: ``sensor``, ``display_name``, ``severity``,
    ``abnormality``, ``relevance``, ``contribution``.
    """
    info = enrich_sensor(sensor_name)
    display = info["display_name"]
    desc = info["description"]
    subsystem = info["subsystem"]

    # -- Extract per-sensor detail -------------------------------------------
    fault_raw = sensor_results_raw.get(fault_id, {})
    sensor_details = {
        s["sensor"]: s
        for s in fault_raw.get("sensor_results", [])
    }
    detail = sensor_details.get(sensor_name, {})
    severity = detail.get("status", "WARNING")
    sensor_conf = detail.get("sensor_confidence")

    # -- Sensor mapping entry for this fault/speed ---------------------------
    mapping = _load_sensor_mapping()
    fault_mapping = (
        mapping.get(fault_id, {})
        .get(str(speed), {})
        .get("sensors", [])
    )
    mapping_entry = next(
        (m for m in fault_mapping if m["sensor"] == sensor_name),
        {},
    )
    imp = mapping_entry.get("importance_score")
    es = mapping_entry.get("effect_size")

    # -- Comparison data (fault-vs-nominal) ----------------------------------
    comp_df = _load_comparison(speed, fault_id)
    comp_row = None
    if comp_df is not None and "Sensor" in comp_df.columns:
        match = comp_df[comp_df["Sensor"] == sensor_name]
        if not match.empty:
            comp_row = match.iloc[0].to_dict()

    # ------------------------------------------------------------------
    # (1) Abnormality — why the current reading is abnormal
    # ------------------------------------------------------------------
    abn_parts: List[str] = []
    cv = detail.get("current_value")
    nm = detail.get("nominal_mean")
    z = detail.get("z_score")
    pc = detail.get("percent_change")

    if cv is not None and nm is not None:
        direction = "above" if cv > nm else "below"
        diff = abs(cv - nm)
        abn_parts.append(
            f"Current reading ({cv:.2f}) is {direction} the nominal "
            f"mean ({nm:.2f}) by {diff:.2f}"
        )
        if z is not None:
            abn_parts.append(
                f"with a z-score of {z:.1f} ({_describe_z(z)} "
                f"the normal band)"
            )
        if pc is not None:
            abn_parts.append(f"a {pc:+.1f}% shift from nominal")

    elif comp_row is not None:
        fm = comp_row.get("Fault Mean")
        nm_c = comp_row.get("Nominal Mean")
        if fm is not None and nm_c is not None:
            direction = "above" if fm > nm_c else "below"
            abn_parts.append(
                f"Fault-condition mean ({fm:.2f}) is {direction} "
                f"nominal ({nm_c:.2f})"
            )
            es_c = comp_row.get("Effect Size")
            if es_c is not None:
                abn_parts.append(f"effect size {es_c:.2f} sigma")

    abnormality = (
        ". ".join(abn_parts) + "."
        if abn_parts
        else (
            f"{display} is flagged as deviating from the normal "
            f"operating range by the sensor validation pipeline."
        )
    )

    # ------------------------------------------------------------------
    # (2) Relevance — why this sensor matters for the predicted fault
    # ------------------------------------------------------------------
    rel_parts: List[str] = []

    if desc:
        rel_parts.append(desc)

    if imp is not None:
        if imp > 0.8:
            rel_parts.append(
                f"It is a primary diagnostic indicator "
                f"for {fault_id} (importance {imp:.2f})."
            )
        elif imp > 0.5:
            rel_parts.append(
                f"It has moderate diagnostic relevance "
                f"to {fault_id} (importance {imp:.2f})."
            )
        else:
            rel_parts.append(
                f"It has minor diagnostic relevance "
                f"to {fault_id} (importance {imp:.2f})."
            )
    if es is not None:
        rel_parts.append(f"Effect size {es:.2f} sigma.")

    relevance = (
        " ".join(rel_parts)
        if rel_parts
        else (
            f"No additional diagnostic context is available "
            f"for this sensor."
        )
    )

    # ------------------------------------------------------------------
    # (3) Contribution — how the sensor deviation supports the diagnosis
    # ------------------------------------------------------------------
    is_primary = (imp or 0) > 0.8 if imp is not None else False
    is_critical = severity == "CRITICAL"

    if sensor_conf is not None:
        if sensor_conf > 0.8:
            sim_note = (
                "The reading closely matches the expected profile "
                "for this fault type."
            )
        elif sensor_conf > 0.5:
            sim_note = (
                "The reading shows partial alignment with the "
                "fault profile."
            )
        else:
            sim_note = (
                "The reading does not closely match the expected "
                "fault profile, reducing diagnostic confidence."
            )
    else:
        sim_note = ""

    if is_critical and is_primary:
        contrib = (
            f"Strongly confirms the diagnosis. The deviation is both "
            f"severe and directly relevant to the predicted fault "
            f"mechanism. {sim_note}" if sim_note else ""
        )
    elif is_critical:
        contrib = (
            f"Provides supporting evidence. The deviation is severe, "
            f"though this sensor is not a top-ranked diagnostic "
            f"indicator for {fault_id}. {sim_note}" if sim_note else ""
        )
    elif is_primary:
        contrib = (
            f"Lends weight to the diagnosis. The sensor is "
            f"diagnostically important for {fault_id} but the "
            f"deviation magnitude is moderate. {sim_note}" if sim_note else ""
        )
    else:
        contrib = (
            f"Offers minor supporting context. The deviation is "
            f"moderate and the sensor is not a primary indicator "
            f"for {fault_id}. {sim_note}" if sim_note else ""
        )

    return {
        "sensor": sensor_name,
        "display_name": display,
        "description": desc,
        "subsystem": subsystem,
        "severity": severity,
        "abnormality": abnormality,
        "relevance": relevance,
        "contribution": contrib,
    }


def build_sensor_interpretations(
    fault_id: str,
    sensor_results_raw: Dict[str, Any],
    sensor_evidence: Dict[str, Any],
    speed: int,
) -> Dict[str, Any]:
    """Build per-sensor interpretations and an overall narrative.

    Parameters
    ----------
    fault_id : str
        NavicEngine fault ID.
    sensor_results_raw : dict
        Full ``sensor_results`` dict from ``DiagnosticReport``.
    sensor_evidence : dict
        Badges dict from ``DiagnosticReport``.
    speed : int
        Engine speed RPM.

    Returns
    -------
    dict with keys:
        - ``interpretations``: dict keyed by sensor name, each value is
          the output of ``interpret_flagged_sensor()``.
        - ``overall_narrative``: str — synthesised summary of all
          flagged sensors for this fault.
        - ``has_evidence``: bool.
    """
    evidence = sensor_evidence.get(fault_id, {})
    flagged = evidence.get("critical", []) + evidence.get("warning", [])
    has_evidence = bool(flagged)

    if not has_evidence:
        return {
            "interpretations": {},
            "overall_narrative": (
                "No flagged sensor readings were detected for this "
                "fault candidate. All checked sensors are within "
                "their normal operating ranges."
            ),
            "has_evidence": False,
        }

    interpretations: Dict[str, Any] = {}
    for s in flagged:
        interpretations[s] = interpret_flagged_sensor(
            s, fault_id, speed, sensor_results_raw,
        )

    overall = _build_overall_narrative(
        fault_id, interpretations, evidence,
    )

    return {
        "interpretations": interpretations,
        "overall_narrative": overall,
        "has_evidence": True,
    }


def _build_overall_narrative(
    fault_id: str,
    interpretations: Dict[str, Any],
    evidence: Dict[str, Any],
) -> str:
    """Synthesise all flagged-sensor interpretations into one paragraph."""
    status = evidence.get("status", "No Evidence")
    confidence = evidence.get("sensor_confidence", 0.0)
    num_critical = len(evidence.get("critical", []))
    num_warning = len(evidence.get("warning", []))

    subsystems: set = set()
    for interp in interpretations.values():
        sub = interp.get("subsystem")
        if sub:
            subsystems.add(sub)

    parts: List[str] = []

    # Opening — verdict
    if status == "Supported":
        parts.append(
            f"The numerical sensor evidence supports the {fault_id} "
            f"diagnosis with a weighted confidence of "
            f"{confidence:.0%}."
        )
    elif status == "Contradicted":
        parts.append(
            f"The numerical sensor evidence does not align with the "
            f"{fault_id} diagnosis — other predicted faults have "
            f"stronger sensor indicators."
        )
    else:
        parts.append(
            f"The numerical sensor evidence for {fault_id} is "
            f"inconclusive (no strong deviations)."
        )

    # Flagged count
    if num_critical > 0 or num_warning > 0:
        labels = []
        if num_critical > 0:
            labels.append(f"{num_critical} critical")
        if num_warning > 0:
            labels.append(f"{num_warning} warning-level")
        parts.append(
            f"Of the sensors checked, {' and '.join(labels)} "
            f"deviation{'s were' if len(labels) > 1 else ' was'} "
            f"detected."
        )

    # Affected subsystems
    if subsystems:
        parts.append(
            f"The affected subsystem{'s are' if len(subsystems) > 1 else ' is'}: "
            f"{', '.join(sorted(subsystems))}."
        )

    # Primary deviations
    primary_names = [
        interpretations[n]["display_name"]
        for n in interpretations
        if interpretations[n]["severity"] == "CRITICAL"
    ]
    if primary_names:
        if len(primary_names) == 1:
            parts.append(
                f"The primary deviation is in "
                f"{primary_names[0]}."
            )
        else:
            parts.append(
                f"The primary deviations are in "
                f"{', '.join(primary_names[:-1])} and "
                f"{primary_names[-1]}."
            )

    return " ".join(parts)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Pre-rendered box plot lookup
# ---------------------------------------------------------------------------


def get_sensor_boxplot_path(
    speed: int, sensor_name: str, condition: str = "NOMINAL"
) -> "Optional[str]":
    """Return the absolute path to the pre-rendered box plot PNG, or None."""
    here = Path(__file__).resolve().parent
    path = (
        here.parent.parent
        / "outputs"
        / "eda"
        / f"INCA_SPEED_{speed}_{condition}"
        / "boxplots"
        / f"{sensor_name}.png"
    )
    return str(path) if path.exists() else None


# ---------------------------------------------------------------------------
# Histogram generation (on-the-fly, no pre-rendered images)
# ---------------------------------------------------------------------------

_NOMINAL_CSV_CACHE: Dict[int, "pd.DataFrame"] = {}


def _load_nominal_csv(speed: int) -> "Optional[pd.DataFrame]":
    """Load the nominal-condition CSV for *speed*, caching across calls."""
    if speed in _NOMINAL_CSV_CACHE:
        return _NOMINAL_CSV_CACHE[speed]

    path = _BASE_DIR / "data" / "processed" / f"INCA_SPEED_{speed}_NOMINAL.csv"
    if not path.exists():
        _NOMINAL_CSV_CACHE[speed] = None
        return None

    import pandas as pd
    df = pd.read_csv(path)
    _NOMINAL_CSV_CACHE[speed] = df
    return df


def generate_sensor_histogram(
    sensor_name: str,
    speed: int,
    current_value: Optional[float],
    nominal_mean: Optional[float],
) -> "Optional[plt.Figure]":
    """Render a histogram of the nominal distribution for *sensor_name*.

    Overlays the *current_value* (red dashed line) and *nominal_mean*
    (green dashed line) directly on the distribution so the reader can
    visually assess how far the current reading deviates.

    Parameters
    ----------
    sensor_name : str
        INCA signal name.
    speed : int
        Engine speed RPM.
    current_value : float or None
        The current sensor reading (from the fault sample).  When
        ``None`` the overlay is omitted.
    nominal_mean : float or None
        The nominal mean for this sensor.  When ``None`` the overlay
        is omitted.

    Returns
    -------
    matplotlib.figure.Figure or None
        ``None`` when the CSV cannot be loaded or the sensor column
        does not exist.
    """
    df = _load_nominal_csv(speed)
    if df is None or sensor_name not in df.columns:
        return None

    col = df[sensor_name].dropna()
    if len(col) < 2:
        return None

    # TEMPORARY DEBUG: identify the Python environment at runtime
    import sys as _sys, os as _os
    _dbg = (
        f"\n=== HISTOGRAM DEBUG ===\n"
        f"sys.executable: {_sys.executable}\n"
        f"sys.version: {_sys.version}\n"
        f"PYTHONNOUSERSITE={_os.environ.get('PYTHONNOUSERSITE', '(not set)')}\n"
        f"PYTHONPATH={_os.environ.get('PYTHONPATH', '(not set)')}\n"
        f"VIRTUAL_ENV={_os.environ.get('VIRTUAL_ENV', '(not set)')}\n"
        f"sys.path:\n"
    )
    for _p in _sys.path:
        _dbg += f"  {_p}\n"
    # Try importing
    try:
        import matplotlib as _mpl
        _dbg += f"matplotlib.__file__: {_mpl.__file__}\n"
        import matplotlib.pyplot
        _dbg += "matplotlib.pyplot: OK\n"
    except ImportError as _e:
        _dbg += f"IMPORT FAILED: {_e}\n"
    _dbg += "=== END DEBUG ===\n"
    _sys.stderr.write(_dbg)
    _sys.stderr.flush()

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(col, bins=40, color="#4682B4", alpha=0.65, edgecolor="white", linewidth=0.4)

    # Horizontal axis label
    ax.set_xlabel(sensor_name)
    ax.set_ylabel("Frequency")

    # Vertical line for the current reading
    if current_value is not None:
        ax.axvline(
            current_value,
            color="#D32F2F",
            linestyle="--",
            linewidth=1.8,
            label=f"Current Reading ({current_value:.2f})",
        )

    # Vertical line for the nominal mean
    if nominal_mean is not None:
        ax.axvline(
            nominal_mean,
            color="#2E7D32",
            linestyle="--",
            linewidth=1.4,
            label=f"Nominal Mean ({nominal_mean:.2f})",
        )

    if current_value is not None or nominal_mean is not None:
        ax.legend(fontsize=7, loc="best")

    ax.set_title(f"{sensor_name} — Nominal Distribution", fontsize=9)
    plt.tight_layout()
    return fig
