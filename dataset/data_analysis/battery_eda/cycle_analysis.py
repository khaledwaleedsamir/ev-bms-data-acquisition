"""
cycle_analysis.py
==================

Charge/discharge/rest segmentation and cycle detection.

The dataset does not log an explicit "cycle number" that is consistent
across runs (bms_cycles is the BMS's own lifetime counter and is not always
populated consistently - see the project's data-quality notes), so cycles
are detected here directly from the current signal:

  1. classify_state(): each sample -> "charge" / "discharge" / "rest",
     based on the sign of current against a small dead-band (rest).
  2. detect_segments(): contiguous runs of the same state (within one
     run_name) become one "segment", identified by a global segment_id.
  3. summarize_segments(): one row per segment with duration, Ah, Wh, and
     start/end SOC - the building blocks for everything else.
  4. detect_cycles(): pairs a discharge segment with the charge segment that
     follows it (possibly in a later run, if the experiment was split across
     runs) into a "cycle", and computes coulombic/energy efficiency and
     capacity - the core SOH-relevant table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def classify_state(df: pd.DataFrame,
                    current_col: str = config.SIGNAL_MAP["current"],
                    threshold_a: float = config.CURRENT_REST_THRESHOLD_A) -> pd.DataFrame:
    """Add a 'state' column: 'charge' (current > +threshold), 'discharge' (< -threshold), else 'rest'."""
    df = df.copy()
    current = df[current_col]
    df["state"] = np.select(
        [current > threshold_a, current < -threshold_a],
        ["charge", "discharge"],
        default="rest",
    )
    return df


def detect_segments(df: pd.DataFrame,
                     min_segment_samples: int = config.MIN_SEGMENT_SAMPLES) -> pd.DataFrame:
    """
    Add a 'segment_id' column (monotonically increasing across the whole
    dataframe) identifying contiguous same-state stretches within each run.

    Segments shorter than `min_segment_samples` are merged into the
    preceding segment's state before the final segment_id is assigned - this
    removes single-sample flicker where current briefly crosses the rest
    threshold without a real state change.
    """
    if "state" not in df.columns:
        df = classify_state(df)
    df = df.sort_values(["run_name", "elapsed_s"]).reset_index(drop=True)

    # First pass: raw segment ids, from state changes within a run.
    new_run = df["run_name"] != df["run_name"].shift()
    new_state = df["state"] != df["state"].shift()
    raw_segment = (new_run | new_state).cumsum()

    # Merge short-lived segments into the previous segment's state.
    segment_lengths = raw_segment.map(raw_segment.value_counts())
    short = segment_lengths < min_segment_samples
    state_fixed = df["state"].copy()
    # Iteratively fold short segments into whatever state precedes them; a
    # single forward-fill pass is enough because raw_segment already grouped
    # consecutive identical states, so a "short" run is always bounded by
    # longer runs on each side (except possibly at the very start of a run).
    state_fixed[short] = np.nan
    state_fixed = state_fixed.groupby(df["run_name"]).ffill().bfill()
    df["state"] = state_fixed

    new_state = df["state"] != df["state"].shift()
    new_run = df["run_name"] != df["run_name"].shift()
    df["segment_id"] = (new_run | new_state).cumsum()

    return df


def _trapz_integral(t: np.ndarray, y: np.ndarray) -> float:
    """Trapezoidal integral of y over t (seconds), returned in the same time unit as t."""
    if len(t) < 2:
        return 0.0
    return float(np.trapezoid(y, t))


def summarize_segments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse each segment_id into one summary row: timing, Ah, Wh, and
    start/end voltage/SOC/temperature - everything downstream cycle and SOH
    analysis is built from this table.
    """
    v_col = config.SIGNAL_MAP["voltage"]
    i_col = config.SIGNAL_MAP["current"]
    p_col = config.SIGNAL_MAP["power"]
    soc_col = config.SIGNAL_MAP["soc"]
    t_col = config.SIGNAL_MAP["temperature"]

    rows = []
    for seg_id, seg in df.groupby("segment_id"):
        t = seg["elapsed_s"].to_numpy()
        ah = _trapz_integral(t, seg[i_col].to_numpy()) / 3600.0 if i_col in seg else np.nan
        wh = _trapz_integral(t, seg[p_col].to_numpy()) / 3600.0 if p_col in seg else np.nan

        rows.append({
            "segment_id": seg_id,
            "run_name": seg["run_name"].iloc[0],
            "state": seg["state"].iloc[0],
            "start_time": seg["datetime"].iloc[0] if "datetime" in seg else None,
            "end_time": seg["datetime"].iloc[-1] if "datetime" in seg else None,
            "start_elapsed_s": seg["elapsed_s"].iloc[0],
            "duration_s": seg["elapsed_s"].iloc[-1] - seg["elapsed_s"].iloc[0],
            "n_samples": len(seg),
            "ah": ah,
            "wh": wh,
            "avg_current_a": seg[i_col].mean() if i_col in seg else np.nan,
            "avg_voltage_v": seg[v_col].mean() if v_col in seg else np.nan,
            "avg_power_w": seg[p_col].mean() if p_col in seg else np.nan,
            "avg_temp_degc": seg[t_col].mean() if t_col in seg else np.nan,
            "max_temp_degc": seg[t_col].max() if t_col in seg else np.nan,
            "start_soc": seg[soc_col].iloc[0] if soc_col in seg else np.nan,
            "end_soc": seg[soc_col].iloc[-1] if soc_col in seg else np.nan,
        })

    segments = pd.DataFrame(rows)
    if not segments.empty:
        segments["delta_soc"] = segments["end_soc"] - segments["start_soc"]
    return segments


def detect_cycles(segments: pd.DataFrame) -> pd.DataFrame:
    """
    Pair each discharge segment with the next charge segment (in
    chronological order, possibly separated by "rest" segments or a run
    boundary) into a cycle, and compute coulombic and energy efficiency.

    A cycle is only reported once both halves are present, so runs that are
    pure-discharge or pure-charge (common in this dataset - see
    dataset_analysis_findings) correctly do not produce a spurious cycle;
    their segments still show up in `segments` on their own.
    """
    ordered = segments.sort_values("start_elapsed_s").reset_index(drop=True)
    # Global chronological order needs the run's own start time too, not
    # just elapsed_s (which restarts at 0 for every run) - resort by
    # start_time when available.
    if "start_time" in ordered.columns and ordered["start_time"].notna().all():
        ordered = ordered.sort_values("start_time").reset_index(drop=True)

    cycles = []
    cycle_number = 1
    pending_discharge = None

    for _, seg in ordered.iterrows():
        if seg["state"] == "discharge":
            pending_discharge = seg
        elif seg["state"] == "charge" and pending_discharge is not None:
            discharge_ah = abs(pending_discharge["ah"])
            charge_ah = abs(seg["ah"])
            discharge_wh = abs(pending_discharge["wh"])
            charge_wh = abs(seg["wh"])

            cycles.append({
                "cycle_number": cycle_number,
                "discharge_segment_id": pending_discharge["segment_id"],
                "charge_segment_id": seg["segment_id"],
                "discharge_run": pending_discharge["run_name"],
                "charge_run": seg["run_name"],
                "discharge_ah": discharge_ah,
                "charge_ah": charge_ah,
                "coulombic_efficiency": (discharge_ah / charge_ah) if charge_ah > 0 else np.nan,
                "discharge_wh": discharge_wh,
                "charge_wh": charge_wh,
                "energy_efficiency": (discharge_wh / charge_wh) if charge_wh > 0 else np.nan,
                "avg_discharge_temp_degc": pending_discharge["avg_temp_degc"],
                "max_discharge_temp_degc": pending_discharge["max_temp_degc"],
                "start_soc": pending_discharge["start_soc"],
                "end_soc": seg["end_soc"],
            })
            cycle_number += 1
            pending_discharge = None

    return pd.DataFrame(cycles)


def run_cycle_analysis(df: pd.DataFrame) -> dict:
    """Convenience entry point: returns {'df': df_with_state, 'segments': ..., 'cycles': ...}."""
    df = classify_state(df)
    df = detect_segments(df)
    segments = summarize_segments(df)
    cycles = detect_cycles(segments)
    return {"df": df, "segments": segments, "cycles": cycles}
