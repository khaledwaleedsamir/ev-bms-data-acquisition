"""
anomaly_detection.py
======================

Flags suspicious samples in the *raw* (uncleaned) dataframe - this is
deliberately run before data_cleaning so it reports what was actually wrong
with the acquired data, not what's left after cleaning papered over it.

Each check adds one boolean column (`anomaly_<check>`); `detect_anomalies`
combines them into a single `is_anomalous` flag plus a per-check summary
count, which is what ends up in the automatic report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def flag_impossible_values(df: pd.DataFrame) -> pd.Series:
    """Physically-impossible pack/cell voltage, current, temperature, or SOC readings."""
    flag = pd.Series(False, index=df.index)
    for col, (lo, hi) in config.PHYSICAL_LIMITS.items():
        if col in df.columns:
            flag |= ~df[col].between(lo, hi) & df[col].notna()

    for prefix, (lo, hi) in ((config.CELL_VOLTAGE_PREFIX, config.CELL_VOLTAGE_LIMITS),
                              (config.CELL_TEMP_PREFIX, config.CELL_TEMP_LIMITS)):
        for col in df.columns:
            if col.startswith(prefix):
                flag |= ~df[col].between(lo, hi) & df[col].notna()
    return flag


def flag_current_jumps(df: pd.DataFrame,
                        threshold_a: float = config.CURRENT_JUMP_THRESHOLD_A) -> pd.Series:
    """Sudden, physically-unrealistic current jump between consecutive samples of the same run."""
    col = config.SIGNAL_MAP["current"]
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    delta = df.groupby("run_name")[col].transform(lambda s: s.diff().abs())
    return delta > threshold_a


def flag_voltage_discontinuities(df: pd.DataFrame,
                                  threshold_v: float = config.VOLTAGE_DISCONTINUITY_THRESHOLD_V
                                  ) -> pd.Series:
    """Sudden pack-voltage jump not explained by an equally sudden current jump (a sensor glitch)."""
    v_col, i_col = config.SIGNAL_MAP["voltage"], config.SIGNAL_MAP["current"]
    if v_col not in df.columns:
        return pd.Series(False, index=df.index)
    dv = df.groupby("run_name")[v_col].transform(lambda s: s.diff().abs())
    voltage_jump = dv > threshold_v
    if i_col in df.columns:
        current_jump = flag_current_jumps(df)
        # A voltage jump is only "suspicious" if current didn't also jump -
        # a real load step legitimately moves voltage quickly.
        return voltage_jump & ~current_jump
    return voltage_jump


def flag_communication_dropouts(df: pd.DataFrame,
                                 gap_factor: float = config.DROPOUT_GAP_FACTOR) -> pd.Series:
    """Timestamp gap much larger than the run's typical sampling interval (BLE/serial dropout)."""
    flag = pd.Series(False, index=df.index)
    if "timestamp_ms" not in df.columns:
        return flag

    for run_name, g in df.groupby("run_name"):
        diffs = g["timestamp_ms"].diff()
        positive = diffs[diffs > 0]
        if positive.empty:
            continue
        median_dt = positive.median()
        gap_mask = diffs > median_dt * gap_factor
        flag.loc[g.index[gap_mask.fillna(False)]] = True
    return flag


def flag_sensor_spikes(df: pd.DataFrame, columns: list[str] | None = None,
                        window: int = config.SPIKE_ROLLING_WINDOW,
                        z_threshold: float = config.SPIKE_ZSCORE_THRESHOLD) -> pd.Series:
    """
    Robust spike detector: a sample far from its local rolling median
    (relative to local rolling MAD) is flagged, independent of the global
    distribution - this catches single-sample sensor spikes that a global
    z-score or IQR check would miss if the run also has genuine large swings
    elsewhere (e.g. HPPC current pulses).
    """
    columns = columns or [config.SIGNAL_MAP[k] for k in ("voltage", "current", "temperature")]
    flag = pd.Series(False, index=df.index)

    for col in columns:
        if col not in df.columns:
            continue

        def _spike_mask(s: pd.Series) -> pd.Series:
            rolling_median = s.rolling(window, center=True, min_periods=3).median()
            abs_dev = (s - rolling_median).abs()
            mad = abs_dev.rolling(window, center=True, min_periods=3).median()
            # 1.4826 scales MAD to be comparable to a standard deviation
            # under a normal-distribution assumption (standard robust z-score).
            robust_z = abs_dev / (1.4826 * mad.replace(0, np.nan))
            return robust_z > z_threshold

        flag |= df.groupby("run_name")[col].transform(_spike_mask).fillna(False)

    return flag


def detect_anomalies(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run every anomaly check and attach the results as boolean columns.

    Returns:
        (df_with_flags, summary) where `summary` is a small DataFrame with
        one row per anomaly type and how many samples it flagged.
    """
    df = df.copy()
    checks = {
        "anomaly_impossible_value": flag_impossible_values(df),
        "anomaly_current_jump": flag_current_jumps(df),
        "anomaly_voltage_discontinuity": flag_voltage_discontinuities(df),
        "anomaly_communication_dropout": flag_communication_dropouts(df),
        "anomaly_sensor_spike": flag_sensor_spikes(df),
    }
    for name, flag in checks.items():
        df[name] = flag

    df["is_anomalous"] = np.logical_or.reduce(list(checks.values())) if checks else False

    summary = pd.DataFrame({
        "anomaly_type": list(checks.keys()),
        "n_flagged": [int(f.sum()) for f in checks.values()],
        "pct_of_samples": [100.0 * f.sum() / len(df) if len(df) else 0.0 for f in checks.values()],
    }).sort_values("n_flagged", ascending=False)

    return df, summary
