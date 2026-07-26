"""
data_cleaning.py
=================

Optional, explicit cleaning steps. Nothing here runs implicitly - the
orchestrator (run_battery_eda.py) decides which of these to apply based on
CLI flags, and always keeps the raw dataframe around so cleaning can be
compared against it (anomaly_detection deliberately re-checks the *raw*
data for exactly this reason).

Every function takes a DataFrame and returns a *new* DataFrame plus a short
text log of what it changed, so the report can state exactly what cleaning
was performed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def interpolate_missing(df: pd.DataFrame, columns: list[str] | None = None,
                         method: str = "linear", limit: int = 30) -> tuple[pd.DataFrame, str]:
    """
    Fill missing values in numeric columns via time-aware interpolation,
    per run (so interpolation never bridges across two different runs).

    `limit` caps how many consecutive missing samples get filled - a long
    stretch of NaNs is more likely a real dropout than noise, and should stay
    missing (and get flagged by anomaly detection) rather than be papered
    over silently.
    """
    df = df.copy()
    columns = columns or df.select_dtypes(include=[np.number]).columns.tolist()
    n_before = df[columns].isna().sum().sum()

    def _interp(group: pd.DataFrame) -> pd.DataFrame:
        group[columns] = group[columns].interpolate(
            method=method, limit=limit, limit_direction="both"
        )
        return group

    df = df.groupby("run_name", group_keys=False).apply(_interp)
    n_after = df[columns].isna().sum().sum()
    log = f"interpolate_missing: filled {n_before - n_after} / {n_before} missing values"
    return df, log


def remove_impossible_values(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    Replace physically-impossible sensor readings with NaN (they are left as
    NaN rather than dropped, so row alignment across columns is preserved -
    interpolate_missing can then fill them, or they can be left for
    anomaly_detection to report).

    Bounds come from config.PHYSICAL_LIMITS / CELL_VOLTAGE_LIMITS /
    CELL_TEMP_LIMITS.
    """
    df = df.copy()
    n_flagged = 0

    for col, (lo, hi) in config.PHYSICAL_LIMITS.items():
        if col not in df.columns:
            continue
        mask = ~df[col].between(lo, hi) & df[col].notna()
        n_flagged += mask.sum()
        df.loc[mask, col] = np.nan

    for prefix, (lo, hi) in (
        (config.CELL_VOLTAGE_PREFIX, config.CELL_VOLTAGE_LIMITS),
        (config.CELL_TEMP_PREFIX, config.CELL_TEMP_LIMITS),
    ):
        cell_cols = [c for c in df.columns if c.startswith(prefix)]
        for col in cell_cols:
            mask = ~df[col].between(lo, hi) & df[col].notna()
            n_flagged += mask.sum()
            df.loc[mask, col] = np.nan

    log = f"remove_impossible_values: {n_flagged} out-of-range readings set to NaN"
    return df, log


def clip_outliers(df: pd.DataFrame, columns: list[str] | None = None,
                   method: str = "iqr", factor: float = config.DEFAULT_OUTLIER_IQR_FACTOR
                   ) -> tuple[pd.DataFrame, str]:
    """
    Clip (not remove) statistical outliers in numeric columns to the
    [Q1 - factor*IQR, Q3 + factor*IQR] range (method="iqr"), or to
    mean +/- factor*std (method="zscore").

    Clipping rather than dropping preserves the time axis, which matters for
    time-series and cycle-detection logic downstream.
    """
    df = df.copy()
    columns = columns or df.select_dtypes(include=[np.number]).columns.tolist()
    n_clipped = 0

    for col in columns:
        series = df[col]
        if method == "iqr":
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - factor * iqr, q3 + factor * iqr
        elif method == "zscore":
            mean, std = series.mean(), series.std()
            if std == 0 or np.isnan(std):
                continue
            lo, hi = mean - factor * std, mean + factor * std
        else:
            raise ValueError(f"Unknown method: {method}")

        mask = (series < lo) | (series > hi)
        n_clipped += mask.sum()
        df[col] = series.clip(lower=lo, upper=hi)

    log = f"clip_outliers ({method}, factor={factor}): {n_clipped} values clipped"
    return df, log


def convert_units(df: pd.DataFrame, conversions: dict[str, tuple[str, float]]
                   ) -> tuple[pd.DataFrame, str]:
    """
    Generic unit-conversion hook. `conversions` maps a source column to
    (new_column_name, multiplier), e.g. {"bms_temp_mv": ("bms_temp_v", 1e-3)}.
    No conversions are needed for the current schema (everything is already
    in SI-ish units - V, A, W, degC) - this exists so a future field in raw
    millivolts/milliamps can be handled without touching other modules.
    """
    df = df.copy()
    applied = []
    for src_col, (new_col, multiplier) in conversions.items():
        if src_col not in df.columns:
            continue
        df[new_col] = df[src_col] * multiplier
        applied.append(f"{src_col} -> {new_col} (x{multiplier})")

    log = "convert_units: " + ("; ".join(applied) if applied else "no conversions applied")
    return df, log


def smooth_signal(series: pd.Series, method: str = "moving_average",
                   window: int = config.DEFAULT_SMOOTH_WINDOW) -> pd.Series:
    """
    Smooth a single signal. Supported methods:
      - "moving_average": centered rolling mean
      - "ewm": exponentially-weighted moving average (span=window)
      - "median": centered rolling median (robust to spikes)
      - "savgol": Savitzky-Golay filter (preserves peak shape better than a
        moving average; needs scipy and window >= polyorder + 2)
    """
    if method == "moving_average":
        return series.rolling(window=window, center=True, min_periods=1).mean()
    if method == "ewm":
        return series.ewm(span=window, adjust=False).mean()
    if method == "median":
        return series.rolling(window=window, center=True, min_periods=1).median()
    if method == "savgol":
        from scipy.signal import savgol_filter

        window = window if window % 2 == 1 else window + 1  # savgol needs an odd window
        window = max(window, 5)
        valid = series.dropna()
        if len(valid) < window:
            return series
        smoothed = pd.Series(
            savgol_filter(valid.to_numpy(), window_length=window, polyorder=2),
            index=valid.index,
        )
        return smoothed.reindex(series.index)

    raise ValueError(f"Unknown smoothing method: {method}")


def smooth_columns(df: pd.DataFrame, columns: list[str], method: str = "moving_average",
                    window: int = config.DEFAULT_SMOOTH_WINDOW, suffix: str = "_smooth"
                    ) -> tuple[pd.DataFrame, str]:
    """Add a smoothed copy of each requested column (per run), named `<col><suffix>`."""
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        df[f"{col}{suffix}"] = df.groupby("run_name")[col].transform(
            lambda s: smooth_signal(s, method=method, window=window)
        )
    log = f"smooth_columns ({method}, window={window}): {columns}"
    return df, log


def run_cleaning_pipeline(df: pd.DataFrame, *, interpolate: bool = True,
                           remove_impossible: bool = True, clip: bool = False,
                           smooth_columns_list: list[str] | None = None,
                           smooth_method: str = "moving_average") -> tuple[pd.DataFrame, list[str]]:
    """
    Convenience wrapper the CLI uses to apply a standard cleaning sequence
    and collect a human-readable log of everything that happened, in order:
    impossible values -> interpolation -> outlier clipping -> smoothing.
    """
    logs = []

    if remove_impossible:
        df, log = remove_impossible_values(df)
        logs.append(log)

    if interpolate:
        df, log = interpolate_missing(df)
        logs.append(log)

    if clip:
        df, log = clip_outliers(df)
        logs.append(log)

    if smooth_columns_list:
        df, log = smooth_columns(df, smooth_columns_list, method=smooth_method)
        logs.append(log)

    return df, logs
