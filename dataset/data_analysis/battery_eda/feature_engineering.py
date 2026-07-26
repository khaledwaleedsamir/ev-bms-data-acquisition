"""
feature_engineering.py
=======================

Derives additional columns useful as SOC/SOH model inputs, and assembles a
second, ML-ready DataFrame (normalized numeric features + encoded
categoricals) that can be exported to CSV.

All time-derivative and rolling features are computed *per run* (via
groupby) so a feature never mixes samples from two different experiments.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from . import config


def _per_run(df: pd.DataFrame, col: str, fn):
    """Apply `fn` to `col` independently within each run_name group."""
    return df.groupby("run_name")[col].transform(fn)


def add_derivative_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add dV/dt, dI/dt, dT/dt (and a couple of related slopes), computed as a
    simple finite difference against elapsed_s. These are useful SOC
    features (a battery's voltage slope under load is informative of SOC)
    and SOH features (resistance growth shows up as a steeper dV/dt at the
    start of a load step).
    """
    df = df.copy()
    v = config.SIGNAL_MAP["voltage"]
    i = config.SIGNAL_MAP["current"]
    t = config.SIGNAL_MAP["temperature"]
    dt = config.SIGNAL_MAP["time"]

    if dt not in df.columns:
        return df

    def _slope(col: str, name: str):
        if col not in df.columns:
            return
        dv = _per_run(df, col, lambda s: s.diff())
        dtime = _per_run(df, dt, lambda s: s.diff()).replace(0, np.nan)
        df[name] = dv / dtime

    _slope(v, "voltage_slope_v_per_s")
    _slope(i, "current_slope_a_per_s")
    _slope(t, "temperature_slope_degc_per_s")

    return df


def add_rolling_features(df: pd.DataFrame,
                          windows: list[int] = config.ROLLING_WINDOWS_S) -> pd.DataFrame:
    """
    Add rolling mean / std / RMS features for voltage and current at each
    window size (in samples). Rolling std of current is a good proxy for
    "how dynamic is the load right now", which helps a SOC model distinguish
    a resting pack from one under a bursty load at the same voltage.
    """
    df = df.copy()
    v = config.SIGNAL_MAP["voltage"]
    i = config.SIGNAL_MAP["current"]

    for col, short_name in ((v, "voltage"), (i, "current")):
        if col not in df.columns:
            continue
        for w in windows:
            df[f"{short_name}_rollmean_{w}"] = _per_run(
                df, col, lambda s, w=w: s.rolling(w, min_periods=1).mean()
            )
            df[f"{short_name}_rollstd_{w}"] = _per_run(
                df, col, lambda s, w=w: s.rolling(w, min_periods=1).std()
            )
            df[f"{short_name}_rollrms_{w}"] = _per_run(
                df, col, lambda s, w=w: np.sqrt((s ** 2).rolling(w, min_periods=1).mean())
            )

    p = config.SIGNAL_MAP["power"]
    if p in df.columns:
        for w in windows:
            df[f"power_rollmean_{w}"] = _per_run(
                df, p, lambda s, w=w: s.rolling(w, min_periods=1).mean()
            )

    return df


def add_cumulative_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cumulative Ah (charge throughput) and Wh (energy throughput) per run,
    via trapezoidal numerical integration over elapsed_s. These are the
    classic "energy throughput" / "cumulative Ah" degradation-relevant
    features used as SOH model inputs, and cumulative Ah in particular is
    also a strong SOC feature (it's a coulomb counter).
    """
    df = df.copy()
    i = config.SIGNAL_MAP["current"]
    p = config.SIGNAL_MAP["power"]
    dt = config.SIGNAL_MAP["time"]

    if dt not in df.columns:
        return df

    def _cumulative_trapz(group: pd.DataFrame, value_col: str, out_col: str, to_hours: float):
        if value_col not in group.columns:
            return group
        t = group[dt].to_numpy()
        y = group[value_col].to_numpy()
        # np.trapz-style running integral: cumulative_integral[k] = integral of y dt from 0..k
        increments = np.diff(t, prepend=t[0]) * (y + np.concatenate(([y[0]], y[:-1]))) / 2.0
        group[out_col] = np.cumsum(increments) / to_hours
        return group

    if i in df.columns:
        df = df.groupby("run_name", group_keys=False).apply(
            _cumulative_trapz, value_col=i, out_col="cumulative_ah", to_hours=3600.0
        )
        df["cumulative_ah_abs"] = df.groupby("run_name")["cumulative_ah"].transform(
            lambda s: s.abs()
        )

    if p in df.columns:
        df = df.groupby("run_name", group_keys=False).apply(
            _cumulative_trapz, value_col=p, out_col="cumulative_wh", to_hours=3600.0
        )

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add elapsed-time-derived features: elapsed_min/h and a simple sample-count-based cycle index."""
    df = df.copy()
    dt = config.SIGNAL_MAP["time"]
    if dt in df.columns:
        df["elapsed_min"] = df[dt] / 60.0
        df["elapsed_h"] = df[dt] / 3600.0
    return df


def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature-engineering pipeline in a sensible order."""
    df = add_time_features(df)
    df = add_derivative_features(df)
    df = add_rolling_features(df)
    df = add_cumulative_features(df)
    return df


# ---------------------------------------------------------------------------
# ML-ready export
# ---------------------------------------------------------------------------
def build_ml_dataframe(df: pd.DataFrame, feature_columns: list[str] | None = None,
                        categorical_columns: list[str] | None = None,
                        normalize: bool = True) -> pd.DataFrame:
    """
    Build a second DataFrame suitable for feeding into scikit-learn/PyTorch:
      - numeric feature columns are standardized (zero mean, unit variance)
      - categorical columns are label-encoded
      - identifier columns (run_name, datetime, time_string) are kept
        un-transformed for traceability, prefixed with "id_".

    If `feature_columns` is None, every numeric column engineered by
    engineer_all_features()/the base loader is used except identifiers.
    """
    df = df.copy()
    from . import data_loader  # local import avoids a cycle at module load time

    types = data_loader.detect_column_types(df)
    feature_columns = feature_columns or types["numerical"]
    categorical_columns = categorical_columns or [
        c for c in types["categorical"] if c not in ("run_name",)
    ]

    ml_df = pd.DataFrame(index=df.index)
    ml_df["id_run_name"] = df.get("run_name")
    if "datetime" in df.columns:
        ml_df["id_datetime"] = df["datetime"]

    numeric_block = df[feature_columns].astype(float)
    if normalize:
        scaler = StandardScaler()
        # Rows that are entirely NaN in a column would break StandardScaler;
        # fill with the column median first purely for the ML export (the
        # analysis dataframe returned elsewhere is untouched).
        numeric_block = numeric_block.fillna(numeric_block.median(numeric_only=True))
        scaled = scaler.fit_transform(numeric_block)
        numeric_block = pd.DataFrame(scaled, columns=feature_columns, index=df.index)

    ml_df = pd.concat([ml_df, numeric_block], axis=1)

    for col in categorical_columns:
        if col not in df.columns:
            continue
        encoder = LabelEncoder()
        non_null = df[col].astype(str).fillna("missing")
        ml_df[f"{col}_encoded"] = encoder.fit_transform(non_null)

    return ml_df


def export_ml_dataframe(ml_df: pd.DataFrame, out_csv_path: str) -> None:
    """Write the ML-ready dataframe to CSV."""
    ml_df.to_csv(out_csv_path, index=False)
