"""
data_loader.py
===============

Turns a single HDF5 file (as written by `dataset/dataset_utils.py`) into one
tidy pandas.DataFrame, and reports basic data-quality facts about it.

The HDF5 layout is:

    <run_name>/
        timestamp_ms          (float64)
        time_string            (utf-8 string)
        bms/<field>            (1D or 2D float32/int32)
        hoverboard/<field>      (1D or 2D float32/int32)
        attrs: {...}            run-level metadata (date, description, ...)

A file usually contains several run groups. We load *all* of them into one
DataFrame (tagged by a "run_name" column) because most of the interesting
analysis - comparing a discharge run against the charge that follows it,
tracking capacity across runs - is cross-run.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import h5py
import numpy as np
import pandas as pd

from . import config


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _flatten_dataset(group: h5py.Group, dataset_name: str, prefix: str) -> dict:
    """
    Read one HDF5 dataset and return {column_name: np.ndarray}.

    1D datasets become a single column ("<prefix>_<dataset_name>").
    2D datasets (e.g. cell_voltages, temp_values) become one column per
    index, 1-based, so a run's cell voltages become
    "bms_cell_voltages_1" ... "bms_cell_voltages_10".
    """
    values = group[dataset_name][:]
    col_base = f"{prefix}_{dataset_name}"

    if values.ndim == 1:
        return {col_base: values}

    if values.ndim == 2:
        return {f"{col_base}_{i + 1}": values[:, i] for i in range(values.shape[1])}

    # Datasets with more dimensions aren't expected by this schema; skip
    # rather than guess how to flatten them.
    return {}


def _load_run(f: h5py.File, run_name: str) -> pd.DataFrame:
    """Load a single run group into a DataFrame with one row per sample."""
    run_group = f[run_name]

    columns: dict = {}
    # Root-level datasets (timestamp_ms, time_string, ...)
    for name, obj in run_group.items():
        if isinstance(obj, h5py.Dataset):
            columns.update(_flatten_dataset(run_group, name, prefix=""))
        elif isinstance(obj, h5py.Group):
            # One level of subgroup nesting: bms/*, hoverboard/*
            for sub_name in obj.keys():
                columns.update(_flatten_dataset(obj, sub_name, prefix=name))

    # "_" + "timestamp_ms" -> leading underscore from the empty root prefix;
    # strip it so root-level columns keep their plain HDF5 names.
    columns = {k.lstrip("_"): v for k, v in columns.items()}

    # A run's per-field datasets are supposed to be logged in lockstep (one
    # row per acquisition tick), but a dropped/duplicated sample in one
    # background thread (e.g. the hoverboard serial RX thread) can leave one
    # field one or more rows longer/shorter than the rest. Truncate every
    # field to the shortest length in the run rather than letting
    # pd.DataFrame raise, and note it so it surfaces in the quality report.
    lengths = {k: len(v) for k, v in columns.items()}
    if len(set(lengths.values())) > 1:
        min_len = min(lengths.values())
        mismatched = {k: n for k, n in lengths.items() if n != min_len}
        print(f"  [warn] {run_name}: field-length mismatch {mismatched}; "
              f"truncating all fields to {min_len} rows")
        columns = {k: v[:min_len] for k, v in columns.items()}

    df = pd.DataFrame(columns)
    df.insert(0, "run_name", run_name)

    # Run-level metadata (attrs) is broadcast to every row of the run so it
    # can be used as a grouping/filter variable later (e.g. group by
    # meta_description, or filter to battery_age == 'new').
    for key, value in run_group.attrs.items():
        safe_key = f"meta_{key.strip().lower().replace(' ', '_')}"
        df[safe_key] = value

    return df


def load_h5_to_dataframe(h5_path: str, run_names: list[str] | None = None) -> pd.DataFrame:
    """
    Load one or more run groups from an HDF5 file into a single DataFrame.

    Args:
        h5_path: path to the .h5 file.
        run_names: specific run groups to load; if None, load every run in
            the file.

    Returns:
        A DataFrame with columns from every run, outer-joined (a column only
        present in some runs, e.g. "hoverboard_hppc_phase", is NaN elsewhere).
    """
    if not os.path.exists(h5_path):
        raise FileNotFoundError(h5_path)

    with h5py.File(h5_path, "r") as f:
        available = list(f.keys())
        selected = run_names if run_names is not None else available
        missing = set(selected) - set(available)
        if missing:
            raise ValueError(f"Run(s) not found in {h5_path}: {sorted(missing)}")

        run_frames = [_load_run(f, run) for run in selected]

    df = pd.concat(run_frames, ignore_index=True, sort=False)
    df = _add_time_columns(df)
    return df


def _add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Parse timestamp_ms into a real datetime and add a per-run elapsed-time column."""
    if "timestamp_ms" not in df.columns:
        return df

    df["datetime"] = pd.to_datetime(df["timestamp_ms"], unit="ms", errors="coerce")

    # elapsed_s: seconds since the first sample of *that run* - the natural
    # x-axis for time-series plots and for numerical integration, since runs
    # were recorded on different days.
    df = df.sort_values(["run_name", "timestamp_ms"]).reset_index(drop=True)
    df["elapsed_s"] = (
        df.groupby("run_name")["timestamp_ms"].transform(lambda s: (s - s.iloc[0]) / 1000.0)
    )
    return df


# ---------------------------------------------------------------------------
# Column typing
# ---------------------------------------------------------------------------
def detect_column_types(df: pd.DataFrame) -> dict:
    """
    Split columns into numerical vs categorical, ignoring identifier/time
    columns that don't belong in either bucket for statistical purposes.

    Returns:
        {"numerical": [...], "categorical": [...], "excluded": [...]}
    """
    excluded = {"run_name", "time_string", "datetime"}
    numerical, categorical = [], []

    for col in df.columns:
        if col in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            # Low-cardinality integer columns that behave like flags/IDs
            # (e.g. bms_battery_charging, bms_cell_count) are more useful
            # summarized as categorical than as continuous statistics.
            n_unique = df[col].nunique(dropna=True)
            if pd.api.types.is_integer_dtype(df[col]) and n_unique <= 10:
                categorical.append(col)
            else:
                numerical.append(col)
        else:
            categorical.append(col)

    return {"numerical": numerical, "categorical": categorical, "excluded": sorted(excluded)}


# ---------------------------------------------------------------------------
# Data-quality
# ---------------------------------------------------------------------------
@dataclass
class DataQualityReport:
    n_rows_raw: int
    n_duplicate_rows: int
    missing_by_column: dict = field(default_factory=dict)
    constant_columns: list = field(default_factory=list)
    non_monotonic_runs: list = field(default_factory=list)
    large_time_gaps: list = field(default_factory=list)  # (run_name, gap_seconds, index)

    def as_text(self) -> str:
        lines = [
            "Data Quality Report",
            "--------------------",
            f"Raw rows loaded:        {self.n_rows_raw}",
            f"Exact duplicate rows:   {self.n_duplicate_rows}",
        ]
        missing = {k: v for k, v in self.missing_by_column.items() if v > 0}
        if missing:
            lines.append("Columns with missing values:")
            for col, count in sorted(missing.items(), key=lambda kv: -kv[1]):
                lines.append(f"  - {col}: {count} missing")
        else:
            lines.append("Columns with missing values: none")

        if self.constant_columns:
            lines.append(f"Constant columns (zero variance): {self.constant_columns}")
        if self.non_monotonic_runs:
            lines.append(f"Runs with non-monotonic timestamps: {self.non_monotonic_runs}")
        if self.large_time_gaps:
            lines.append(f"Large timestamp gaps detected: {len(self.large_time_gaps)}")
            for run, gap_s, idx in self.large_time_gaps[:10]:
                lines.append(f"  - run '{run}' at row {idx}: gap = {gap_s:.1f} s")
        return "\n".join(lines)


def remove_duplicate_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop exact-duplicate rows (same run, same timestamp, same everything)."""
    before = len(df)
    df = df.drop_duplicates()
    return df.reset_index(drop=True), before - len(df)


def assess_data_quality(df: pd.DataFrame) -> DataQualityReport:
    """
    Build a DataQualityReport for `df` (call this on the *loaded, but not yet
    cleaned* dataframe, so cleaning decisions can be informed by it).
    """
    n_rows_raw = len(df)
    _, n_dup = remove_duplicate_rows(df)

    missing_by_column = df.isna().sum().to_dict()

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    constant_columns = [c for c in numeric_cols if df[c].nunique(dropna=True) <= 1]

    non_monotonic_runs = []
    large_gaps = []
    if "timestamp_ms" in df.columns:
        for run_name, run_df in df.groupby("run_name"):
            ts = run_df["timestamp_ms"].to_numpy()
            if len(ts) < 2:
                continue
            diffs = np.diff(ts)
            if np.any(diffs < 0):
                non_monotonic_runs.append(run_name)

            positive_diffs = diffs[diffs > 0]
            if len(positive_diffs) == 0:
                continue
            median_dt = np.median(positive_diffs)
            gap_mask = diffs > median_dt * config.DROPOUT_GAP_FACTOR
            for local_idx in np.where(gap_mask)[0]:
                gap_seconds = diffs[local_idx] / 1000.0
                large_gaps.append((run_name, gap_seconds, run_df.index[local_idx + 1]))

    return DataQualityReport(
        n_rows_raw=n_rows_raw,
        n_duplicate_rows=n_dup,
        missing_by_column=missing_by_column,
        constant_columns=constant_columns,
        non_monotonic_runs=non_monotonic_runs,
        large_time_gaps=large_gaps,
    )


# ---------------------------------------------------------------------------
# Dataset summary (section 2 of the spec)
# ---------------------------------------------------------------------------
def summarize_dataset(df: pd.DataFrame) -> str:
    """
    Build the human-readable dataset summary: shape, memory footprint,
    dtypes, missing values, descriptive statistics, min/max, and unique
    counts for categorical columns.
    """
    types = detect_column_types(df)
    lines = []

    lines.append("Dataset Summary")
    lines.append("---------------")
    lines.append(f"Samples (rows):        {len(df)}")
    lines.append(f"Features (columns):    {df.shape[1]}")
    lines.append(f"Runs included:         {df['run_name'].nunique() if 'run_name' in df else 'n/a'}")
    mem_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    lines.append(f"Memory usage:          {mem_mb:.2f} MB")
    lines.append("")

    lines.append("Data types:")
    lines.append(df.dtypes.astype(str).to_string())
    lines.append("")

    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    lines.append("Missing values (columns with > 0 missing):")
    lines.append(missing.to_string() if not missing.empty else "  none")
    lines.append("")

    lines.append(f"Numerical features ({len(types['numerical'])}): {types['numerical']}")
    lines.append(f"Categorical features ({len(types['categorical'])}): {types['categorical']}")
    lines.append("")

    if types["numerical"]:
        desc = df[types["numerical"]].describe().T
        desc["range"] = desc["max"] - desc["min"]
        lines.append("Descriptive statistics (numerical columns):")
        lines.append(desc.to_string())
        lines.append("")

    if types["categorical"]:
        lines.append("Unique values (categorical columns):")
        for col in types["categorical"]:
            n_unique = df[col].nunique(dropna=True)
            sample_values = df[col].dropna().unique()[:8]
            lines.append(f"  - {col}: {n_unique} unique, e.g. {list(sample_values)}")

    return "\n".join(lines)
