"""
usage_statistics.py
======================

Headline "how has this battery been used" numbers: operating time, charge
and energy throughput, average/peak power, and a breakdown of time spent
charging / discharging / idle. Needs `state` (from cycle_analysis) to break
down time-by-activity; falls back to current-sign-only stats otherwise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def compute_usage_statistics(df: pd.DataFrame) -> dict:
    i_col, p_col, t_col = (config.SIGNAL_MAP[k] for k in ("current", "power", "temperature"))

    stats: dict = {}

    # Total operating time = sum, per run, of that run's own duration - runs
    # don't overlap in wall-clock time in this dataset, but summing per-run
    # durations (rather than max(datetime) - min(datetime) across the whole
    # file) avoids counting the multi-day gaps *between* runs as "operating".
    if "elapsed_s" in df.columns:
        per_run_duration = df.groupby("run_name")["elapsed_s"].max()
        stats["total_operating_time_h"] = per_run_duration.sum() / 3600.0
        stats["n_runs"] = len(per_run_duration)

    if i_col in df.columns and "elapsed_s" in df.columns:
        charge_ah, discharge_ah = 0.0, 0.0
        for _, g in df.groupby("run_name"):
            t = g["elapsed_s"].to_numpy()
            i = g[i_col].to_numpy()
            if len(t) < 2:
                continue
            dt = np.diff(t, prepend=t[0])
            # Split into charging (+) and discharging (-) contributions
            # before integrating, so the two throughputs don't cancel out.
            charge_ah += np.sum(dt * np.clip(i, 0, None)) / 3600.0
            discharge_ah += np.sum(dt * np.clip(-i, 0, None)) / 3600.0
        stats["total_charge_throughput_ah"] = charge_ah
        stats["total_discharge_throughput_ah"] = discharge_ah
        stats["avg_charging_current_a"] = df.loc[df[i_col] > 0, i_col].mean()
        stats["avg_discharging_current_a"] = df.loc[df[i_col] < 0, i_col].mean()

    if p_col in df.columns:
        stats["max_power_w"] = df[p_col].max()
        stats["avg_power_w"] = df[p_col].mean()
        if "elapsed_s" in df.columns:
            energy_wh = 0.0
            for _, g in df.groupby("run_name"):
                t, p = g["elapsed_s"].to_numpy(), g[p_col].to_numpy()
                if len(t) > 1:
                    energy_wh += np.trapezoid(np.abs(p), t) / 3600.0
            stats["total_energy_throughput_wh"] = energy_wh

    if t_col in df.columns:
        stats["overall_mean_temp_degc"] = df[t_col].mean()
        stats["overall_max_temp_degc"] = df[t_col].max()

    if "state" in df.columns and "elapsed_s" in df.columns:
        # Approximate per-sample dt as the local diff (falls back to the
        # run's median dt for the first sample of each run).
        df = df.copy()
        df["_dt"] = df.groupby("run_name")["elapsed_s"].diff()
        median_dt = df["_dt"].median()
        df["_dt"] = df["_dt"].fillna(median_dt)

        time_by_state = df.groupby("state")["_dt"].sum() / 3600.0
        for state in ("charge", "discharge", "rest"):
            stats[f"{state}_time_h"] = float(time_by_state.get(state, 0.0))

    return stats


def usage_statistics_as_text(stats: dict) -> str:
    lines = ["Battery Usage Statistics", "-------------------------"]
    for key, value in stats.items():
        if isinstance(value, float):
            lines.append(f"{key}: {value:.3f}")
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)
