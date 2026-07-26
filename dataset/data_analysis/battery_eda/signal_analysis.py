"""
signal_analysis.py
====================

Battery-signal-specific analysis: current, voltage (pack + per-cell),
temperature, power, energy (via numerical integration), and internal
resistance estimation from current steps.

Each `*_analysis` function computes summary statistics, produces its plots
via visualization.py, and returns a plain dict of
{"stats": {...}, "figures": {name: path, ...}} so report.py can render the
results without needing to know how each analysis works internally.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, visualization as viz


# ---------------------------------------------------------------------------
# Current
# ---------------------------------------------------------------------------
def current_analysis(df: pd.DataFrame, outdir: str, interactive: bool = True) -> dict:
    col = config.SIGNAL_MAP["current"]
    if col not in df.columns:
        return {"stats": {}, "figures": {}}

    current = df[col].dropna()
    derivative = df.groupby("run_name")[col].transform(lambda s: s.diff())

    stats = {
        "mean_a": current.mean(),
        "std_a": current.std(),
        "min_a": current.min(),
        "max_a": current.max(),
        "mean_abs_a": current.abs().mean(),
        "max_abs_step_a": derivative.abs().max(),
    }

    figures = {
        "current_timeseries": viz.plot_timeseries(df, [col], outdir, "current_timeseries",
                                                    title="Current vs time", interactive=interactive),
        "current_histogram": viz.plot_histogram(df, col, outdir, "current_histogram",
                                                  title="Current distribution", interactive=interactive),
        "current_box": viz.plot_box(df, col, outdir, "current_box_by_run", by="run_name",
                                     title="Current by run", interactive=interactive),
        "current_derivative": viz.plot_line(
            df["elapsed_s"], derivative, outdir, "current_derivative",
            xlabel="elapsed_s", ylabel="dI (A/sample)", title="Current derivative",
            interactive=interactive,
        ),
    }
    return {"stats": stats, "figures": figures}


# ---------------------------------------------------------------------------
# Voltage (pack + cells)
# ---------------------------------------------------------------------------
def voltage_analysis(df: pd.DataFrame, outdir: str, interactive: bool = True) -> dict:
    v_col = config.SIGNAL_MAP["voltage"]
    soc_col = config.SIGNAL_MAP["soc"]
    if v_col not in df.columns:
        return {"stats": {}, "figures": {}}

    voltage = df[v_col].dropna()
    stats = {
        "mean_v": voltage.mean(), "std_v": voltage.std(),
        "min_v": voltage.min(), "max_v": voltage.max(),
    }

    figures = {
        "voltage_timeseries": viz.plot_timeseries(df, [v_col], outdir, "voltage_timeseries",
                                                    title="Pack voltage vs time", interactive=interactive),
        "voltage_histogram": viz.plot_histogram(df, v_col, outdir, "voltage_histogram",
                                                  title="Voltage distribution", interactive=interactive),
        "voltage_violin": viz.plot_violin(df, v_col, outdir, "voltage_violin_by_run", by="run_name",
                                           title="Voltage spread by run", interactive=interactive),
    }
    if soc_col in df.columns:
        figures["voltage_vs_soc"] = viz.plot_scatter(
            df, soc_col, v_col, outdir, "voltage_vs_soc", color="run_name",
            title="Voltage vs SOC", interactive=interactive,
        )

    # Voltage relaxation: voltage trajectory during "rest" segments (current
    # near zero), which is exactly the classic OCV-relaxation curve used for
    # SOC calibration - only meaningful if state/segment_id have been added
    # (cycle_analysis.run_cycle_analysis), so this is optional.
    if "state" in df.columns:
        rest = df[df["state"] == "rest"]
        if not rest.empty:
            figures["voltage_relaxation"] = viz.plot_timeseries(
                rest, [v_col], outdir, "voltage_relaxation", title="Voltage during rest segments",
                interactive=interactive,
            )

    return {"stats": stats, "figures": figures}


def cell_voltage_analysis(df: pd.DataFrame, outdir: str, interactive: bool = True) -> dict:
    """
    Per-cell voltage spread: max cell, min cell, and delta (imbalance) over
    time. Cell imbalance is one of the clearest available degradation/SOH
    indicators in this dataset (a weak cell drifts further from the pack
    average as it ages).
    """
    cell_cols = [c for c in df.columns if c.startswith(config.CELL_VOLTAGE_PREFIX)]
    if not cell_cols:
        return {"stats": {}, "figures": {}}

    df = df.copy()
    df["cell_v_max"] = df[cell_cols].max(axis=1)
    df["cell_v_min"] = df[cell_cols].min(axis=1)
    df["cell_v_delta"] = df["cell_v_max"] - df["cell_v_min"]

    stats = {
        "mean_max_cell_v": df["cell_v_max"].mean(),
        "mean_min_cell_v": df["cell_v_min"].mean(),
        "mean_imbalance_v": df["cell_v_delta"].mean(),
        "max_imbalance_v": df["cell_v_delta"].max(),
    }
    # If the BMS's own delta_voltage field is present, report how closely it
    # tracks our computed imbalance - a sanity check on both the sensor and
    # our max-min computation.
    bms_delta = config.SIGNAL_MAP["delta_voltage"]
    if bms_delta in df.columns:
        stats["mean_bms_reported_delta_v"] = df[bms_delta].mean()

    figures = {
        "cell_voltage_minmax": viz.plot_timeseries(
            df, ["cell_v_max", "cell_v_min"], outdir, "cell_voltage_minmax",
            title="Max / min cell voltage vs time", interactive=interactive,
        ),
        "cell_imbalance_timeseries": viz.plot_timeseries(
            df, ["cell_v_delta"], outdir, "cell_imbalance_timeseries",
            title="Cell imbalance (max - min) vs time", interactive=interactive,
        ),
        "cell_imbalance_histogram": viz.plot_histogram(
            df, "cell_v_delta", outdir, "cell_imbalance_histogram",
            title="Cell imbalance distribution", interactive=interactive,
        ),
    }
    return {"stats": stats, "figures": figures, "df": df[["run_name", "elapsed_s",
                                                            "cell_v_max", "cell_v_min", "cell_v_delta"]]}


# ---------------------------------------------------------------------------
# Temperature
# ---------------------------------------------------------------------------
def temperature_analysis(df: pd.DataFrame, outdir: str, interactive: bool = True) -> dict:
    t_col = config.SIGNAL_MAP["temperature"]
    i_col = config.SIGNAL_MAP["current"]
    p_col = config.SIGNAL_MAP["power"]
    if t_col not in df.columns:
        return {"stats": {}, "figures": {}}

    temp_rate = df.groupby("run_name").apply(
        lambda g: (g[t_col].iloc[-1] - g[t_col].iloc[0]) / max(g["elapsed_s"].iloc[-1], 1e-6),
        include_groups=False,
    )

    stats = {
        "mean_temp_degc": df[t_col].mean(),
        "max_temp_degc": df[t_col].max(),
        "min_temp_degc": df[t_col].min(),
        "mean_temp_rise_rate_degc_per_s": temp_rate.mean(),
        "max_temp_rise_rate_degc_per_s": temp_rate.max(),
    }

    figures = {
        "temperature_timeseries": viz.plot_timeseries(df, [t_col], outdir, "temperature_timeseries",
                                                        title="Temperature vs time", interactive=interactive),
        "temperature_histogram": viz.plot_histogram(df, t_col, outdir, "temperature_histogram",
                                                      title="Temperature distribution", interactive=interactive),
    }
    if i_col in df.columns:
        figures["temperature_vs_current"] = viz.plot_scatter(
            df, i_col, t_col, outdir, "temperature_vs_current", color="run_name",
            title="Temperature vs current", interactive=interactive,
        )
    if p_col in df.columns:
        figures["temperature_vs_power"] = viz.plot_scatter(
            df, p_col, t_col, outdir, "temperature_vs_power", color="run_name",
            title="Temperature vs power", interactive=interactive,
        )

    return {"stats": stats, "figures": figures, "temp_rise_rate_by_run": temp_rate}


# ---------------------------------------------------------------------------
# Power / Energy
# ---------------------------------------------------------------------------
def power_analysis(df: pd.DataFrame, outdir: str, interactive: bool = True) -> dict:
    v_col, i_col, p_col = (config.SIGNAL_MAP[k] for k in ("voltage", "current", "power"))
    df = df.copy()

    # Recompute power from V*I as a cross-check against the BMS-reported
    # power field (large disagreement would indicate a sensor or unit bug).
    if v_col in df.columns and i_col in df.columns:
        df["power_computed_w"] = df[v_col] * df[i_col]

    power_col = p_col if p_col in df.columns else "power_computed_w"
    if power_col not in df.columns:
        return {"stats": {}, "figures": {}}

    stats = {
        "mean_power_w": df[power_col].mean(),
        "max_power_w": df[power_col].max(),
        "min_power_w": df[power_col].min(),
    }
    if p_col in df.columns and "power_computed_w" in df.columns:
        stats["mean_abs_power_discrepancy_w"] = (df[p_col] - df["power_computed_w"]).abs().mean()

    def _cumulative_energy(group: pd.DataFrame) -> pd.Series:
        t = group["elapsed_s"].to_numpy()
        p = group[power_col].to_numpy()
        if len(t) < 2:
            return pd.Series(np.zeros(len(t)), index=group.index)
        increments = np.diff(t) * (p[1:] + p[:-1]) / 2.0  # trapezoidal rule, per step
        return pd.Series(np.concatenate(([0.0], np.cumsum(increments))) / 3600.0, index=group.index)

    df["cumulative_energy_wh"] = df.groupby("run_name", group_keys=False).apply(
        _cumulative_energy, include_groups=False
    )

    figures = {
        "power_timeseries": viz.plot_timeseries(df, [power_col], outdir, "power_timeseries",
                                                  title="Power vs time", interactive=interactive),
        "power_histogram": viz.plot_histogram(df, power_col, outdir, "power_histogram",
                                                title="Power distribution", interactive=interactive),
        "cumulative_energy": viz.plot_timeseries(df, ["cumulative_energy_wh"], outdir,
                                                   "cumulative_energy",
                                                   title="Cumulative energy (Wh) vs time",
                                                   interactive=interactive),
    }
    return {"stats": stats, "figures": figures}


def energy_analysis(df: pd.DataFrame) -> dict:
    """
    Per-run Ah and Wh throughput via trapezoidal integration (numerical
    integration, as requested) - the authoritative "how much charge/energy
    moved in this run" numbers, independent of the BMS's own coulomb
    counters (bms_cycle_charge / bms_cycle_capacity), so they can be
    cross-checked against them.
    """
    i_col, p_col = config.SIGNAL_MAP["current"], config.SIGNAL_MAP["power"]
    rows = []
    for run_name, g in df.groupby("run_name"):
        t = g["elapsed_s"].to_numpy()
        ah = np.trapezoid(g[i_col].to_numpy(), t) / 3600.0 if i_col in g else np.nan
        wh = np.trapezoid(g[p_col].to_numpy(), t) / 3600.0 if p_col in g else np.nan
        rows.append({
            "run_name": run_name,
            "duration_h": (t[-1] - t[0]) / 3600.0 if len(t) > 1 else 0.0,
            "net_ah": ah,
            "net_wh": wh,
            "charge_ah": g.loc[g[i_col] > 0, i_col].sum() * (np.median(np.diff(t)) if len(t) > 1 else 1) / 3600.0
            if i_col in g else np.nan,
            "discharge_ah": g.loc[g[i_col] < 0, i_col].sum() * (np.median(np.diff(t)) if len(t) > 1 else 1) / 3600.0
            if i_col in g else np.nan,
        })
    return {"per_run": pd.DataFrame(rows)}


# ---------------------------------------------------------------------------
# Internal resistance estimation
# ---------------------------------------------------------------------------
def estimate_resistance(df: pd.DataFrame,
                         min_di_a: float = config.RESISTANCE_MIN_DI_A,
                         settle_samples: int = config.RESISTANCE_SETTLE_SAMPLES,
                         max_ohm: float = config.RESISTANCE_MAX_OHM) -> pd.DataFrame:
    """
    Whenever current changes abruptly by more than `min_di_a` between two
    consecutive samples, estimate R = dV / dI using the voltage
    `settle_samples` after the step vs. the voltage immediately before it.

    This is the standard "pulse resistance" estimate used by HPPC-style
    testing (this repo already runs HPPC campaigns - see
    hoverboard_bms_HPPC_data.h5) and also fires naturally on ordinary
    discharge/charge transitions.
    """
    v_col, i_col = config.SIGNAL_MAP["voltage"], config.SIGNAL_MAP["current"]
    if v_col not in df.columns or i_col not in df.columns:
        return pd.DataFrame()

    records = []
    for run_name, g in df.groupby("run_name"):
        g = g.reset_index(drop=True)
        current = g[i_col].to_numpy()
        voltage = g[v_col].to_numpy()
        di = np.diff(current)
        step_indices = np.where(np.abs(di) >= min_di_a)[0]  # index i means step between i and i+1

        for i in step_indices:
            after_idx = min(i + 1 + settle_samples, len(g) - 1)
            before_idx = i
            delta_i = current[after_idx] - current[before_idx]
            delta_v = voltage[after_idx] - voltage[before_idx]
            if abs(delta_i) < min_di_a:
                continue
            r_ohm = abs(delta_v / delta_i)
            if r_ohm > max_ohm:
                continue  # discard - almost certainly a sensor artifact, not real pack resistance
            records.append({
                "run_name": run_name,
                "elapsed_s": g["elapsed_s"].iloc[before_idx],
                "delta_i_a": delta_i,
                "delta_v_v": delta_v,
                "r_ohm": r_ohm,
                "soc_pct": g[config.SIGNAL_MAP["soc"]].iloc[before_idx]
                if config.SIGNAL_MAP["soc"] in g else np.nan,
            })

    return pd.DataFrame(records)


def resistance_analysis(df: pd.DataFrame, outdir: str, interactive: bool = True) -> dict:
    r_df = estimate_resistance(df)
    if r_df.empty:
        return {"stats": {}, "figures": {}, "resistance_df": r_df}

    stats = {
        "n_estimates": len(r_df),
        "mean_r_ohm": r_df["r_ohm"].mean(),
        "median_r_ohm": r_df["r_ohm"].median(),
        "std_r_ohm": r_df["r_ohm"].std(),
        "min_r_ohm": r_df["r_ohm"].min(),
        "max_r_ohm": r_df["r_ohm"].max(),
    }
    figures = {
        "resistance_over_time": viz.plot_line(
            r_df["elapsed_s"], r_df["r_ohm"], outdir, "resistance_over_time",
            xlabel="elapsed_s", ylabel="R (ohm)", title="Estimated internal resistance vs time",
            color_values=r_df["run_name"].astype("category").cat.codes, interactive=interactive,
        ),
        "resistance_histogram": viz.plot_histogram(
            r_df, "r_ohm", outdir, "resistance_histogram",
            title="Internal resistance estimate distribution", interactive=interactive,
        ),
    }
    if r_df["soc_pct"].notna().any():
        figures["resistance_vs_soc"] = viz.plot_scatter(
            r_df, "soc_pct", "r_ohm", outdir, "resistance_vs_soc", color="run_name",
            title="Resistance vs SOC", interactive=interactive,
        )
    return {"stats": stats, "figures": figures, "resistance_df": r_df}
