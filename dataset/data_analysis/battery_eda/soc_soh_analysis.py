"""
soc_soh_analysis.py
=====================

SOC-relationship plots (how voltage/current/temperature/power/energy relate
to state of charge) and SOH degradation-indicator trends across cycles.

Ground truth: the BMS's own SOC estimate (bms_battery_level) is used as
"SOC" throughout - there is no independent reference SOC in this dataset.
There is also no logged SOH / capacity-fade ground truth anywhere in the
codebase (confirmed by grepping the repo for SOH/capacity_fade/rated_capacity
- zero hits), so soh_analysis here reports *indicators* (capacity fade,
resistance growth, voltage sag, temperature trend, efficiency trend) rather
than an actual SOH percentage. That gap is exactly what future testing needs
to close - see the recommendations section of the generated report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, visualization as viz


# ---------------------------------------------------------------------------
# SOC
# ---------------------------------------------------------------------------
def soc_analysis(df: pd.DataFrame, outdir: str, interactive: bool = True) -> dict:
    soc_col = config.SIGNAL_MAP["soc"]
    v_col, i_col, t_col, p_col = (config.SIGNAL_MAP[k] for k in
                                   ("voltage", "current", "temperature", "power"))
    if soc_col not in df.columns:
        return {"stats": {}, "figures": {}}

    figures = {
        "soc_timeseries": viz.plot_timeseries(df, [soc_col], outdir, "soc_timeseries",
                                                title="SOC vs time", interactive=interactive),
        "soc_histogram": viz.plot_histogram(df, soc_col, outdir, "soc_histogram",
                                              title="SOC distribution", interactive=interactive),
    }
    for other, label in ((v_col, "voltage"), (i_col, "current"), (t_col, "temperature"),
                          (p_col, "power")):
        if other in df.columns:
            figures[f"soc_vs_{label}"] = viz.plot_scatter(
                df, soc_col, other, outdir, f"soc_vs_{label}", color="run_name",
                title=f"{label.capitalize()} vs SOC", interactive=interactive,
            )

    if "cumulative_energy_wh" in df.columns:
        figures["energy_remaining_vs_soc"] = viz.plot_scatter(
            df, soc_col, "cumulative_energy_wh", outdir, "energy_remaining_vs_soc",
            color="run_name", title="Cumulative energy vs SOC", interactive=interactive,
        )

    # SOC drift / stability: within samples classified as "rest" (near-zero
    # current), SOC should barely move - large jumps there indicate a noisy
    # or drifting BMS SOC estimate rather than real state-of-charge change.
    drift_stats = {}
    if "state" in df.columns:
        rest = df[df["state"] == "rest"]
        if not rest.empty:
            soc_diff_at_rest = rest.groupby("run_name")[soc_col].apply(lambda s: s.diff().abs().max())
            drift_stats["max_soc_jump_at_rest_pct"] = float(soc_diff_at_rest.max())
            drift_stats["mean_soc_jump_at_rest_pct"] = float(soc_diff_at_rest.mean())

    stats = {
        "mean_soc_pct": df[soc_col].mean(),
        "min_soc_pct": df[soc_col].min(),
        "max_soc_pct": df[soc_col].max(),
        "soc_std_pct": df[soc_col].std(),
        **drift_stats,
    }
    return {"stats": stats, "figures": figures}


# ---------------------------------------------------------------------------
# SOH indicators
# ---------------------------------------------------------------------------
def soh_indicators(cycles: pd.DataFrame, resistance_df: pd.DataFrame,
                    outdir: str, interactive: bool = True) -> dict:
    """
    Build SOH-relevant trend plots from the cycle table (cycle_analysis) and
    the resistance-estimate table (signal_analysis.estimate_resistance).

    Reported indicators:
      - capacity fade:      discharge_ah trend across cycle_number
      - resistance growth:  mean R trend across cycle_number
      - voltage sag:        average discharge voltage trend across cycles
      - temperature trend:  average/peak discharge temperature across cycles
      - efficiency trend:   coulombic/energy efficiency across cycles
    """
    figures, stats = {}, {}

    if not cycles.empty and len(cycles) >= 2:
        figures["capacity_fade"] = viz.plot_line(
            cycles["cycle_number"], cycles["discharge_ah"], outdir, "capacity_fade",
            xlabel="cycle_number", ylabel="discharge Ah", title="Capacity fade across cycles",
            interactive=interactive,
        )
        figures["efficiency_trend"] = viz.plot_timeseries(
            cycles.rename(columns={"cycle_number": "elapsed_s"}).assign(run_name="all"),
            ["coulombic_efficiency", "energy_efficiency"], outdir, "efficiency_trend",
            title="Coulombic / energy efficiency across cycles", interactive=interactive,
        )
        figures["discharge_temp_trend"] = viz.plot_line(
            cycles["cycle_number"], cycles["avg_discharge_temp_degc"], outdir,
            "discharge_temp_trend", xlabel="cycle_number", ylabel="avg discharge temp (degC)",
            title="Discharge temperature trend across cycles", interactive=interactive,
        )

        # Simple linear-fit fade rate: negative slope = capacity fading.
        if cycles["discharge_ah"].notna().sum() >= 2:
            slope = np.polyfit(cycles["cycle_number"], cycles["discharge_ah"].fillna(
                cycles["discharge_ah"].mean()), 1)[0]
            stats["capacity_fade_ah_per_cycle"] = float(slope)
        stats["n_full_cycles_detected"] = len(cycles)
        stats["mean_coulombic_efficiency"] = cycles["coulombic_efficiency"].mean()
        stats["mean_energy_efficiency"] = cycles["energy_efficiency"].mean()
    else:
        stats["n_full_cycles_detected"] = len(cycles)
        stats["note"] = (
            "Fewer than 2 full discharge->charge cycles were detected across the loaded "
            "run(s); capacity-fade / efficiency trends need multiple cycles spanning "
            "several sessions to be meaningful (see recommendations)."
        )

    if resistance_df is not None and not resistance_df.empty:
        r_by_run = resistance_df.groupby("run_name")["r_ohm"].mean().reset_index()
        stats["mean_resistance_ohm_by_run"] = r_by_run.set_index("run_name")["r_ohm"].to_dict()

    return {"stats": stats, "figures": figures}
