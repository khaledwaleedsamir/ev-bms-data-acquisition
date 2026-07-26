"""
correlation_analysis.py
=========================

Pearson and Spearman correlation matrices, heatmaps, a pair plot over a
curated subset of key signals, and a ranked list of which columns correlate
most strongly with SOC (for SOC-model feature selection) and which correlate
most strongly with cycle-level degradation indicators (for SOH-model feature
selection).
"""

from __future__ import annotations

import pandas as pd

from . import config, visualization as viz


def compute_correlations(df: pd.DataFrame, columns: list[str] | None = None) -> dict:
    """Return {'pearson': DataFrame, 'spearman': DataFrame} correlation matrices."""
    columns = columns or df.select_dtypes(include="number").columns.tolist()
    # Constant columns produce NaN correlations and clutter the heatmap.
    columns = [c for c in columns if df[c].nunique(dropna=True) > 1]
    numeric = df[columns]
    return {
        "pearson": numeric.corr(method="pearson"),
        "spearman": numeric.corr(method="spearman"),
    }


def rank_correlations_with(corr: pd.DataFrame, target_col: str, top_n: int = 15) -> pd.Series:
    """Rank all other columns by |correlation| with `target_col`, descending."""
    if target_col not in corr.columns:
        return pd.Series(dtype=float)
    ranked = corr[target_col].drop(index=target_col).dropna()
    return ranked.reindex(ranked.abs().sort_values(ascending=False).index).head(top_n)


def run_correlation_analysis(df: pd.DataFrame, outdir: str, interactive: bool = True) -> dict:
    """
    Compute correlation matrices + heatmaps, a pair plot of key signals, and
    ranked SOC/SOH-relevant feature lists.
    """
    # Keep the heatmap/pairplot readable: use pack-level signals plus the
    # cell-imbalance / temperature-derived columns if they were computed,
    # rather than every individual cell-voltage column.
    preferred = [
        config.SIGNAL_MAP["voltage"], config.SIGNAL_MAP["current"], config.SIGNAL_MAP["power"],
        config.SIGNAL_MAP["soc"], config.SIGNAL_MAP["temperature"],
        config.SIGNAL_MAP["delta_voltage"], "cell_v_delta", "cumulative_ah", "cumulative_wh",
        "voltage_slope_v_per_s", "current_slope_a_per_s",
    ]
    columns = [c for c in preferred if c in df.columns]
    corrs = compute_correlations(df, columns)

    figures = {
        "pearson_heatmap": viz.plot_correlation_heatmap(
            corrs["pearson"], outdir, "pearson_correlation_heatmap", title="Pearson correlation"
        ),
        "spearman_heatmap": viz.plot_correlation_heatmap(
            corrs["spearman"], outdir, "spearman_correlation_heatmap", title="Spearman correlation"
        ),
        "pairplot": viz.plot_pairplot(df, columns, outdir, "pairplot", color="run_name"),
    }

    soc_col = config.SIGNAL_MAP["soc"]
    soc_ranking = rank_correlations_with(corrs["pearson"], soc_col) if soc_col in columns \
        else pd.Series(dtype=float)

    return {
        "correlations": corrs,
        "figures": figures,
        "soc_feature_ranking": soc_ranking,
        "columns_used": columns,
    }


def rank_soh_features(cycles: pd.DataFrame) -> pd.Series:
    """
    Rank cycle-level columns by |correlation| with cycle_number - the
    columns that shift most consistently as cycling progresses are the best
    candidate SOH features (this is only meaningful once several cycles have
    been detected; see soc_soh_analysis.soh_indicators for the "not enough
    cycles yet" fallback message).
    """
    if cycles.empty or len(cycles) < 3:
        return pd.Series(dtype=float)
    numeric = cycles.select_dtypes(include="number")
    corr = numeric.corr(method="pearson")
    return rank_correlations_with(corr, "cycle_number", top_n=len(numeric.columns))
