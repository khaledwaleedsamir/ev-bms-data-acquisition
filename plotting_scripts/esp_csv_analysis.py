"""
Battery Analysis Script
========================
Reusable functions for analysing battery CSV data with the following columns:
    datetime_utc, esp_timestamp_ms, voltage_V, current_A, temperature_degC,
    cycle_charge_Ah, cycle_capacity_Wh, bms_soc_pct, pred_soc_pct,
    inference_us, esp_temp_degC

Usage:
    import battery_analysis as ba
    df = ba.load_csv("your_file.csv")
    ba.plot_overview(df)
    ba.average_inference_time(df)
    ba.clean_esp_temp(df)
    ba.calculate_actual_capacity(df)
    ba.soc_metrics(df)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.interpolate import interp1d

# ---------------------------------------------------------------------------
# Publication-ready style defaults
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "font.size":          11,
    "axes.titlesize":     12,
    "axes.labelsize":     11,
    "axes.linewidth":     0.8,
    "axes.grid":          True,
    "grid.linestyle":     "--",
    "grid.linewidth":     0.5,
    "grid.alpha":         0.6,
    "legend.fontsize":    10,
    "legend.framealpha":  0.9,
    "legend.edgecolor":   "0.8",
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "xtick.major.width":  0.8,
    "ytick.major.width":  0.8,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "lines.linewidth":    1.4,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
})

# Colour palette (colour-blind friendly)
C = {
    "voltage":    "#1f77b4",   # blue
    "current":    "#d62728",   # red
    "temp":       "#2ca02c",   # green
    "bms_soc":    "#9467bd",   # purple
    "pred_soc":   "#ff7f0e",   # orange
    "esp_temp":   "#17becf",   # teal
    "cleaned":    "#1f77b4",   # blue (cleaned curve)
    "outlier":    "#d62728",   # red (removed points)
}


# ---------------------------------------------------------------------------
# 1. DATA LOADER
# ---------------------------------------------------------------------------

def load_csv(filepath: str) -> pd.DataFrame:
    """
    Load a battery CSV file and parse timestamps.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame with datetime_utc parsed and sorted chronologically.
    """
    df = pd.read_csv(filepath, parse_dates=["datetime_utc"])
    df = df.sort_values("datetime_utc").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 2. PLOT 1 – 2×2 Overview Grid
# ---------------------------------------------------------------------------

def plot_overview(df: pd.DataFrame,
                  save_path: str | None = None) -> plt.Figure:
    """
    Plot a 2×2 grid: Voltage | Current | Temperature | SOC (BMS + Predicted).

    Parameters
    ----------
    df : pd.DataFrame
        Battery data loaded with ``load_csv``.
    save_path : str, optional
        If provided, save the figure to this path (e.g. "overview.pdf").

    Returns
    -------
    matplotlib.figure.Figure
    """
    time = df["datetime_utc"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    fig.suptitle("Battery Cycle Overview", fontsize=13, fontweight="bold", y=1.01)

    # ── Voltage ──────────────────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(time, df["voltage_V"], color=C["voltage"], label="Voltage")
    ax.set_ylabel("Voltage (V)")
    ax.set_title("Terminal Voltage")
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.legend()

    # ── Current ──────────────────────────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(time, df["current_A"], color=C["current"], label="Current")
    ax.set_ylabel("Current (A)")
    ax.set_title("Load Current")
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.legend()

    # ── Temperature ──────────────────────────────────────────────────────────
    ax = axes[1, 0]
    ax.plot(time, df["temperature_degC"], color=C["temp"], label="Temperature")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Cell Temperature")
    ax.set_xlabel("Time (UTC)")
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.legend()

    # ── SOC ──────────────────────────────────────────────────────────────────
    ax = axes[1, 1]
    ax.plot(time, df["bms_soc_pct"],  color=C["bms_soc"],  label="BMS SoC",
            linestyle="-")
    ax.plot(time, df["pred_soc_pct"], color=C["pred_soc"], label="Predicted SoC",
            linestyle="--", linewidth=1.2)
    ax.set_ylabel("State of Charge (%)")
    ax.set_title("State of Charge")
    ax.set_xlabel("Time (UTC)")
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.legend()

    # Shared x-axis formatting
    for ax_row in axes:
        for ax in ax_row:
            ax.xaxis.set_major_formatter(
                plt.matplotlib.dates.DateFormatter("%H:%M")
            )
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path)
        print(f"[plot_overview] Figure saved to '{save_path}'")

    plt.show()
    return fig


# ---------------------------------------------------------------------------
# 3. AVERAGE INFERENCE TIME
# ---------------------------------------------------------------------------

def average_inference_time(df: pd.DataFrame,
                            print_result: bool = True) -> dict:
    """
    Calculate average, median, min, and max inference time per sample.

    Parameters
    ----------
    df : pd.DataFrame
        Battery data loaded with ``load_csv``.
    print_result : bool
        If True, print a summary table to stdout.

    Returns
    -------
    dict
        Keys: 'mean_us', 'median_us', 'min_us', 'max_us', 'std_us',
              'mean_ms', 'n_samples'
    """
    inf = df["inference_us"].dropna()
    stats = {
        "n_samples":  len(inf),
        "mean_us":    inf.mean(),
        "median_us":  inf.median(),
        "std_us":     inf.std(),
        "min_us":     inf.min(),
        "max_us":     inf.max(),
        "mean_ms":    inf.mean() / 1_000,
    }

    if print_result:
        sep = "─" * 40
        print(sep)
        print("  Inference Time Statistics (ESP32)")
        print(sep)
        print(f"  Samples          : {stats['n_samples']}")
        print(f"  Mean             : {stats['mean_us']:.1f} µs  "
              f"({stats['mean_ms']:.3f} ms)")
        print(f"  Median           : {stats['median_us']:.1f} µs")
        print(f"  Std Dev          : {stats['std_us']:.1f} µs")
        print(f"  Min              : {stats['min_us']:.1f} µs")
        print(f"  Max              : {stats['max_us']:.1f} µs")
        print(sep)

    return stats


# ---------------------------------------------------------------------------
# 4. ESP TEMPERATURE CLEANING
# ---------------------------------------------------------------------------

def clean_esp_temp(df: pd.DataFrame,
                   z_thresh: float = 2.5,
                   window: int = 11,
                   min_delta: float = 5.0,
                   save_path: str | None = None,
                   plot: bool = True) -> pd.Series:
    """
    Remove outliers from ``esp_temp_degC`` and interpolate the cleaned signal.

    Detection strategy (two-stage):
      1. Rolling Z-score – flag points whose deviation from the local rolling
         mean exceeds ``z_thresh`` standard deviations.
      2. Absolute delta threshold – also flag points where the sample-to-sample
         jump exceeds ``min_delta`` degrees (catches isolated spikes that are
         too short for the Z-score window).

    After flagging, outliers are replaced by cubic-spline interpolation over
    the remaining valid samples.

    Parameters
    ----------
    df : pd.DataFrame
        Battery data loaded with ``load_csv``.
    z_thresh : float
        Z-score threshold for outlier detection (default 2.5).
    window : int
        Rolling window length for Z-score computation (default 11 samples).
    min_delta : float
        Absolute jump threshold in °C (default 5.0 °C).
    save_path : str, optional
        If provided, save the diagnostic figure to this path.
    plot : bool
        If True, render a diagnostic plot.

    Returns
    -------
    pd.Series
        Cleaned ``esp_temp_degC`` series (same index as ``df``).
    """
    raw = df["esp_temp_degC"].copy().astype(float)
    idx = np.arange(len(raw))

    # ── Stage 1: rolling Z-score ──────────────────────────────────────────
    roll_mean = raw.rolling(window, center=True, min_periods=1).mean()
    roll_std  = raw.rolling(window, center=True, min_periods=1).std().fillna(1e-6)
    z_score   = (raw - roll_mean).abs() / roll_std

    # ── Stage 2: absolute delta ───────────────────────────────────────────
    delta = raw.diff().abs()

    outlier_mask = (z_score > z_thresh) | (delta > min_delta)

    n_out = outlier_mask.sum()
    print(f"[clean_esp_temp] {n_out} outlier(s) detected and removed.")

    # ── Interpolation on valid samples ────────────────────────────────────
    valid_idx = idx[~outlier_mask]
    valid_val = raw.values[~outlier_mask]

    if len(valid_idx) < 2:
        print("[clean_esp_temp] Too few valid samples – returning raw signal.")
        return raw

    interp_fn  = interp1d(valid_idx, valid_val, kind="cubic",
                          fill_value="extrapolate")
    cleaned    = pd.Series(interp_fn(idx), index=raw.index, name="esp_temp_degC_clean")

    # ── Diagnostic plot ───────────────────────────────────────────────────
    if plot:
        time = df["datetime_utc"]
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        fig.suptitle("ESP32 Temperature – Outlier Removal & Interpolation",
                     fontsize=13, fontweight="bold")

        # Top panel: raw + flagged
        ax = axes[0]
        ax.plot(time, raw, color=C["esp_temp"], linewidth=1.2, label="Raw signal")
        ax.scatter(time[outlier_mask], raw[outlier_mask],
                   color=C["outlier"], s=30, zorder=5, label="Detected outliers")
        ax.set_ylabel("ESP Temp (°C)")
        ax.set_title("Raw Signal with Detected Outliers")
        ax.legend()
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

        # Bottom panel: cleaned
        ax = axes[1]
        ax.plot(time, cleaned, color=C["cleaned"], linewidth=1.2,
                label="Cleaned & interpolated")
        ax.set_ylabel("ESP Temp (°C)")
        ax.set_title("Cleaned Signal (Cubic Interpolation)")
        ax.set_xlabel("Time (UTC)")
        ax.legend()
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.xaxis.set_major_formatter(
            plt.matplotlib.dates.DateFormatter("%H:%M")
        )
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path)
            print(f"[clean_esp_temp] Figure saved to '{save_path}'")
        plt.show()

    return cleaned


# ---------------------------------------------------------------------------
# 5. ACTUAL BATTERY CAPACITY
# ---------------------------------------------------------------------------

def calculate_actual_capacity(df: pd.DataFrame,
                               nominal_capacity_Ah: float | None = None,
                               print_result: bool = True) -> dict:
    """
    Calculate the actual discharged capacity of the battery from the
    measured current and elapsed time (coulomb counting).

    The function integrates ``current_A`` over time using the trapezoidal
    rule with the real sample timestamps.

    Parameters
    ----------
    df : pd.DataFrame
        Battery data loaded with ``load_csv``.
    nominal_capacity_Ah : float, optional
        Rated / initial capacity of the battery in Ah. If supplied, a
        State-of-Health (SoH) estimate is also returned.
    print_result : bool
        If True, print a formatted results table.

    Returns
    -------
    dict
        Keys:
          'duration_h'         – total test duration in hours
          'capacity_Ah'        – integrated discharged capacity (Ah)
          'capacity_Wh'        – energy (Wh), using mean voltage
          'mean_current_A'     – average current during the test
          'mean_voltage_V'     – average voltage during the test
          'soh_pct'            – SoH % (only if nominal_capacity_Ah given)
    """
    # Elapsed time in hours for each sample
    t_s  = (df["datetime_utc"] - df["datetime_utc"].iloc[0]).dt.total_seconds().values
    t_h  = t_s / 3600.0
    I    = df["current_A"].values
    V    = df["voltage_V"].values

    duration_h   = t_h[-1] - t_h[0]
    capacity_Ah  = float(np.trapz(I, t_h))          # ∫ I dt  [Ah]
    capacity_Wh  = float(np.trapz(I * V, t_h))      # ∫ I·V dt  [Wh]
    mean_I       = float(np.mean(I))
    mean_V       = float(np.mean(V))

    result = {
        "duration_h":     duration_h,
        "capacity_Ah":    abs(capacity_Ah),
        "capacity_Wh":    abs(capacity_Wh),
        "mean_current_A": mean_I,
        "mean_voltage_V": mean_V,
    }

    if nominal_capacity_Ah is not None:
        result["soh_pct"] = 100.0 * abs(capacity_Ah) / nominal_capacity_Ah

    if print_result:
        sep = "─" * 44
        print(sep)
        print("  Actual Battery Capacity  (Coulomb Counting)")
        print(sep)
        print(f"  Test duration            : {duration_h:.3f} h "
              f"({duration_h*60:.1f} min)")
        print(f"  Mean current             : {mean_I:.3f} A")
        print(f"  Mean voltage             : {mean_V:.3f} V")
        print(f"  Discharged capacity      : {result['capacity_Ah']:.4f} Ah")
        print(f"  Discharged energy        : {result['capacity_Wh']:.4f} Wh")
        if "soh_pct" in result:
            print(f"  State of Health (SoH)    : {result['soh_pct']:.2f} %")
        print(sep)

    return result


# ---------------------------------------------------------------------------
# 6. SOC PREDICTION METRICS  (R², MAE, RMSE)
# ---------------------------------------------------------------------------

def soc_metrics(df: pd.DataFrame,
                print_result: bool = True,
                save_path: str | None = None) -> dict:
    """
    Compute regression metrics between predicted SoC and BMS SoC, and
    produce a two-panel diagnostic figure (time-series overlay + scatter plot).

    Metrics
    -------
    - **R²**   – coefficient of determination (1 = perfect fit)
    - **MAE**  – mean absolute error (percentage points)
    - **RMSE** – root mean squared error (percentage points)

    Parameters
    ----------
    df : pd.DataFrame
        Battery data loaded with ``load_csv``.
    print_result : bool
        If True, print a formatted metrics table to stdout.
    save_path : str, optional
        If provided, save the diagnostic figure to this path.

    Returns
    -------
    dict
        Keys: 'r2', 'mae', 'rmse', 'n_samples', 'mean_error', 'max_abs_error'
    """
    valid = df[["bms_soc_pct", "pred_soc_pct"]].dropna()
    y_true = valid["bms_soc_pct"].values
    y_pred = valid["pred_soc_pct"].values
    n      = len(y_true)

    residuals    = y_pred - y_true
    mae          = float(np.mean(np.abs(residuals)))
    rmse         = float(np.sqrt(np.mean(residuals ** 2)))
    ss_res       = float(np.sum(residuals ** 2))
    ss_tot       = float(np.sum((y_true - y_true.mean()) ** 2))
    r2           = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    mean_error   = float(np.mean(residuals))          # signed bias
    max_abs_err  = float(np.max(np.abs(residuals)))

    metrics = {
        "n_samples":     n,
        "r2":            r2,
        "mae":           mae,
        "rmse":          rmse,
        "mean_error":    mean_error,
        "max_abs_error": max_abs_err,
    }

    if print_result:
        sep = "─" * 44
        print(sep)
        print("  SoC Prediction Metrics  (Predicted vs. BMS)")
        print(sep)
        print(f"  Samples              : {n}")
        print(f"  R²                   : {r2:.6f}")
        print(f"  MAE                  : {mae:.4f} pp")
        print(f"  RMSE                 : {rmse:.4f} pp")
        print(f"  Mean error (bias)    : {mean_error:+.4f} pp")
        print(f"  Max absolute error   : {max_abs_err:.4f} pp")
        print(sep)

    # ── Diagnostic figure ─────────────────────────────────────────────────
    time = df["datetime_utc"].iloc[valid.index]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("SoC Prediction Performance", fontsize=13,
                 fontweight="bold")

    # ── Left: time-series overlay ─────────────────────────────────────────
    ax = axes[0]
    ax.plot(time, y_true, color=C["bms_soc"],  linewidth=1.4,
            label="BMS SoC")
    ax.plot(time, y_pred, color=C["pred_soc"], linewidth=1.2,
            linestyle="--", label="Predicted SoC")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("State of Charge (%)")
    ax.set_title("Time-Series Comparison")
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%H:%M"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.legend()

    # Annotate metrics on the time-series panel
    metric_txt = (f"$R^2$  = {r2:.4f}\n"
                  f"MAE  = {mae:.4f} pp\n"
                  f"RMSE = {rmse:.4f} pp")
    ax.text(0.02, 0.04, metric_txt, transform=ax.transAxes,
            fontsize=9.5, verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="0.75", alpha=0.9))

    # ── Right: scatter / parity plot ──────────────────────────────────────
    ax = axes[1]
    ax.scatter(y_true, y_pred, s=8, alpha=0.55,
               color=C["pred_soc"], edgecolors="none", label="Samples")

    # Perfect-prediction line
    lims = [min(y_true.min(), y_pred.min()) - 1,
            max(y_true.max(), y_pred.max()) + 1]
    ax.plot(lims, lims, color="0.3", linewidth=1.0,
            linestyle="--", label="Perfect prediction")

    # ±RMSE band
    ax.fill_between(lims,
                    [l - rmse for l in lims],
                    [l + rmse for l in lims],
                    color=C["pred_soc"], alpha=0.12, label=f"±RMSE band")

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("BMS SoC (%)")
    ax.set_ylabel("Predicted SoC (%)")
    ax.set_title("Parity Plot")
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.legend(fontsize=9)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"[soc_metrics] Figure saved to '{save_path}'")
    plt.show()

    return metrics


# ---------------------------------------------------------------------------
# CONVENIENCE: run all analyses on a single file
# ---------------------------------------------------------------------------

def analyse_all(filepath: str,
                nominal_capacity_Ah: float | None = None,
                save_dir: str | None = None):
    """
    Load a CSV and run every analysis function in sequence.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    nominal_capacity_Ah : float, optional
        Rated battery capacity for SoH calculation.
    save_dir : str, optional
        Directory to save all figures (e.g. "results/").
    """
    import os

    df = load_csv(filepath)
    print(f"\n[analyse_all] Loaded {len(df)} rows from '{filepath}'")

    def _path(name):
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            return os.path.join(save_dir, name)
        return None

    # 1. Overview grid
    plot_overview(df, save_path=_path("01_overview.pdf"))

    # 2. Inference time
    print()
    average_inference_time(df)

    # 3. ESP temperature cleaning
    print()
    cleaned_temp = clean_esp_temp(df, save_path=_path("03_esp_temp.pdf"))
    df["esp_temp_degC_clean"] = cleaned_temp

    # 4. Capacity
    print()
    calculate_actual_capacity(df, nominal_capacity_Ah=nominal_capacity_Ah)

    # 5. SOC metrics
    print()
    soc_metrics(df, save_path=_path("05_soc_metrics.pdf"))

    return df


# ---------------------------------------------------------------------------
# EXAMPLE USAGE (run this file directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(__doc__)
        print("\nUsage: python battery_analysis.py <path_to_csv> [nominal_capacity_Ah]")
        sys.exit(0)

    csv_file = sys.argv[1]
    nom_cap  = float(sys.argv[2]) if len(sys.argv) >= 3 else None
    analyse_all(csv_file, nominal_capacity_Ah=nom_cap, save_dir="output_figures")