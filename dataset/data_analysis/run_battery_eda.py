"""
run_battery_eda.py
====================

Command-line entry point for the battery_eda toolkit: loads a single HDF5
file, runs the full exploratory-data-analysis pipeline (data quality,
cleaning, time-series plots, battery-specific analysis, correlation,
feature engineering, anomaly detection, frequency analysis, usage stats),
saves every figure, exports an ML-ready CSV, and writes an automatic
Markdown + HTML report.

Usage
-----
    python -m dataset.data_analysis.run_battery_eda --h5 path/to/file.h5
    python -m dataset.data_analysis.run_battery_eda --h5 path/to/file.h5 --run run_hppc_01
    python -m dataset.data_analysis.run_battery_eda --h5 path/to/file.h5 --no-interactive

Run from the repository root so `dataset.*` imports resolve (see CLAUDE.md).
"""

from __future__ import annotations

import argparse
import os

from dataset.data_analysis.battery_eda import (
    anomaly_detection,
    config,
    correlation_analysis,
    cycle_analysis,
    data_cleaning,
    data_loader,
    feature_engineering,
    frequency_analysis,
    report,
    signal_analysis,
    soc_soh_analysis,
    usage_statistics,
    visualization as viz,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Battery HDF5 exploratory data analysis")
    parser.add_argument("--h5", required=True, help="Path to a single .h5 dataset file")
    parser.add_argument("--run", nargs="+", default=None,
                         help="Specific run name(s) to analyze (default: every run in the file)")
    parser.add_argument("--outdir", default=None,
                         help="Output directory (default: output_figures/battery_eda/<h5 stem>)")
    parser.add_argument("--no-interactive", action="store_true",
                         help="Use static Matplotlib figures instead of interactive Plotly")

    cleaning = parser.add_argument_group("cleaning")
    cleaning.add_argument("--no-interpolate", action="store_true",
                           help="Skip interpolating missing values")
    cleaning.add_argument("--no-remove-impossible", action="store_true",
                           help="Skip replacing physically-impossible readings with NaN")
    cleaning.add_argument("--clip-outliers", action="store_true",
                           help="Clip statistical outliers (IQR method) after interpolation")
    cleaning.add_argument("--smooth", nargs="*", default=["voltage", "current"],
                           help="Canonical signal names to add a smoothed column for "
                                "(default: voltage current; pass --smooth with no values to disable)")
    cleaning.add_argument("--smooth-method", default="moving_average",
                           choices=["moving_average", "ewm", "median", "savgol"])

    parser.add_argument("--force-fft", action="store_true",
                         help=f"Run per-run FFT even with more than "
                              f"{config.MAX_RUNS_FOR_FFT} runs loaded")
    parser.add_argument("--no-ml-export", action="store_true",
                         help="Skip building/exporting the ML-ready feature CSV")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    interactive = not args.no_interactive

    h5_stem = os.path.splitext(os.path.basename(args.h5))[0]
    outdir = args.outdir or os.path.join("output_figures", "battery_eda", h5_stem)
    figures_dir = os.path.join(outdir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # ------------------------------------------------------------------ 1. Load
    print(f"Loading {args.h5} ...")
    raw_df = data_loader.load_h5_to_dataframe(args.h5, run_names=args.run)
    quality_report = data_loader.assess_data_quality(raw_df)
    df, n_removed_dupes = data_loader.remove_duplicate_rows(raw_df)
    print(f"Loaded {len(raw_df)} rows across {df['run_name'].nunique()} run(s); "
          f"removed {n_removed_dupes} exact duplicate rows.")
    print(quality_report.as_text())

    # ------------------------------------------------------------------ 2. Summary
    summary_text = data_loader.summarize_dataset(df)
    print("\n" + summary_text)

    # ------------------------------------------------------------------ 3. Anomaly detection (on raw data)
    print("\nDetecting anomalies ...")
    anomalous_df, anomaly_summary = anomaly_detection.detect_anomalies(df)
    print(anomaly_summary.to_string(index=False))

    # ------------------------------------------------------------------ 4. Cleaning
    print("\nCleaning data ...")
    smooth_cols = [config.SIGNAL_MAP[name] for name in args.smooth if name in config.SIGNAL_MAP]
    clean_df, cleaning_log = data_cleaning.run_cleaning_pipeline(
        df,
        interpolate=not args.no_interpolate,
        remove_impossible=not args.no_remove_impossible,
        clip=args.clip_outliers,
        smooth_columns_list=smooth_cols,
        smooth_method=args.smooth_method,
    )
    for line in cleaning_log:
        print(f"  {line}")

    # ------------------------------------------------------------------ 5. Cycle / segment detection
    print("\nDetecting charge/discharge segments and cycles ...")
    cycle_result = cycle_analysis.run_cycle_analysis(clean_df)
    clean_df = cycle_result["df"]
    segments, cycles = cycle_result["segments"], cycle_result["cycles"]
    print(f"  {len(segments)} segments detected -> {len(cycles)} full discharge/charge cycles")

    # ------------------------------------------------------------------ 6. Feature engineering
    print("\nEngineering features ...")
    engineered_df = feature_engineering.engineer_all_features(clean_df)

    # ------------------------------------------------------------------ 7. Time-series overview
    print("\nPlotting major signals vs time ...")
    major_signal_groups = {
        "pack_voltage": [config.SIGNAL_MAP["voltage"]],
        "current": [config.SIGNAL_MAP["current"]],
        "power": [config.SIGNAL_MAP["power"]],
        "temperature": [config.SIGNAL_MAP["temperature"]],
        "soc": [config.SIGNAL_MAP["soc"]],
        "motor_speed": [config.SIGNAL_MAP["motor_speed_r"], config.SIGNAL_MAP["motor_speed_l"]],
        "hoverboard_voltage": [config.SIGNAL_MAP["hb_voltage"]],
        "charging_status": [config.SIGNAL_MAP["charging_flag"]],
    }
    overview_cols = [c for cols in major_signal_groups.values() for c in cols if c in engineered_df.columns]
    overview_fig_path = viz.plot_timeseries(
        engineered_df, overview_cols, figures_dir, "00_overview_all_signals",
        title="All major signals vs time", interactive=interactive,
    )

    # ------------------------------------------------------------------ 8. Battery-specific analysis
    print("Running current / voltage / temperature / power / energy / resistance analysis ...")
    current_result = signal_analysis.current_analysis(engineered_df, figures_dir, interactive)
    voltage_result = signal_analysis.voltage_analysis(engineered_df, figures_dir, interactive)
    cell_voltage_result = signal_analysis.cell_voltage_analysis(engineered_df, figures_dir, interactive)
    temperature_result = signal_analysis.temperature_analysis(engineered_df, figures_dir, interactive)
    power_result = signal_analysis.power_analysis(engineered_df, figures_dir, interactive)
    energy_result = signal_analysis.energy_analysis(engineered_df)
    resistance_result = signal_analysis.resistance_analysis(engineered_df, figures_dir, interactive)

    print("Running SOC / SOH analysis ...")
    soc_result = soc_soh_analysis.soc_analysis(engineered_df, figures_dir, interactive)
    soh_result = soc_soh_analysis.soh_indicators(
        cycles, resistance_result.get("resistance_df"), figures_dir, interactive
    )

    # ------------------------------------------------------------------ 9. Correlation analysis
    print("Running correlation analysis ...")
    correlation_result = correlation_analysis.run_correlation_analysis(
        engineered_df, figures_dir, interactive
    )
    soh_ranking = correlation_analysis.rank_soh_features(cycles)

    # ------------------------------------------------------------------ 10. Frequency analysis
    n_runs = engineered_df["run_name"].nunique()
    dominant_frequencies = None
    if n_runs <= config.MAX_RUNS_FOR_FFT or args.force_fft:
        print("Running FFT analysis ...")
        freq_result = frequency_analysis.run_frequency_analysis(engineered_df, figures_dir, interactive)
        dominant_frequencies = freq_result["dominant_frequencies"]
    else:
        print(f"Skipping FFT: {n_runs} runs loaded (> {config.MAX_RUNS_FOR_FFT}); "
              f"pass --force-fft or --run <name> to restrict scope.")

    # ------------------------------------------------------------------ 11. Usage statistics
    usage_stats = usage_statistics.compute_usage_statistics(engineered_df)
    print("\n" + usage_statistics.usage_statistics_as_text(usage_stats))

    # ------------------------------------------------------------------ 12. ML-ready export
    if not args.no_ml_export:
        print("\nBuilding ML-ready dataframe ...")
        ml_df = feature_engineering.build_ml_dataframe(engineered_df)
        ml_csv_path = os.path.join(outdir, f"{h5_stem}_ml_features.csv")
        feature_engineering.export_ml_dataframe(ml_df, ml_csv_path)
        print(f"  saved {ml_csv_path} ({ml_df.shape[0]} rows x {ml_df.shape[1]} columns)")

    # ------------------------------------------------------------------ 13. Report
    print("\nAssembling report ...")
    findings = _build_findings(quality_report, anomaly_summary, cycles, cell_voltage_result)
    key_figures = {
        "All major signals vs time": overview_fig_path,
        "Voltage vs SOC": voltage_result["figures"].get("voltage_vs_soc"),
        "Pearson correlation heatmap": correlation_result["figures"].get("pearson_heatmap"),
        "Cell imbalance vs time": cell_voltage_result["figures"].get("cell_imbalance_timeseries"),
        "Resistance vs SOC": resistance_result["figures"].get("resistance_vs_soc"),
    }
    key_figures = {k: v for k, v in key_figures.items() if v}

    context = {
        "h5_path": args.h5,
        "run_names": args.run or "all",
        "quality_report": quality_report,
        "cleaning_log": cleaning_log,
        "anomaly_summary": anomaly_summary,
        "usage_stats": usage_stats,
        "cycles": cycles,
        "soh_stats": soh_result["stats"],
        "soc_ranking": correlation_result["soc_feature_ranking"],
        "soh_ranking": soh_ranking,
        "key_figures": key_figures,
        "findings": findings,
    }
    rb = report.generate_report(context)
    saved_paths = rb.save(outdir, basename="eda_report")

    print("\nDone. Outputs:")
    print(f"  figures:  {figures_dir}")
    print(f"  report:   {saved_paths['markdown']}")
    print(f"            {saved_paths['html']}")


def _build_findings(quality_report, anomaly_summary, cycles, cell_voltage_result) -> list[str]:
    """Turn the numeric results into a short bullet list of plain-English findings."""
    findings = []

    if quality_report.n_duplicate_rows:
        findings.append(f"{quality_report.n_duplicate_rows} exact duplicate rows were removed.")
    if quality_report.large_time_gaps:
        findings.append(
            f"{len(quality_report.large_time_gaps)} large timestamp gaps detected "
            "(likely BLE/serial communication dropouts)."
        )
    if quality_report.constant_columns:
        findings.append(f"Constant (zero-variance) columns found: {quality_report.constant_columns}")

    total_anomalies = int(anomaly_summary["n_flagged"].sum()) if not anomaly_summary.empty else 0
    if total_anomalies:
        top = anomaly_summary.iloc[0]
        findings.append(
            f"{total_anomalies} anomaly flags raised in total; most common: "
            f"{top['anomaly_type']} ({int(top['n_flagged'])} samples)."
        )
    else:
        findings.append("No anomalies were flagged by any check.")

    if not cycles.empty:
        findings.append(f"{len(cycles)} full discharge->charge cycle(s) detected.")
    else:
        findings.append("No full discharge->charge cycle was detected (data likely covers "
                         "partial charge or discharge sessions only).")

    stats = cell_voltage_result.get("stats", {})
    if stats.get("max_imbalance_v") is not None:
        findings.append(f"Max observed cell-voltage imbalance: {stats['max_imbalance_v']:.3f} V "
                         f"(mean: {stats.get('mean_imbalance_v', float('nan')):.3f} V).")

    return findings


if __name__ == "__main__":
    main()
