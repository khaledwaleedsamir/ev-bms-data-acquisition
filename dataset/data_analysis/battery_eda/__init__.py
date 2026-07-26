"""
battery_eda
===========

A modular exploratory-data-analysis toolkit for the HDF5 datasets produced by
this repository's acquisition scripts (`dataset/run_scripts/`).

Each analysis concern lives in its own module so it can be read, tested, and
edited independently:

    config.py               shared constants (schema map, physical limits, thresholds)
    data_loader.py           HDF5 -> tidy pandas.DataFrame, data-quality reporting
    data_cleaning.py         interpolation, impossible-value removal, clipping, smoothing
    feature_engineering.py   rolling stats, derivatives, cumulative Ah/Wh, ML-ready dataframe
    cycle_analysis.py        charge/discharge segment + cycle detection, capacity trends
    signal_analysis.py       current / voltage / cell-voltage / temperature / power / energy / R
    soc_soh_analysis.py      SOC-relationship plots and SOH degradation indicators
    correlation_analysis.py  Pearson/Spearman matrices, heatmaps, pair plots, feature ranking
    anomaly_detection.py     spike / dropout / impossible-value / discontinuity flags
    frequency_analysis.py    FFT of current & voltage
    usage_statistics.py      operating-time and throughput summary statistics
    visualization.py         shared matplotlib/Plotly plotting + figure-saving helpers
    report.py                assembles all findings into a Markdown/HTML report

The command-line entry point that wires all of this together is
`dataset/data_analysis/run_battery_eda.py`.
"""
