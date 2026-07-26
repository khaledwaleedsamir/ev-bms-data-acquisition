"""
frequency_analysis.py
========================

FFT of current and voltage, per run, to spot periodic oscillations (e.g. the
demo run's sinusoidal speed profile) or switching/control-loop noise.

Caveat worth keeping in mind when reading the results: acquisition runs at
roughly 1 Hz (see CLAUDE.md / dataset_analysis_findings), so the Nyquist
frequency is only ~0.5 Hz - this can resolve slow periodic load cycling
(seconds-to-tens-of-seconds period) but cannot see genuine high-frequency
switching noise from the motor controller or BMS balancing circuitry, which
would need a much higher logging rate to observe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, visualization as viz


def fft_spectrum(series: pd.Series, elapsed_s: pd.Series) -> pd.DataFrame:
    """
    Compute the single-sided amplitude spectrum of `series`, resampled onto
    a uniform time grid first (FFT assumes uniform sampling; the real
    acquisition loop has small jitter around its nominal 1 Hz rate).
    """
    valid = series.notna() & elapsed_s.notna()
    t, y = elapsed_s[valid].to_numpy(), series[valid].to_numpy()
    if len(t) < 8:
        return pd.DataFrame(columns=["frequency_hz", "amplitude"])

    dt = np.median(np.diff(t))
    if dt <= 0:
        return pd.DataFrame(columns=["frequency_hz", "amplitude"])

    uniform_t = np.arange(t[0], t[-1], dt)
    uniform_y = np.interp(uniform_t, t, y)
    uniform_y = uniform_y - uniform_y.mean()  # remove DC offset before transforming

    n = len(uniform_y)
    windowed = uniform_y * np.hanning(n)
    spectrum = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(n, d=dt)
    amplitude = (2.0 / n) * np.abs(spectrum)

    return pd.DataFrame({"frequency_hz": freqs, "amplitude": amplitude})


def run_frequency_analysis(df: pd.DataFrame, outdir: str, interactive: bool = True) -> dict:
    """FFT of current and voltage for each run; saves one spectrum plot per signal per run."""
    v_col, i_col = config.SIGNAL_MAP["voltage"], config.SIGNAL_MAP["current"]
    figures, dominant_frequencies = {}, []

    for run_name, g in df.groupby("run_name"):
        for col, label in ((i_col, "current"), (v_col, "voltage")):
            if col not in g.columns:
                continue
            spectrum = fft_spectrum(g[col], g["elapsed_s"])
            if spectrum.empty:
                continue

            # Skip the (near-)zero-frequency bin - it's the residual DC
            # trend, not an oscillation, and would otherwise dominate.
            non_dc = spectrum[spectrum["frequency_hz"] > 0]
            if not non_dc.empty:
                peak = non_dc.loc[non_dc["amplitude"].idxmax()]
                dominant_frequencies.append({
                    "run_name": run_name, "signal": label,
                    "dominant_frequency_hz": peak["frequency_hz"],
                    "amplitude": peak["amplitude"],
                    "dominant_period_s": (1.0 / peak["frequency_hz"]) if peak["frequency_hz"] else np.nan,
                })

            name = f"fft_{label}_{run_name}"
            figures[name] = viz.plot_line(
                spectrum["frequency_hz"], spectrum["amplitude"], outdir, name,
                xlabel="frequency (Hz)", ylabel="amplitude",
                title=f"FFT of {label} - {run_name}", interactive=interactive,
            )

    return {"figures": figures, "dominant_frequencies": pd.DataFrame(dominant_frequencies)}
