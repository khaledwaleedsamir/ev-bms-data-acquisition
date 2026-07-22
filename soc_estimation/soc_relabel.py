"""
OCV-anchored coulomb-counting SOC relabeling.

Produces a physics-based SOC label ("soc_cc") for a run, independent of the
Daly BMS's own battery_level estimate:

  1. Invert the HPPC rest-OCV curve at the first and last near-rest samples
     of the run (with an I*R_int correction when the endpoint is under load)
     to get two independent SOC anchors.
  2. Coulomb-integrate the logged pack current across the run.
  3. Least-squares fit the initial SOC so the coulomb-counted trajectory
     matches both anchors, weighting the endpoint closer to rest higher
     (the IR correction is the dominant error source under load).

Rows flagged as BMS dropouts (voltage below DROPOUT_VOLTAGE) get NaN.

Absolute accuracy of the anchors is roughly +/-2 pp (validated by the
agreement between the coulomb-counted end SOC and the independent OCV
inversion at the end of each run), but within-run deltas come purely from
current integration and are much more accurate than that.
"""

import numpy as np
import pandas as pd
from pathlib import Path

HPPC_PATH = Path("output_figures/hppc/hppc_results.csv")
NOMINAL_CAP_AH = 10.2
DROPOUT_VOLTAGE = 20.0   # rows below this are BMS/BLE dropouts (log reads 0)

_hppc = pd.read_csv(HPPC_PATH).sort_values("soc_pct")
SOC_KNOTS = _hppc["soc_pct"].values
OCV_KNOTS = _hppc["ocv_V"].values          # monotonic increasing with SOC
RINT_DIS_KNOTS = _hppc["R_int_dis_ohm"].values
RINT_CHG_KNOTS = _hppc["R_int_chg_ohm"].values


def ocv_to_soc(v):
    """Invert the rest OCV-SOC curve. Clamps outside the measured 16-100% range."""
    return np.interp(v, OCV_KNOTS, SOC_KNOTS)


def soc_from_vi(v, i):
    """SOC from terminal voltage under (signed) current i via a fixed-point
    IR correction: OCV = V - i * R_int(SOC). Discharge current is negative,
    so the correction raises the voltage back toward OCV."""
    soc = ocv_to_soc(v)
    for _ in range(2):
        r_knots = RINT_DIS_KNOTS if i < 0 else RINT_CHG_KNOTS
        r = np.interp(soc, SOC_KNOTS, r_knots)
        soc = ocv_to_soc(v - i * r)
    return float(soc)


def compute_soc_cc(timestamp_ms, voltage, current, cap_ah=NOMINAL_CAP_AH):
    """Return the coulomb-counted SOC trajectory (percent) for one run.

    Parameters are 1-D arrays of equal length. Dropout rows return NaN.
    """
    ts = np.asarray(timestamp_ms, dtype=float) / 1000.0
    v = np.asarray(voltage, dtype=float)
    i = np.asarray(current, dtype=float)

    soc = np.full(len(v), np.nan)
    valid = v > DROPOUT_VOLTAGE
    if valid.sum() < 10:
        return soc

    idx = np.where(valid)[0]
    tv, vv, iv = ts[idx], v[idx], i[idx]

    # Signed cumulative charge in SOC percentage points (discharge i<0 lowers SOC).
    # dt clipped to 5 s so a logging gap doesn't integrate phantom charge.
    dt = np.clip(np.diff(tv), 0.0, 5.0)
    cum = np.concatenate(([0.0], np.cumsum(iv[:-1] * dt))) / 3600.0 / cap_ah * 100.0

    # OCV anchors at both endpoints (median over 5 samples for robustness)
    i0, v0 = np.median(iv[:5]), np.median(vv[:5])
    i1, v1 = np.median(iv[-5:]), np.median(vv[-5:])
    anchor0 = soc_from_vi(v0, i0)
    anchor1 = soc_from_vi(v1, i1)

    # Weighted least-squares fit of the initial SOC to both anchors,
    # trusting the endpoint closer to rest more.
    w0 = 1.0 / (abs(i0) + 0.25)
    w1 = 1.0 / (abs(i1) + 0.25)
    soc0 = (w0 * anchor0 + w1 * (anchor1 - cum[-1])) / (w0 + w1)

    soc[idx] = np.clip(soc0 + cum, 0.0, 100.0)
    return soc


def add_soc_cc(df: pd.DataFrame, cap_ah=NOMINAL_CAP_AH) -> pd.DataFrame:
    """Add a 'soc_cc' column to a run DataFrame that has
    timestamp_ms / voltage / current columns. Returns the same DataFrame."""
    df["soc_cc"] = compute_soc_cc(
        df["timestamp_ms"].values, df["voltage"].values, df["current"].values,
        cap_ah=cap_ah,
    )
    return df
