"""
Shared HPPC step/pulse analysis core, used by both hppc_2ndlife_analysis.py
and hppc_newpack_analysis.py.

Segments a run's `hoverboard/hppc_phase` trace (as logged by hppc_run.py /
the DC-load HPPC variant) into OCV_REST / DISCHARGE_PULSE / REST_BETWEEN /
CHARGE_PULSE / REST_AFTER / STEP_DISCHARGE blocks, then builds one row per
OCV_REST checkpoint: its rest OCV, the Coulomb-counted Ah removed to reach
it (measured directly from raw current -- independent of the BMS's own
battery_level%), and the discharge/charge pulse internal resistance that
follows it.

Deliberately does NOT compute a capacity/SOH number here -- see the
pack-specific scripts for why that number needs care (or isn't defensible)
for a given run.
"""

import os

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PHASE_OCV_REST, PHASE_DISCHARGE_PULSE, PHASE_REST_BETWEEN = 0, 1, 2
PHASE_CHARGE_PULSE, PHASE_REST_AFTER, PHASE_STEP_DISCHARGE = 4, 6, 7

REST_AVG_SAMPLES = 30  # samples averaged at the end of a rest segment for a stable OCV


def load_runs(h5_file, run_names):
    """Concatenate one or more runs into a single DataFrame with a continuous
    time axis. Within a run, real per-sample timestamps are used (accurate
    Coulomb counting). Between runs, the gap is collapsed to a nominal 1s
    step regardless of the real wall-clock gap between sessions -- runs are
    only split where the protocol was at rest, so no charge was missed, and
    the real gap can be hours/days apart, which would otherwise corrupt the
    integration if current is even slightly non-zero at the boundary."""
    frames = []
    offset = 0.0
    with h5py.File(h5_file, "r") as f:
        for run_name in run_names:
            g = f[run_name]
            ts = g["timestamp_ms"][()] / 1000.0
            t_local = ts - ts[0]
            df = pd.DataFrame({
                "t_s":     t_local + offset,
                "phase":   g["hoverboard/hppc_phase"][()],
                "voltage": g["bms/voltage"][()].astype(np.float64),
                "current": g["bms/current"][()].astype(np.float64),
                "bms_soc": g["bms/battery_level"][()],
                "run_name": run_name,
            })
            frames.append(df)
            offset = df["t_s"].iloc[-1] + 1.0
    return pd.concat(frames, ignore_index=True)


def contiguous_segments(phase):
    """[(phase_value, start_idx, end_idx_inclusive), ...] in order."""
    segs = []
    start = 0
    for i in range(1, len(phase) + 1):
        if i == len(phase) or phase[i] != phase[start]:
            segs.append((int(phase[start]), start, i - 1))
            start = i
    return segs


def cumulative_ah_removed(t_s, current_a):
    """Trapezoidal Coulomb count of removed Ah (discharge current is negative)."""
    dq = -(current_a[:-1] + current_a[1:]) / 2.0 * np.diff(t_s)
    return np.concatenate([[0.0], np.cumsum(dq)]) / 3600.0


def rest_ocv(df, seg):
    _, start, end = seg
    n = min(REST_AVG_SAMPLES, end - start + 1)
    return float(df["voltage"].iloc[end - n + 1:end + 1].mean())


def pulse_stats(df, seg):
    """Mean current and end-of-pulse voltage for a discharge/charge pulse segment."""
    _, start, end = seg
    i_mean = float(df["current"].iloc[start:end + 1].mean())
    v_end = float(df["voltage"].iloc[max(start, end - 2):end + 1].mean())
    return i_mean, v_end


def build_step_table(df, cum_ah, segs):
    """One row per OCV_REST checkpoint, with the pulse cycle that follows it (if any)."""
    rows = []
    ocv_idx = [k for k, s in enumerate(segs) if s[0] == PHASE_OCV_REST]

    for step, k in enumerate(ocv_idx):
        seg = segs[k]
        ocv_v = rest_ocv(df, seg)
        row = {
            "step": step,
            "bms_soc_label_pct": float(df["bms_soc"].iloc[seg[2]]),
            "rest_ocv_V": ocv_v,
            "cum_ah_removed": float(cum_ah[seg[2]]),
        }

        # Collect the segments between this OCV_REST and the next one (the pulse cycle).
        next_k = ocv_idx[step + 1] if step + 1 < len(ocv_idx) else len(segs)
        cycle = segs[k + 1:next_k]
        by_phase = {}
        for s in cycle:
            by_phase.setdefault(s[0], []).append(s)

        if PHASE_DISCHARGE_PULSE in by_phase:
            i_dis, v_dis = pulse_stats(df, by_phase[PHASE_DISCHARGE_PULSE][0])
            r_dis = (ocv_v - v_dis) / abs(i_dis) if i_dis != 0 else np.nan
            row.update({"I_dis_A": i_dis, "V_dis_V": v_dis, "R_int_dis_ohm": r_dis})
        else:
            row.update({"I_dis_A": np.nan, "V_dis_V": np.nan, "R_int_dis_ohm": np.nan})

        if PHASE_REST_BETWEEN in by_phase:
            row["ocv_recovery_V"] = rest_ocv(df, by_phase[PHASE_REST_BETWEEN][0])
        else:
            row["ocv_recovery_V"] = np.nan

        if PHASE_CHARGE_PULSE in by_phase:
            i_chg, v_chg = pulse_stats(df, by_phase[PHASE_CHARGE_PULSE][0])
            baseline = row["ocv_recovery_V"] if not np.isnan(row["ocv_recovery_V"]) else ocv_v
            r_chg = (v_chg - baseline) / abs(i_chg) if i_chg != 0 else np.nan
            row.update({"I_chg_A": i_chg, "V_chg_V": v_chg, "R_int_chg_ohm": r_chg})
        else:
            row.update({"I_chg_A": np.nan, "V_chg_V": np.nan, "R_int_chg_ohm": np.nan})

        if PHASE_REST_AFTER in by_phase:
            row["ocv_post_chg_V"] = rest_ocv(df, by_phase[PHASE_REST_AFTER][0])
        else:
            row["ocv_post_chg_V"] = np.nan

        rows.append(row)

    table = pd.DataFrame(rows)
    table["step_ah_removed"] = table["cum_ah_removed"].diff().fillna(0.0)
    return table


def plot_full_trace(df, out_dir, label, run_title):
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(df["t_s"] / 3600.0, df["voltage"], color="#4e79a7", lw=0.8)
    axes[0].set_ylabel("Voltage (V)")
    axes[1].plot(df["t_s"] / 3600.0, df["current"], color="#e15759", lw=0.8)
    axes[1].axhline(0, color="#888888", lw=0.8, ls="--")
    axes[1].set_ylabel("Current (A)")
    axes[2].plot(df["t_s"] / 3600.0, df["bms_soc"], color="#59a14f", lw=0.8)
    axes[2].set_ylabel("BMS battery_level (%)")
    axes[2].set_xlabel("Time (h)")
    fig.suptitle(f"HPPC {label} -- full trace ({run_title})")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "01_full_trace.png"), dpi=150)
    plt.close(fig)


def plot_ocv_vs_ah(table, out_dir, label):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(table["cum_ah_removed"], table["rest_ocv_V"], "o-", color="#4e79a7")
    for _, r in table.iterrows():
        ax.annotate(f"{r['bms_soc_label_pct']:.0f}% (BMS)",
                    (r["cum_ah_removed"], r["rest_ocv_V"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8, color="#888888")
    ax.set_xlabel("Cumulative Ah removed (Coulomb-counted, measured)")
    ax.set_ylabel("Rest OCV (V)")
    ax.set_title(f"{label}: rest-OCV vs Ah removed\n"
                  "(BMS % labels shown for reference only -- not an independent SOC)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "02_ocv_vs_ah_removed.png"), dpi=150)
    plt.close(fig)


def plot_rint(table, out_dir, label):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(table["cum_ah_removed"], table["R_int_dis_ohm"] * 1000, "o-",
            color="#e15759", label="Discharge pulse")
    ax.plot(table["cum_ah_removed"], table["R_int_chg_ohm"] * 1000, "o-",
            color="#59a14f", label="Charge pulse")
    ax.set_xlabel("Cumulative Ah removed")
    ax.set_ylabel("Internal resistance (mOhm)")
    ax.set_title(f"{label}: pulse internal resistance vs Ah removed")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "03_rint_vs_ah_removed.png"), dpi=150)
    plt.close(fig)


def plot_pulse_detail(df, segs, out_dir, label, step_index=3):
    ocv_idx = [k for k, s in enumerate(segs) if s[0] == PHASE_OCV_REST]
    if step_index >= len(ocv_idx) - 1:
        return
    k = ocv_idx[step_index]
    next_k = ocv_idx[step_index + 1]
    lo = max(0, segs[k][2] - 30)
    hi = min(len(df) - 1, segs[next_k][1] + 30)
    window = df.iloc[lo:hi + 1]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    t0 = window["t_s"].iloc[0]
    axes[0].plot((window["t_s"] - t0), window["voltage"], color="#4e79a7", lw=1.2)
    axes[0].set_ylabel("Voltage (V)")
    axes[1].plot((window["t_s"] - t0), window["current"], color="#e15759", lw=1.2)
    axes[1].axhline(0, color="#888888", lw=0.8, ls="--")
    axes[1].set_ylabel("Current (A)")
    axes[1].set_xlabel("Time (s, relative)")
    fig.suptitle(f"{label}: pulse-cycle detail -- step {step_index} "
                 f"(BMS label {df['bms_soc'].iloc[segs[k][2]]:.0f}%)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "04_pulse_detail.png"), dpi=150)
    plt.close(fig)


def run_analysis(h5_file, run_names, out_dir, label, csv_name, pulse_detail_step=3):
    """Full pipeline: load -> segment -> step table -> CSV + 4 plots.
    Returns (df, table) for any extra pack-specific reporting the caller wants."""
    os.makedirs(out_dir, exist_ok=True)
    df = load_runs(h5_file, run_names)
    cum_ah = cumulative_ah_removed(df["t_s"].values, df["current"].values)
    df["cum_ah_removed"] = cum_ah
    segs = contiguous_segments(df["phase"].values)
    table = build_step_table(df, cum_ah, segs)

    csv_path = os.path.join(out_dir, csv_name)
    table.to_csv(csv_path, index=False)

    run_title = run_names[0] if len(run_names) == 1 else f"{run_names[0]}..{run_names[-1]}"
    plot_full_trace(df, out_dir, label, run_title)
    plot_ocv_vs_ah(table, out_dir, label)
    plot_rint(table, out_dir, label)
    plot_pulse_detail(df, segs, out_dir, label, step_index=pulse_detail_step)

    print(table.to_string(index=False))
    print(f"\nResults saved -> {csv_path}")
    print(f"Figures saved -> {out_dir}")
    return df, table
