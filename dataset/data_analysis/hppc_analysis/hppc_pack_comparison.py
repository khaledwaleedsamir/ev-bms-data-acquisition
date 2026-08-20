"""
Compares the 2nd-life pack against the original "new" pack using only what's
independently measurable from each run: Coulomb-counted Ah removed vs. rest
OCV (see hppc_2ndlife_analysis.py / hppc_newpack_analysis.py for why the
BMS's own battery_level% and a cross-pack OCV-SOC mapping are NOT used).

The comparison is Ah-removed-between-two-rest-voltages, restricted to the
voltage window BOTH packs actually traversed:

    2nd-life pack : rest OCV 41.70 V -> 35.40 V
    new pack      : rest OCV 41.00 V -> 30.26 V
    overlap       : 41.00 V -> 35.40 V

Within that overlap, "Ah removed to bring the rested pack from OCV_hi down
to OCV_lo" is directly comparable between the two packs -- it needs no
assumption that they share an OCV-SOC curve, only that both voltage sensors
read the same physical quantity in the same units.

This is NOT a full-pack capacity comparison: neither run discharged to an
actual voltage cutoff, and the two packs weren't tested over the same
absolute voltage span (2nd-life never went below 35.40 V; new pack was
pushed all the way to 30.26 V). It only answers "which pack delivers more
charge across the shared window," which is a real degradation/health
signal, not a full-capacity or full-SOH number.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dataset.data_analysis.hppc_analysis import hppc_common as common

OUT_DIR = os.path.join("output_figures", "hppc_comparison")

PACKS = {
    "2nd-life": dict(
        h5_file=os.path.join("dataset", "all_data", "h5_files", "dc_load_bms_HPPC_data_2ndlife.h5"),
        run_names=["run_hppc_dc_load_01"],
        color="#e15759",
    ),
    "new": dict(
        h5_file=os.path.join("dataset", "all_data", "h5_files", "hoverboard_bms_HPPC_data.h5"),
        run_names=[f"run_hppc_{i:02d}" for i in range(1, 8)],
        color="#4e79a7",
    ),
}


def ocv_ah_curve(h5_file, run_names):
    """Return (ocv_ascending, cum_ah_matching) from a run's OCV_REST checkpoints,
    ready for np.interp (which requires ascending x)."""
    df = common.load_runs(h5_file, run_names)
    cum_ah = common.cumulative_ah_removed(df["t_s"].values, df["current"].values)
    segs = common.contiguous_segments(df["phase"].values)
    table = common.build_step_table(df, cum_ah, segs)
    ordered = table.sort_values("rest_ocv_V")
    return ordered["rest_ocv_V"].values, ordered["cum_ah_removed"].values


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    curves = {name: ocv_ah_curve(cfg["h5_file"], cfg["run_names"]) for name, cfg in PACKS.items()}

    ocv_hi = min(ocv.max() for ocv, _ in curves.values())
    ocv_lo = max(ocv.min() for ocv, _ in curves.values())
    print(f"Overlapping rest-OCV window tested by both packs: {ocv_lo:.3f} V - {ocv_hi:.3f} V\n")

    grid = np.arange(ocv_lo, ocv_hi + 1e-9, 0.2)
    rows = []
    for v in grid:
        row = {"ocv_V": round(v, 2)}
        for name, (ocv, cum_ah) in curves.items():
            row[f"cum_ah_{name}"] = float(np.interp(v, ocv, cum_ah))
        rows.append(row)

    import pandas as pd
    table = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, "hppc_pack_comparison.csv")
    table.to_csv(csv_path, index=False)
    print(table.to_string(index=False))

    ah_span = {}
    for name, (ocv, cum_ah) in curves.items():
        ah_span[name] = float(np.interp(ocv_lo, ocv, cum_ah) - np.interp(ocv_hi, ocv, cum_ah))
    print(f"\nAh removed across the shared window ({ocv_hi:.2f}V -> {ocv_lo:.2f}V):")
    for name, ah in ah_span.items():
        print(f"  {name:10s}: {ah:.3f} Ah")
    ratio = ah_span["2nd-life"] / ah_span["new"]
    print(f"\n2nd-life pack delivers {ratio*100:.1f}% of the new pack's Ah across this shared window.")
    print("(Not a full-capacity ratio -- see module docstring. Both packs likely have "
          "usable charge outside this window that isn't captured here.)")

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, (ocv, cum_ah) in curves.items():
        ax.plot(cum_ah, ocv, "o-", color=PACKS[name]["color"], label=f"{name} pack")
    ax.axhspan(ocv_lo, ocv_hi, color="gray", alpha=0.08, label="shared tested window")
    ax.set_xlabel("Cumulative Ah removed (Coulomb-counted, measured)")
    ax.set_ylabel("Rest OCV (V)")
    ax.set_title("Rest-OCV vs Ah removed -- 2nd-life pack vs new pack")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "pack_comparison_ocv_vs_ah.png"), dpi=150)
    plt.close(fig)

    print(f"\nResults saved -> {csv_path}")
    print(f"Figure saved  -> {os.path.join(OUT_DIR, 'pack_comparison_ocv_vs_ah.png')}")


if __name__ == "__main__":
    main()
