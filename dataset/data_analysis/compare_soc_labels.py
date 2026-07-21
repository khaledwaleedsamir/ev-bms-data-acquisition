"""
Compare the Daly BMS SOC label (battery_level) against the OCV-anchored
coulomb-counted label (soc_cc) for every run in the dataset.

Saves one PNG per run plus a summary CSV to output_figures/soc_labels/.

Run from repo root:
    python -m dataset.data_analysis.compare_soc_labels
"""

import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from soc_estimation.soc_relabel import compute_soc_cc

H5_PATH = Path("dataset/all_data/h5_files/hoverboard_bms_dataset.h5")
OUT_DIR = Path("output_figures/soc_labels")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    rows = []
    with h5py.File(H5_PATH, "r") as f:
        for run in f.keys():
            g = f[run]
            ts = g["timestamp_ms"][:]
            v = g["bms"]["voltage"][:]
            i = g["bms"]["current"][:]
            soc_bms = g["bms"]["battery_level"][:].astype(float)

            soc_cc = compute_soc_cc(ts, v, i)
            t_min = (ts - ts[0]) / 60000.0

            valid = ~np.isnan(soc_cc)
            bias = soc_bms[valid] - soc_cc[valid]
            rows.append({
                "run": run,
                "soc_cc_start": soc_cc[valid][0],
                "soc_cc_end": soc_cc[valid][-1],
                "soc_bms_start": soc_bms[valid][0],
                "soc_bms_end": soc_bms[valid][-1],
                "bms_minus_cc_mean": bias.mean(),
                "bms_minus_cc_max_abs": np.abs(bias).max(),
            })

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(t_min, soc_bms, color="0.4", linewidth=1.2,
                    label="BMS battery_level")
            ax.plot(t_min, soc_cc, color="tab:red", linewidth=1.2,
                    label="soc_cc (OCV-anchored coulomb count)")
            ax.set_xlabel("Time (min)")
            ax.set_ylabel("SOC (%)")
            ax.set_title(f"{run} — mean BMS bias {bias.mean():+.1f} pp")
            ax.legend(loc="best", fontsize=9)
            ax.grid(True, linestyle="--", alpha=0.5)
            fig.tight_layout()
            fig.savefig(OUT_DIR / f"{run}.png", dpi=150)
            plt.close(fig)

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "label_comparison_summary.csv", index=False)

    pd.set_option("display.width", 200)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:7.1f}"))
    print(f"\nPlots + summary saved -> {OUT_DIR}")


if __name__ == "__main__":
    main()
