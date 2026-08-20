"""
OCV-vs-SOC and DCIR-vs-SOC profile plots for the new and 2nd-life packs,
built from each pack's HPPC step table (hppc_common.build_step_table).

SOC-axis caveat -- read before trusting these plots:
The x-axis here is the BMS's own battery_level% label at each OCV_REST
checkpoint (bms_soc_label_pct), not an independently measured SOC.
  - New pack: the BMS is calibrated to a fresh, ~10.2 Ah pack, and its SOC
    labels have no known reason to diverge from reality (see
    hppc_newpack_analysis.py's docstring).
  - 2nd-life pack: hppc_2ndlife_analysis.py already found this pack's BMS
    SOC% is circular (coulomb-counted against a *configured* 10.2 Ah
    nominal capacity, not this pack's real one) and that no independent
    OCV-SOC mapping is available for it. Its x-axis here is the BMS's
    assumed SOC label only -- read its curve *shape* (where DCIR rises,
    where OCV knees), not the absolute SOC% values, and do not compare it
    point-for-point against the new pack's SOC axis.

Also: neither run necessarily spans the full 0-100% range -- each plot
shows whatever range that pack's run actually covered.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset.data_analysis.hppc_analysis import hppc_common as common

OUT_DIR = os.path.join("output_figures", "hppc_ocv_dcir_profiles")

PACKS = {
    "new pack": dict(
        h5_file=os.path.join("dataset", "all_data", "h5_files", "hoverboard_bms_HPPC_data.h5"),
        run_names=[f"run_hppc_{i:02d}" for i in range(1, 8)],
        color="#4e79a7",
    ),
    "2nd-life pack": dict(
        h5_file=os.path.join("dataset", "all_data", "h5_files", "dc_load_bms_HPPC_data_2ndlife.h5"),
        run_names=["run_hppc_dc_load_01"],
        color="#e15759",
    ),
}


def build_table(cfg):
    df = common.load_runs(cfg["h5_file"], cfg["run_names"])
    cum_ah = common.cumulative_ah_removed(df["t_s"].values, df["current"].values)
    segs = common.contiguous_segments(df["phase"].values)
    table = common.build_step_table(df, cum_ah, segs)
    return table.sort_values("bms_soc_label_pct").reset_index(drop=True)


def plot_ocv_vs_soc(tables, out_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, table in tables.items():
        ax.plot(table["bms_soc_label_pct"], table["rest_ocv_V"], "o-",
                color=PACKS[name]["color"], label=name)
    ax.set_xlabel("SOC (%) -- BMS battery_level label at OCV_REST checkpoint")
    ax.set_ylabel("Rest OCV (V)")
    ax.set_xlim(0, 100)
    ax.set_title("OCV vs SOC")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "01_ocv_vs_soc.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_dcir_vs_soc(tables, out_dir):
    fig, axes = plt.subplots(1, len(tables), figsize=(7 * len(tables), 6), sharey=False)
    if len(tables) == 1:
        axes = [axes]
    for ax, (name, table) in zip(axes, tables.items()):
        ax.plot(table["bms_soc_label_pct"], table["R_int_dis_ohm"] * 1000, "o-",
                color="#e15759", label="Discharge DCIR")
        ax.plot(table["bms_soc_label_pct"], table["R_int_chg_ohm"] * 1000, "o-",
                color="#59a14f", label="Charge / regen DCIR")
        ax.set_xlabel("SOC (%) -- BMS battery_level label")
        ax.set_ylabel("DCIR (mΩ)")
        ax.set_xlim(0, 100)
        ax.set_title(name)
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("Plot 2: Internal resistance profile -- DCIR vs SOC")
    fig.tight_layout()
    path = os.path.join(out_dir, "02_dcir_vs_soc.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    tables = {name: build_table(cfg) for name, cfg in PACKS.items()}

    for name, table in tables.items():
        soc = table["bms_soc_label_pct"]
        print(f"{name}: {len(table)} OCV_REST checkpoints, "
              f"SOC label range {soc.min():.1f}% - {soc.max():.1f}%")

    p1 = plot_ocv_vs_soc(tables, OUT_DIR)
    p2 = plot_dcir_vs_soc(tables, OUT_DIR)
    print(f"\nSaved -> {p1}")
    print(f"Saved -> {p2}")
    print("\nReminder: the 2nd-life pack's SOC% axis is the BMS's own "
          "circular label (see hppc_2ndlife_analysis.py) -- treat its curve "
          "shape as informative, not as a validated absolute SOC scale.")


if __name__ == "__main__":
    main()
