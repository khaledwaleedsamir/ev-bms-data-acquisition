"""
HPPC analysis for the original "new" pack test (hoverboard_bms_HPPC_data.h5,
run_hppc_01..07 -- run_full_charge excluded, same as extract_hppc_csv.py).

Same step-discharge-until-BMS-SOC-drops-10% protocol as the 2nd-life run
(see hppc_2ndlife_analysis.py's docstring), so the same circularity applies
in principle: each step's Ah is bounded by the BMS's *configured* 10.2 Ah
nominal, not an independent measurement. The difference here is that this
pack was fresh/new at test time, so its true capacity is expected to be at
or near that same configured 10.2 Ah -- there's no reason to expect the
BMS's assumption and reality to diverge the way they clearly did for the
2nd-life pack (see that script's finding #2). That expectation, not a new
independent measurement, is why output_figures/hppc/hppc_results.csv's OCV-SOC
curve has been treated as a usable reference elsewhere in this repo
(soc_relabel.py). This script recomputes the same rest-OCV / Ah-removed /
R_int table from raw data as a cross-check against that existing CSV, using
the shared, non-circular Coulomb-counting method from hppc_common.py.
"""

import os

from dataset.data_analysis.hppc_analysis import hppc_common as common

H5_FILE   = os.path.join("dataset", "all_data", "h5_files", "hoverboard_bms_HPPC_data.h5")
RUN_NAMES = [f"run_hppc_{i:02d}" for i in range(1, 8)]
OUT_DIR   = os.path.join("output_figures", "hppc_newpack")
LABEL     = "new pack"


def main():
    df, table = common.run_analysis(
        H5_FILE, RUN_NAMES, OUT_DIR, LABEL,
        csv_name="hppc_newpack_results.csv",
    )
    total_ah = df["cum_ah_removed"].iloc[-1]
    print(f"\nTotal measured Ah removed across the run: {total_ah:.3f} Ah "
          f"(BMS battery_level {df['bms_soc'].iloc[0]:.1f}% -> {df['bms_soc'].iloc[-1]:.1f}%)")


if __name__ == "__main__":
    main()
