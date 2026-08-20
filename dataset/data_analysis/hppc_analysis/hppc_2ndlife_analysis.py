"""
HPPC analysis for the 2nd-life pack run (dc_load_bms_HPPC_data_2ndlife.h5,
run_hppc_dc_load_01).

No absolute capacity/SOH number is computed here -- two checks ruled that
out before this script was written:

  1. Circular stop condition. Each step discharges until the Daly BMS's
     `battery_level` drops by 10%. Coulomb-counting the raw pack current
     between rest points gives ~1.02 Ah on every single step (1.01-1.02 Ah,
     <2% spread) regardless of how much the OCV actually moved that step
     (0.3-1.2 V, highly non-uniform, as a real Li-ion OCV curve should be).
     A flat Ah/step while OCV/step varies is the signature of `battery_level`
     being coulomb-counted against the BMS's *configured* 10.2 Ah nominal
     capacity, not anything it measures about this pack -- so "8.15 Ah over
     an 80-point battery_level swing" is guaranteed by the test's own control
     loop, not a measurement of this pack.

  2. The one independent check available doesn't transfer. Inverting this
     pack's rest OCVs against the OCV-SOC curve from the original "new" pack
     (output_figures/hppc/hppc_results.csv, itself BMS-independent) implies
     ~16.5 Ah of capacity -- more than nominal, for a supposedly degraded
     pack. Confirmed with the user: this 2nd-life pack is a different
     physical unit from that earlier test, so its curve isn't a valid
     reference here.

What IS independently measurable from raw current + voltage, and what this
script reports instead: a rest-OCV-vs-cumulative-Ah-removed curve and an
internal-resistance-vs-step curve. Getting an actual capacity number needs a
follow-up run that discharges to a real voltage cutoff using coulomb-counted
Ah (not BMS battery_level) as the stop condition.
"""

import os

from dataset.data_analysis.hppc_analysis import hppc_common as common

H5_FILE  = os.path.join("dataset", "all_data", "h5_files", "dc_load_bms_HPPC_data_2ndlife.h5")
RUN_NAMES = ["run_hppc_dc_load_01"]
OUT_DIR  = os.path.join("output_figures", "hppc_2ndlife")
LABEL    = "2nd-life pack"


def main():
    df, table = common.run_analysis(
        H5_FILE, RUN_NAMES, OUT_DIR, LABEL,
        csv_name="hppc_2ndlife_results.csv",
    )
    total_ah = df["cum_ah_removed"].iloc[-1]
    print(f"\nTotal measured Ah removed across the run: {total_ah:.3f} Ah")
    print("This is a real, non-circular measurement (raw current integration), "
          "but it spans only rest OCV 41.70V -> 35.40V, not a full 100%->cutoff "
          "discharge, so it is NOT a pack capacity figure on its own.")


if __name__ == "__main__":
    main()
