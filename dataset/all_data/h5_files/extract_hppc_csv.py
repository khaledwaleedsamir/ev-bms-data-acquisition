"""
Concatenates all HPPC runs from hoverboard_bms_HPPC_data.h5 into a single CSV.

Time axis: continuous sample index (0, 1, 2, ...) at 1 Hz — one second per sample.
Runs are joined in order with no gap (switching happened during rest, so no time lost).
"""

import os
import h5py
import numpy as np
import pandas as pd

H5_FILE   = os.path.join(os.path.dirname(__file__), "hoverboard_bms_HPPC_data.h5")
OUT_DIR   = os.path.join(os.path.dirname(__file__), "..", "csv")
OUT_FILE  = os.path.join(OUT_DIR, "hppc_full.csv")

RUNS = [
    "run_hppc_01",
    "run_hppc_02",
    "run_hppc_03",
    "run_hppc_04",
    "run_hppc_05",
    "run_hppc_06",
    "run_hppc_07",
]


def flatten_run(grp):
    """Return a dict of column_name -> 1-D numpy array for one run group."""
    cols = {}

    def add_dataset(prefix, ds):
        data = ds[()]
        if data.dtype.kind in ("O", "S"):
            data = data.astype(str)
        if data.ndim == 1:
            cols[prefix] = data
        elif data.ndim == 2:
            for i in range(data.shape[1]):
                cols[f"{prefix}_{i}"] = data[:, i]

    for key in grp.keys():
        item = grp[key]
        if isinstance(item, h5py.Dataset):
            # skip the original time columns — replaced by sample index below
            if key in ("timestamp_ms", "time_string"):
                continue
            add_dataset(key, item)
        elif isinstance(item, h5py.Group):
            for sig_key in item.keys():
                sig = item[sig_key]
                if isinstance(sig, h5py.Dataset):
                    add_dataset(f"{key}/{sig_key}", sig)

    return cols


os.makedirs(OUT_DIR, exist_ok=True)

frames = []
sample_offset = 0

with h5py.File(H5_FILE, "r") as f:
    for run_name in RUNS:
        grp = f[run_name]
        n = len(grp["timestamp_ms"])
        print(f"  {run_name}: {n:,} samples  (starts at sample {sample_offset})")

        cols = flatten_run(grp)
        df = pd.DataFrame(cols)
        df.insert(0, "time_s", np.arange(sample_offset, sample_offset + n))
        frames.append(df)
        sample_offset += n

combined = pd.concat(frames, ignore_index=True)
combined.to_csv(OUT_FILE, index=False)

print(f"\nTotal samples : {len(combined):,}")
print(f"Columns       : {len(combined.columns)}")
print(f"Saved -> {os.path.abspath(OUT_FILE)}")
