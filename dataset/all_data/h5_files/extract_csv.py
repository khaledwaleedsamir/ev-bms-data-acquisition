import os
import h5py
import numpy as np
import pandas as pd

INPUT_FILE = r"C:\Users\assas\Desktop\NU\Experimental Setup\ev-bms-data-acquisition\dataset\all_data\h5_files\hoverboard_bms_dataset2.h5"  # <-- change this to your file path
OUTPUT_DIR = r"C:\Users\assas\Desktop\NU\Experimental Setup\ev-bms-data-acquisition\dataset\all_data\csv"

RUNS = [
    "run_016_charge",
    "run_017_charge",
    "run_018_charge",
    "run_019_charge",
]


def flatten_run(hdf_file, run_name):
    run_grp = hdf_file[run_name]
    columns = {}

    # Top-level datasets (timestamp_ms, time_string)
    for key in run_grp.keys():
        item = run_grp[key]
        if not isinstance(item, h5py.Dataset):
            continue
        data = item[()]
        if data.dtype.kind in ("O", "S"):
            data = data.astype(str)
        if data.ndim == 1:
            columns[key] = data
        elif data.ndim == 2:
            for i in range(data.shape[1]):
                columns[f"{key}_{i}"] = data[:, i]

    # Sub-groups (bms, hoverboard)
    for grp_key in run_grp.keys():
        item = run_grp[grp_key]
        if not isinstance(item, h5py.Group):
            continue
        for sig_key in item.keys():
            sig = item[sig_key]
            if not isinstance(sig, h5py.Dataset):
                continue
            data = sig[()]
            if data.dtype.kind in ("O", "S"):
                data = data.astype(str)
            col_prefix = f"{grp_key}/{sig_key}"
            if data.ndim == 1:
                columns[col_prefix] = data
            elif data.ndim == 2:
                for i in range(data.shape[1]):
                    columns[f"{col_prefix}_{i}"] = data[:, i]

    df = pd.DataFrame(columns)

    # Put timestamp columns first
    priority = ["timestamp_ms", "time_string"]
    ordered = [c for c in priority if c in df.columns]
    rest = [c for c in df.columns if c not in priority]
    return df[ordered + rest]


os.makedirs(OUTPUT_DIR, exist_ok=True)

with h5py.File(INPUT_FILE, "r") as f:
    for run_name in RUNS:
        print(f"Processing {run_name} ...", end=" ", flush=True)
        df = flatten_run(f, run_name)
        out_path = os.path.join(OUTPUT_DIR, f"{run_name}.csv")
        df.to_csv(out_path, index=False)
        print(f"saved -> {out_path}  ({len(df):,} rows x {len(df.columns)} cols)")

print("Done.")