"""
Inspect what StandardScaler does to train_mlp_v3's features on one sample run.

StandardScaler performs z-score standardization, NOT min-max/normalization:
    x_scaled = (x - mean) / std
where mean/std are computed per-feature from the training set only (fit_transform
on X_train, transform on val/test — see train_mlp_v3.py). Output is centered at 0
with unit variance per column; it is NOT bounded to [0, 1].

Writes a CSV with one row per timestep and, for each feature, a "<feat>_raw" and
"<feat>_scaled" column pair side by side, plus the fitted mean_/scale_ as a
trailing summary block for reference.

Run from repo root:
    python -m soc_estimation.mlp.inspect_scaler [--run RUN_NAME] [--rows N]
"""

import argparse
import joblib
import pandas as pd

from soc_estimation.mlp.train_mlp_v3 import (
    H5_PATH, SAVE_DIR, FEATURE_COLS, load_run, engineer_features,
)
from soc_estimation.soc_relabel import add_soc_cc
import h5py

DEFAULT_RUN = "file2_run_015_80pct_speed_discharge"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=DEFAULT_RUN, help="run_name in the HDF5 file")
    parser.add_argument("--rows", type=int, default=200, help="rows to write (0 = all)")
    args = parser.parse_args()

    scalers = joblib.load(SAVE_DIR / "scalers_v3.pkl")
    scaler_X = scalers["scaler_X"]

    with h5py.File(H5_PATH, "r") as f:
        if args.run not in f.keys():
            raise SystemExit(f"run '{args.run}' not found. Available: {list(f.keys())}")
        df = load_run(f, args.run)

    df = add_soc_cc(df)
    df = df.dropna(subset=["soc_cc"]).reset_index(drop=True)
    df = engineer_features(df)

    raw = df[FEATURE_COLS].reset_index(drop=True)
    if args.rows:
        raw = raw.iloc[: args.rows]
    scaled = pd.DataFrame(scaler_X.transform(raw.values), columns=FEATURE_COLS)

    out = pd.DataFrame(index=raw.index)
    for col in FEATURE_COLS:
        out[f"{col}_raw"] = raw[col].values
        out[f"{col}_scaled"] = scaled[col].values

    out_path = SAVE_DIR / f"scaler_before_after_{args.run}.csv"
    out.to_csv(out_path, index=False)

    summary = pd.DataFrame({
        "feature": FEATURE_COLS,
        "fitted_mean_": scaler_X.mean_,
        "fitted_std_": scaler_X.scale_,
    })
    summary_path = SAVE_DIR / "scaler_v3_mean_std.csv"
    summary.to_csv(summary_path, index=False)

    print(f"Run: {args.run}  |  rows written: {len(out)}")
    print(f"Before/after CSV -> {out_path}")
    print(f"Fitted mean_/scale_ per feature -> {summary_path}\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
