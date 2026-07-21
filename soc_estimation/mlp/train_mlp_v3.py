"""
SOC Multi-Horizon Delta Prediction — MLP v3
============================================
Reframed from v2: instead of predicting absolute SOC at t+1..t+5 s (which a
persistence baseline solves at ~0.03 pp MAE), the model predicts the *change*
in SOC over multiple horizons from voltage/current-derived signals alone —
current SOC is NOT fed in as a model input. Reconstructing an absolute SOC
at deployment means adding the predicted delta to a SOC estimate you already
track independently (e.g. the BMS's own reading).

Label: soc_cc — OCV-anchored coulomb-counted SOC (soc_estimation/soc_relabel.py),
not the Daly BMS battery_level (quantized to 0.1 pp and biased 10-20 pp on
later runs).

Features (all computable online at deployment):
    voltage, C_rate, dV_dt, delta_voltage, I_mean_10s, I_mean_30s

Targets:
    [soc_cc(t+h) - soc_cc(t) for h in HORIZONS]  (percentage points, signed)

Baselines reported alongside the model:
    zero      — SOC does not change (persistence)
    coulomb   — current draw stays at I(t) for the whole horizon

Run from repo root:
    python -m soc_estimation.mlp.train_mlp_v3
"""

import h5py
import numpy as np
import pandas as pd
import joblib
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from pathlib import Path

from soc_estimation.mlp.mlp import MLP_SOC, ModelManager
from soc_estimation.soc_relabel import add_soc_cc, NOMINAL_CAP_AH

# ── CONFIG ─────────────────────────────────────────────────────────────────────
H5_PATH      = Path("dataset/all_data/h5_files/hoverboard_bms_dataset.h5")
SAVE_DIR     = Path("soc_estimation/mlp/outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS     = [5, 15, 30, 60]   # seconds ahead (rows ahead at ~1 Hz)
HIDDEN_SIZES = [64, 32, 16]
BATCH_SIZE   = 128
EPOCHS       = 300
PATIENCE     = 30
LR           = 3e-4

# FEATURE_COLS = ["voltage", "C_rate", "dV_dt", "delta_voltage",
#                 "I_mean_10s", "I_mean_30s"]

FEATURE_COLS = ["voltage", "C_rate", "dV_dt", "I_mean_30s"]
# ── Run splits (same as v2) ────────────────────────────────────────────────────
TRAIN_RUNS = [
    "file1_run_001",
    "file1_run_002",
    "file1_run_003",
    "file2_run_001_40pct_speed_15kg_load_discharge",
    "file2_run_002_40pct_speed_15kg_load_discharge",
    "file2_run_003_charge",
    "file2_run_009_40pct_speed_25kg_load_discharge",
    "file2_run_010_80pct_speed_25kg_load_discharge",
    "file2_run_011_80pct_speed_25kg_load_discharge",
    "file2_run_012_80pct_speed_25kg_load_discharge",
    "file2_run_013_80pct_speed_25kg_load_discharge",
    "file2_run_014_charge",
    "file2_run_016_charge",
    "file2_run_017_charge",
    "file2_run_018_charge",
]
VAL_RUNS = [
    "file1_run_004",
    "file1_run_005",
    "file2_run_006_60pct_speed_15kg_load_discharge",
    "file2_run_007_60pct_speed_15kg_load_discharge",
    "file2_run_008_charge",
    "file3_run_001_prediction",
    "file3_run_002_prediction",
]
TEST_RUNS = [
    "file3_run_003_speed_profile_1",
    "file2_run_004_80pct_speed_15kg_load_discharge",
    "file2_run_005_charge",
    "file2_run_015_80pct_speed_discharge",
]


# ── Feature engineering ────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt_s = (df["timestamp_ms"].diff().fillna(1000) / 1000.0).clip(lower=0.1)

    df["C_rate"]     = df["current"] / NOMINAL_CAP_AH
    df["dV_dt"]      = (df["voltage"].diff().fillna(0.0) / dt_s).clip(-1.0, 1.0)
    df["I_mean_10s"] = df["current"].rolling(10, min_periods=1).mean()
    df["I_mean_30s"] = df["current"].rolling(30, min_periods=1).mean()
    return df


# ── HDF5 loader ────────────────────────────────────────────────────────────────
def load_run(f: h5py.File, run_name: str) -> pd.DataFrame:
    g = f[run_name]["bms"]
    df = pd.DataFrame({
        "timestamp_ms":  f[run_name]["timestamp_ms"][:],
        "battery_level": g["battery_level"][:],
        "voltage":       g["voltage"][:],
        "current":       g["current"][:],
        "delta_voltage": g["delta_voltage"][:],
    })
    df["run_name"] = run_name
    return df


def load_runs(h5_path: Path, run_names: list[str]) -> dict[str, pd.DataFrame]:
    runs = {}
    with h5py.File(h5_path, "r") as f:
        available = set(f.keys())
        for name in run_names:
            if name not in available:
                print(f"  [warn] run '{name}' not found in HDF5 — skipping")
                continue
            df = load_run(f, name)
            df = add_soc_cc(df)
            # BLE dropout rows (V=0) get NaN soc_cc — drop before differencing
            df = df.dropna(subset=["soc_cc"]).reset_index(drop=True)
            df = engineer_features(df)
            runs[name] = df
    return runs


# ── Dataset construction ───────────────────────────────────────────────────────
def build_arrays(runs: dict[str, pd.DataFrame]):
    """
    X[t]       = features at time t
    Y[t, k]    = soc_cc(t + HORIZONS[k]) - soc_cc(t)   (pp, signed)
    I_now[t]   = current at time t (for the coulomb-extrapolation baseline)
    SOC_now[t] = soc_cc at time t (not necessarily an input feature — kept
                 separately so absolute SOC can be reconstructed for plotting)
    """
    max_h = max(HORIZONS)
    X_list, Y_list, I_list, SOC_list = [], [], [], []
    for name, df in runs.items():
        n = len(df)
        if n <= max_h:
            print(f"  [warn] '{name}' too short ({n} rows) — skipping")
            continue

        soc = df["soc_cc"].values.astype(np.float32)
        X = df[FEATURE_COLS].values[: n - max_h].astype(np.float32)
        Y = np.stack(
            [soc[h: n - max_h + h] - soc[: n - max_h] for h in HORIZONS],
            axis=1,
        ).astype(np.float32)

        X_list.append(X)
        Y_list.append(Y)
        I_list.append(df["current"].values[: n - max_h].astype(np.float32))
        SOC_list.append(soc[: n - max_h])

    return (np.vstack(X_list), np.vstack(Y_list),
            np.concatenate(I_list), np.concatenate(SOC_list))


def print_model_summary(model: torch.nn.Module):
    """Print layer architecture and parameter counts."""
    print(model)
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,}  |  Trainable: {trainable:,}")


class HorizonDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# ── Evaluation ─────────────────────────────────────────────────────────────────
def predict_all(model, loader, device, scaler_Y):
    model.eval()
    preds = []
    with torch.no_grad():
        for X_batch, _ in loader:
            preds.append(model(X_batch.to(device)).cpu().numpy())
    return scaler_Y.inverse_transform(np.vstack(preds))


def predict_array(model, X_sc, device, scaler_Y, batch_size=512):
    """Run inference directly on a pre-scaled feature array (no DataLoader) —
    used for per-run plotting where each run needs its own contiguous
    prediction array rather than the train/val/test-concatenated one."""
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_sc), batch_size):
            batch = torch.from_numpy(X_sc[i: i + batch_size]).to(device)
            preds.append(model(batch).cpu().numpy())
    return scaler_Y.inverse_transform(np.vstack(preds))


def evaluate(preds_pp, Y_pp, I_now, label=""):
    """Print per-horizon metrics for the model and both baselines.

    All values are SOC percentage points (pp) — numerically identical to "%"
    for an SOC difference, e.g. going from 50% to 45% is both "-5 pp" and
    "-5%". Reported here as "%" to match how SOC is normally read off.
    """
    print(f"\n  {label}")
    print(f"  {'Horizon':>8}  {'model MAE':>10} {'RMSE':>8} {'R²':>8}   "
          f"{'zero MAE':>9}   {'coulomb MAE':>11}   (all in % SOC)")
    print(f"  {'-'*66}")
    metrics = []
    for k, h in enumerate(HORIZONS):
        y, p = Y_pp[:, k], preds_pp[:, k]
        mae  = mean_absolute_error(y, p)
        rmse = root_mean_squared_error(y, p)
        r2   = r2_score(y, p)
        mae_zero    = np.abs(y).mean()
        delta_coul  = I_now * h / 3600.0 / NOMINAL_CAP_AH * 100.0
        mae_coulomb = np.abs(y - delta_coul).mean()
        print(f"  t+{h:>3}s   {mae:>10.4f} {rmse:>8.4f} {r2:>8.4f}   "
              f"{mae_zero:>9.4f}   {mae_coulomb:>11.4f}")
        metrics.append({"horizon_s": h, "mae    _pct": mae, "rmse_pct": rmse,
                        "r2": r2, "zero_mae_pct": mae_zero,
                        "coulomb_mae_pct": mae_coulomb})
    return metrics


# ── Plots ──────────────────────────────────────────────────────────────────────
def plot_history(history: dict, save_path: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["train_loss"], label="Train loss")
    ax.plot(history["val_loss"],   label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (scaled)")
    ax.set_title("Training History — delta-SOC Multi-Horizon Predictor")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved -> {save_path}")


def plot_predictions(preds_pp, Y_pp, soc_now_full, n_samples, save_path, run_name=""):
    """Per-horizon actual vs predicted absolute SOC (%), reconstructed as
    SOC(t) + delta. Both curves use the same SOC(t) baseline, so this is
    just a different view of the same delta error — not a new quantity."""
    time = np.arange(n_samples)
    soc_now = soc_now_full[:n_samples]
    n_rows = len(HORIZONS)
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 2.4 * n_rows), sharex=True)
    title = "SOC Prediction vs Actual" + (f" — {run_name}" if run_name else " (Test Set)")
    fig.suptitle(title, fontweight="bold", y=1.0)

    colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(HORIZONS)))
    for k, h in enumerate(HORIZONS):
        ax = axes[k]
        soc_actual = soc_now + Y_pp[:n_samples, k]
        soc_pred   = soc_now + preds_pp[:n_samples, k]
        ax.plot(time, soc_actual, color="0.3", linewidth=1.0, label="Actual")
        ax.plot(time, soc_pred, color=colors[k], linewidth=0.9,
                linestyle="--", label=f"Predicted t+{h}s")
        ax.set_ylabel("SOC (%)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.5)

    axes[-1].set_xlabel("Sample index")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved -> {save_path}")


def plot_delta_predictions(preds_pp, Y_pp, n_samples, save_path, run_name=""):
    """Per-horizon actual vs predicted SOC *change* (pp) — the raw model
    output/target, not reconstructed to absolute SOC. Complements
    plot_predictions(): errors that are hard to see on a 0-100% axis show
    up clearly here since the y-axis is scaled to the deltas themselves."""
    time = np.arange(n_samples)
    n_rows = len(HORIZONS)
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 2.4 * n_rows), sharex=True)
    title = "Delta-SOC Prediction vs Actual" + (f" — {run_name}" if run_name else " (Test Set)")
    fig.suptitle(title, fontweight="bold", y=1.0)

    colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(HORIZONS)))
    for k, h in enumerate(HORIZONS):
        ax = axes[k]
        ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)
        ax.plot(time, Y_pp[:n_samples, k], color="0.3", linewidth=1.0,
                label="Actual ΔSOC")
        ax.plot(time, preds_pp[:n_samples, k], color=colors[k], linewidth=0.9,
                linestyle="--", label=f"Predicted ΔSOC t+{h}s")
        ax.set_ylabel("ΔSOC (pp)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.5)

    axes[-1].set_xlabel("Sample index")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved -> {save_path}")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("Loading runs from HDF5 (labels: OCV-anchored coulomb count)...")
    train_runs = load_runs(H5_PATH, TRAIN_RUNS)
    val_runs   = load_runs(H5_PATH, VAL_RUNS)
    test_runs  = load_runs(H5_PATH, TEST_RUNS)

    print(f"\nRuns loaded — train: {len(train_runs)}  "
          f"val: {len(val_runs)}  test: {len(test_runs)}")

    print("\nBuilding (X, Y) arrays...")
    X_train, Y_train, I_train, SOC_train = build_arrays(train_runs)
    X_val,   Y_val,   I_val,   SOC_val   = build_arrays(val_runs)
    X_test,  Y_test,  I_test,  SOC_test  = build_arrays(test_runs)

    print(f"  Train: X={X_train.shape}  Y={Y_train.shape}")
    print(f"  Val:   X={X_val.shape}    Y={Y_val.shape}")
    print(f"  Test:  X={X_test.shape}   Y={Y_test.shape}")

    # ── Scale features and targets (fit on train only) ────────────────────────
    scaler_X = StandardScaler()
    X_train_sc = scaler_X.fit_transform(X_train).astype(np.float32)
    X_val_sc   = scaler_X.transform(X_val).astype(np.float32)
    X_test_sc  = scaler_X.transform(X_test).astype(np.float32)

    # Delta targets are small (fractions of a pp) — standardize for training
    scaler_Y = StandardScaler()
    Y_train_sc = scaler_Y.fit_transform(Y_train).astype(np.float32)
    Y_val_sc   = scaler_Y.transform(Y_val).astype(np.float32)
    Y_test_sc  = scaler_Y.transform(Y_test).astype(np.float32)

    joblib.dump({"scaler_X": scaler_X, "scaler_Y": scaler_Y},
                SAVE_DIR / "scalers_v3.pkl")
    print(f"\nScalers saved -> {SAVE_DIR / 'scalers_v3.pkl'}")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_loader = DataLoader(HorizonDataset(X_train_sc, Y_train_sc),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(HorizonDataset(X_val_sc,   Y_val_sc),
                              batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(HorizonDataset(X_test_sc,  Y_test_sc),
                              batch_size=BATCH_SIZE, shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    model = MLP_SOC(input_size=len(FEATURE_COLS),
                    hidden_sizes=HIDDEN_SIZES,
                    output_size=len(HORIZONS),
                    output_activation="none").to(device)

    print_model_summary(model)

    manager = ModelManager(
        model,
        device=str(device),
        optimizer=torch.optim.Adam(model.parameters(), lr=LR),
        criterion=torch.nn.MSELoss(),
    )

    print(f"\nModel: {len(FEATURE_COLS)} features -> {HIDDEN_SIZES} "
          f"-> {len(HORIZONS)} delta outputs")
    print(f"Horizons: {HORIZONS} s  |  Batch: {BATCH_SIZE}  |  LR: {LR}\n")

    # ── Train ─────────────────────────────────────────────────────────────────
    model_path = SAVE_DIR / "mlp_soc_v3_delta.pth"
    history = manager.start_training(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        patience=PATIENCE,
        save_path=str(model_path),
        verbose=True,
    )

    # ── Evaluate (metrics in real pp after inverse-scaling) ──────────────────
    val_preds  = predict_all(model, val_loader,  device, scaler_Y)
    test_preds = predict_all(model, test_loader, device, scaler_Y)

    evaluate(val_preds, Y_val, I_val, label="--- Validation set ---")
    test_metrics = evaluate(test_preds, Y_test, I_test, label="--- Test set ---")

    pd.DataFrame(test_metrics).to_csv(SAVE_DIR / "test_metrics_v3.csv",
                                      index=False)

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_history(history, SAVE_DIR / "training_history_v3.png")

    # One plot per test run (not the train/val/test-concatenated array) so
    # each run's full length is shown and run boundaries never get stitched
    # together on a single x-axis.
    for name, df in test_runs.items():
        X_run, Y_run, _, SOC_run = build_arrays({name: df})
        X_run_sc = scaler_X.transform(X_run).astype(np.float32)
        preds_run = predict_array(model, X_run_sc, device, scaler_Y)

        plot_predictions(preds_run, Y_run, SOC_run,
                         n_samples=len(preds_run),
                         save_path=SAVE_DIR / f"test_predictions_v3_{name}.png",
                         run_name=name)
        plot_delta_predictions(preds_run, Y_run,
                               n_samples=len(preds_run),
                               save_path=SAVE_DIR / f"test_deltas_v3_{name}.png",
                               run_name=name)

    print("\nDone.")
