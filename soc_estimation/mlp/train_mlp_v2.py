"""
SOC Multi-Step Prediction — MLP v2
====================================
Predicts SOC at t+1 through t+5 seconds ahead (5-output regression).

Features (no temperature):
    voltage, C_rate, cycle_charge, dV_dt, delta_voltage, R_int

Label:
    [SOC(t+1), ..., SOC(t+5)] / 100  — 5 simultaneous outputs, each in [0, 1]

Run from repo root:
    python -m soc_estimation.mlp.train_mlp_v2
"""

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from pathlib import Path

from soc_estimation.mlp.mlp import MLP_SOC, ModelManager

# ── CONFIG ─────────────────────────────────────────────────────────────────────
H5_PATH        = Path("dataset/all_data/h5_files/hoverboard_bms_dataset.h5")
HPPC_PATH      = Path("output_figures/hppc/hppc_results.csv")
SAVE_DIR       = Path("soc_estimation/mlp/outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

HORIZON        = 5          # predict SOC at t+1 … t+HORIZON seconds
NOMINAL_CAP_AH = 10.2       # fresh pack capacity for C-rate
HIDDEN_SIZES   = [64, 32, 16]
BATCH_SIZE     = 64
EPOCHS         = 300
PATIENCE       = 30
LR             = 1e-4

FEATURE_COLS = ["voltage", "C_rate", "cycle_charge", "dV_dt", "delta_voltage", "R_int"]
TARGET_COL   = "battery_level"   # will be divided by 100

# ── Run splits (HDF5 only — ESP32 CSV runs excluded, missing delta_voltage) ───
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
    "file2_run_004_80pct_speed_15kg_load_discharge",
    "file2_run_005_charge",
    "file2_run_006_60pct_speed_15kg_load_discharge",
    "file2_run_007_60pct_speed_15kg_load_discharge",
    "file2_run_008_charge",
    "file3_run_003_speed_profile_1",
]
TEST_RUNS = [
    "file2_run_015_80pct_speed_discharge",
    "file3_run_001_prediction",
    "file3_run_002_prediction",
]


# ── HPPC R_int lookup ──────────────────────────────────────────────────────────
def load_hppc_lookup(path: Path):
    hppc = pd.read_csv(path).sort_values("soc_pct")
    return hppc["soc_pct"].values, hppc["R_int_dis_ohm"].values


SOC_KNOTS, RINT_KNOTS = load_hppc_lookup(HPPC_PATH)


# ── Feature engineering ────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features to a per-run DataFrame. Modifies a copy."""
    df = df.copy()

    dt_s = (df["timestamp_ms"].diff().fillna(1000) / 1000.0).clip(lower=0.1)

    df["C_rate"]       = df["current"] / NOMINAL_CAP_AH
    df["dV_dt"]        = df["voltage"].diff().fillna(0.0) / dt_s
    df["R_int"]        = np.interp(df["battery_level"], SOC_KNOTS, RINT_KNOTS)

    # Clip dV_dt outliers from sampling jitter at run boundaries
    q_lo, q_hi = df["dV_dt"].quantile([0.01, 0.99])
    df["dV_dt"] = df["dV_dt"].clip(q_lo, q_hi)

    return df


# ── HDF5 loader ────────────────────────────────────────────────────────────────
def load_run(f: h5py.File, run_name: str) -> pd.DataFrame:
    g = f[run_name]["bms"]
    df = pd.DataFrame({
        "timestamp_ms":  f[run_name]["timestamp_ms"][:],
        "battery_level": g["battery_level"][:],
        "voltage":       g["voltage"][:],
        "current":       g["current"][:],
        "cycle_charge":  g["cycle_charge"][:],
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
            df = engineer_features(df)
            runs[name] = df
    return runs


# ── Dataset construction ───────────────────────────────────────────────────────
def build_arrays(runs: dict[str, pd.DataFrame]):
    """
    For each run: X[t] = features at time t,
                  Y[t] = [SOC(t+1)/100, ..., SOC(t+HORIZON)/100]
    Drops the last HORIZON rows of each run (no future label available).
    """
    X_list, Y_list = [], []
    for name, df in runs.items():
        n = len(df)
        if n <= HORIZON:
            print(f"  [warn] '{name}' too short ({n} rows) — skipping")
            continue

        X = df[FEATURE_COLS].values[:n - HORIZON].astype(np.float32)

        soc_norm = (df[TARGET_COL].values / 100.0).astype(np.float32)
        Y = np.stack(
            [soc_norm[h + 1: n - HORIZON + h + 1] for h in range(HORIZON)],
            axis=1,
        )  # shape: (n-HORIZON, HORIZON)

        X_list.append(X)
        Y_list.append(Y)

    return np.vstack(X_list), np.vstack(Y_list)


class HorizonDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# ── Per-horizon evaluation ─────────────────────────────────────────────────────
def evaluate_per_horizon(model, loader, device):
    # Data in loader is already scaled — no additional transform needed.
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for X_batch, Y_batch in loader:
            preds = model(X_batch.to(device)).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(Y_batch.numpy())

    preds   = np.vstack(all_preds)    # (N, HORIZON)
    targets = np.vstack(all_targets)  # (N, HORIZON)

    print(f"\n  {'Step':>6}  {'MAE (pp)':>10}  {'RMSE (pp)':>10}  {'R²':>8}")
    print(f"  {'-'*40}")
    metrics = []
    for h in range(HORIZON):
        mae  = mean_absolute_error(targets[:, h], preds[:, h]) * 100
        rmse = root_mean_squared_error(targets[:, h], preds[:, h]) * 100
        r2   = r2_score(targets[:, h], preds[:, h])
        print(f"  t+{h+1:>2}s   {mae:>10.3f}  {rmse:>10.3f}  {r2:>8.4f}")
        metrics.append({"step": h + 1, "mae_pp": mae, "rmse_pp": rmse, "r2": r2})

    overall_mae  = mean_absolute_error(targets.flatten(), preds.flatten()) * 100
    overall_rmse = root_mean_squared_error(targets.flatten(), preds.flatten()) * 100
    print(f"  {'-'*40}")
    print(f"  {'Overall':>6}  {overall_mae:>10.3f}  {overall_rmse:>10.3f}")

    return preds, targets, metrics


# ── Training history plot ──────────────────────────────────────────────────────
def plot_history(history: dict, save_path: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["train_loss"], label="Train loss")
    ax.plot(history["val_loss"],   label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training History — SOC 5-Step Predictor")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved -> {save_path}")
    plt.show()


def plot_predictions(preds: np.ndarray, targets: np.ndarray,
                     n_samples: int, save_path: Path):
    """Plot first n_samples actual vs predicted SOC for each horizon step."""
    time = np.arange(n_samples)
    fig, axes = plt.subplots(HORIZON, 1, figsize=(12, 2.5 * HORIZON), sharex=True)
    fig.suptitle("SOC Multi-Step Prediction vs Actual (Test Set)",
                 fontweight="bold", y=1.01)

    colors = plt.cm.plasma(np.linspace(0.15, 0.85, HORIZON))
    for h in range(HORIZON):
        ax = axes[h]
        ax.plot(time, targets[:n_samples, h] * 100,
                color="0.3", linewidth=1.2, label="Actual")
        ax.plot(time, preds[:n_samples, h] * 100,
                color=colors[h], linewidth=1.0, linestyle="--",
                label=f"Predicted t+{h+1}s")
        ax.set_ylabel("SOC (%)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.5)

    axes[-1].set_xlabel("Sample index")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved -> {save_path}")
    plt.show()


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("Loading runs from HDF5...")
    train_runs = load_runs(H5_PATH, TRAIN_RUNS)
    val_runs   = load_runs(H5_PATH, VAL_RUNS)
    test_runs  = load_runs(H5_PATH, TEST_RUNS)

    print(f"\nRuns loaded — train: {len(train_runs)}  "
          f"val: {len(val_runs)}  test: {len(test_runs)}")

    print("\nBuilding (X, Y) arrays...")
    X_train, Y_train = build_arrays(train_runs)
    X_val,   Y_val   = build_arrays(val_runs)
    X_test,  Y_test  = build_arrays(test_runs)

    print(f"  Train: X={X_train.shape}  Y={Y_train.shape}")
    print(f"  Val:   X={X_val.shape}    Y={Y_val.shape}")
    print(f"  Test:  X={X_test.shape}   Y={Y_test.shape}")

    # ── Scale features (fit on train only) ────────────────────────────────────
    scaler_X = StandardScaler()
    X_train_sc = scaler_X.fit_transform(X_train).astype(np.float32)
    X_val_sc   = scaler_X.transform(X_val).astype(np.float32)
    X_test_sc  = scaler_X.transform(X_test).astype(np.float32)

    joblib.dump({"scaler_X": scaler_X},
                SAVE_DIR / "scalers_v2.pkl")
    print(f"\nScaler saved -> {SAVE_DIR / 'scalers_v2.pkl'}")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_loader = DataLoader(HorizonDataset(X_train_sc, Y_train),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(HorizonDataset(X_val_sc,   Y_val),
                              batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(HorizonDataset(X_test_sc,  Y_test),
                              batch_size=BATCH_SIZE, shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    model   = MLP_SOC(input_size=len(FEATURE_COLS),
                      hidden_sizes=HIDDEN_SIZES,
                      output_size=HORIZON).to(device)

    manager = ModelManager(
        model,
        device=str(device),
        optimizer=torch.optim.Adam(model.parameters(), lr=LR),
        criterion=torch.nn.MSELoss(),
    )

    print(f"\nModel: {len(FEATURE_COLS)} features -> {HIDDEN_SIZES} -> {HORIZON} outputs")
    print(f"Horizon: {HORIZON} s  |  Batch: {BATCH_SIZE}  |  LR: {LR}\n")

    # ── Train ──────────────────────────────────────────────────────────────────
    model_path = SAVE_DIR / f"mlp_soc_v2_h{HORIZON}.pth"
    history = manager.start_training(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        patience=PATIENCE,
        save_path=str(model_path),
        verbose=True,
    )

    # ── Evaluate ───────────────────────────────────────────────────────────────
    print("\n--- Validation set ---")
    val_preds, val_targets, _ = evaluate_per_horizon(model, val_loader, device)

    print("\n--- Test set ---")
    test_preds, test_targets, test_metrics = evaluate_per_horizon(
        model, test_loader, device)

    pd.DataFrame(test_metrics).to_csv(
        SAVE_DIR / "test_metrics_v2.csv", index=False)

    # ── Plots ──────────────────────────────────────────────────────────────────
    plot_history(history, SAVE_DIR / "training_history_v2.png")
    plot_predictions(test_preds, test_targets,
                     n_samples=min(500, len(test_preds)),
                     save_path=SAVE_DIR / "test_predictions_v2.png")

    print("\nDone.")
