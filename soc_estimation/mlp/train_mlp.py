"""
Train MLP SOC model directly on the collected BMS dataset (no transfer learning).

Features (all computed from raw sensor data)
--------------------------------------------
  Voltage [V]          pack voltage / N_SERIES   (per-cell)
  Current [A]          pack current / N_PARALLEL  (per-cell)
  Temperature [degC]   mean of 3 BMS temp sensors
  Cycle Charge [Ah]    Coulomb-counted Ah since run start  (resets each run)
  Cycle Capacity [Wh]  Energy-counted Wh since run start   (resets each run)

Split (run-based, no data leakage)
-----------------------------------
  Train : file1 + file2 majority runs
  Val   : held-out file1/file2 runs + speed profile
  Test  : file2_run_013, file2_run_015, file3_run_002  (unseen conditions)

Outputs (saved to soc_estimation/mlp/outputs/)
  mlp_bms.pth
  mlp_bms_scalers.pkl
  mlp_bms_training_curve.png

Run from the repository root:
  python -m soc_estimation.mlp.train_mlp
"""

import os
import numpy as np
import pandas as pd
import h5py
import torch
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

from soc_estimation.mlp.mlp import MLP_SOC, ModelManager

# ──────────────────────────── CONFIG ─────────────────────────────────────────
HDF5_PATH    = r'dataset\all_data\h5_files\hoverboard_bms_dataset.h5'
SAVE_DIR     = r'soc_estimation\mlp\outputs'
MODEL_NAME   = 'mlp_bms'

FEATURE_COLS = ['Voltage [V]', 'Current [A]', 'Temperature [degC]',
                'Cycle Charge [Ah]', 'Cycle Capacity [Wh]']
TARGET_COL   = 'SOC [-]'

N_SERIES   = 10
N_PARALLEL =  3

HIDDEN_SIZES = [128, 64, 32]
BATCH_SIZE   = 256
EPOCHS       = 200
PATIENCE     = 20
LR           = 1e-3

TRAIN_RUNS = [
    'file1_run_001',
    'file1_run_002',
    'file1_run_003',
    'file2_run_001_40pct_speed_15kg_load_discharge',
    'file2_run_002_40pct_speed_15kg_load_discharge',
    'file2_run_003_charge',
    'file2_run_009_40pct_speed_25kg_load_discharge',
    'file2_run_010_80pct_speed_25kg_load_discharge',
    'file2_run_011_80pct_speed_25kg_load_discharge',
    'file2_run_012_80pct_speed_25kg_load_discharge',
    'file2_run_014_charge',
    'file2_run_016_charge',
    'file2_run_017_charge',
    'file2_run_018_charge',
    'file2_run_019_charge',
    'file3_run_001_prediction',
]

VAL_RUNS = [
    'file1_run_004',
    'file1_run_005',
    'file2_run_004_80pct_speed_15kg_load_discharge',
    'file2_run_005_charge',
    'file2_run_006_60pct_speed_15kg_load_discharge',
    'file2_run_007_60pct_speed_15kg_load_discharge',
    'file2_run_008_charge',
    'file3_run_003_speed_profile_1',
]

TEST_RUNS = [
    'file2_run_013_80pct_speed_25kg_load_discharge',
    'file2_run_015_80pct_speed_discharge',
    'file3_run_002_prediction',
]
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(SAVE_DIR, exist_ok=True)


# ── 1. Load HDF5 data ─────────────────────────────────────────────────────────
print("=" * 60)
print("Loading BMS dataset")
print("=" * 60)

run_dfs = []
with h5py.File(HDF5_PATH, 'r') as f:
    for run_name in f.keys():
        g         = f[run_name]['bms']
        ts_ms     = f[run_name]['timestamp_ms'][:].astype(np.float64)
        temp_vals = g['temp_values'][:]
        voltage   = g['voltage'][:].astype(np.float32)
        current   = g['current'][:].astype(np.float32)

        dt_s              = np.diff(ts_ms / 1000.0, prepend=ts_ms[0] / 1000.0).astype(np.float32)
        cycle_charge_ah   = np.cumsum(current * dt_s / 3600.0)
        cycle_capacity_wh = np.cumsum(voltage * current * dt_s / 3600.0)

        run_dfs.append(pd.DataFrame({
            'Voltage [V]':         voltage,
            'Current [A]':         current,
            'Temperature [degC]':  temp_vals.mean(axis=1).astype(np.float32),
            'Cycle Charge [Ah]':   cycle_charge_ah,
            'Cycle Capacity [Wh]': cycle_capacity_wh,
            'SOC [-]':             g['battery_level'][:].astype(np.float32)  / 100.0,
            'run_name':            run_name,
        }))

full_df = pd.concat(run_dfs, ignore_index=True)
print(f"Loaded {len(run_dfs)} runs  ({len(full_df):,} samples total)")


# ── 2. Run-based split ────────────────────────────────────────────────────────
train_df = full_df[full_df['run_name'].isin(TRAIN_RUNS)].copy()
val_df   = full_df[full_df['run_name'].isin(VAL_RUNS)].copy()
test_df  = full_df[full_df['run_name'].isin(TEST_RUNS)].copy()

unassigned = [r for r in full_df['run_name'].unique()
              if r not in set(TRAIN_RUNS) | set(VAL_RUNS) | set(TEST_RUNS)]
if unassigned:
    print(f"[INFO] Excluded runs: {unassigned}")

print(f"\nSplit summary")
print(f"  Train : {len(train_df):>9,} samples  ({len(TRAIN_RUNS)} runs)")
print(f"  Val   : {len(val_df):>9,} samples  ({len(VAL_RUNS)} runs)")
print(f"  Test  : {len(test_df):>9,} samples  ({len(TEST_RUNS)} runs)")


# ── 3. Scale ──────────────────────────────────────────────────────────────────
scaler_X = StandardScaler()
X_train  = scaler_X.fit_transform(train_df[FEATURE_COLS].values.astype(np.float32))
y_train  = train_df[TARGET_COL].values.reshape(-1, 1).astype(np.float32)
X_val    = scaler_X.transform(val_df[FEATURE_COLS].values.astype(np.float32))
y_val    = val_df[TARGET_COL].values.reshape(-1, 1).astype(np.float32)
X_test   = scaler_X.transform(test_df[FEATURE_COLS].values.astype(np.float32))
y_test   = test_df[TARGET_COL].values.reshape(-1, 1).astype(np.float32)

scaler_path = os.path.join(SAVE_DIR, f'{MODEL_NAME}_scalers.pkl')
joblib.dump({'scaler_X': scaler_X, 'scaler_y': None}, scaler_path)
print(f"\nScaler saved → {scaler_path}")
for name, mean, scale in zip(FEATURE_COLS, scaler_X.mean_, scaler_X.scale_):
    print(f"  {name:<25} mean={mean:.4f}  std={scale:.4f}")


# ── 4. DataLoaders ────────────────────────────────────────────────────────────
class _PairDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(_PairDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(_PairDataset(X_val,   y_val),   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(_PairDataset(X_test,  y_test),  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


# ── 5. Model ──────────────────────────────────────────────────────────────────
device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model     = MLP_SOC(input_size=len(FEATURE_COLS), hidden_sizes=HIDDEN_SIZES, output_size=1)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = torch.nn.MSELoss()
manager   = ModelManager(model, device=device, optimizer=optimizer, criterion=criterion)
manager.scaler_X = scaler_X

print(f"\nModel: MLP {HIDDEN_SIZES}  |  params: {sum(p.numel() for p in model.parameters()):,}")


# ── 6. Train ──────────────────────────────────────────────────────────────────
model_path = os.path.join(SAVE_DIR, f'{MODEL_NAME}.pth')
print(f"Training on {device}  →  {model_path}")
print("=" * 60)

history = manager.start_training(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=EPOCHS,
    patience=PATIENCE,
    save_path=model_path,
    verbose=True,
)


# ── 7. Test-set evaluation ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Test-set results")
print("=" * 60)
test_metrics = manager.validate(test_loader)
print(f"  Loss  (MSE) : {test_metrics['loss']:.6f}")
print(f"  MAE         : {test_metrics['mae']*100:.2f} %")
print(f"  RMSE        : {test_metrics['rmse']*100:.2f} %")
print(f"  R²          : {test_metrics['r2']:.4f}")


# ── 8. Training curve ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(history['train_loss'], label='Train Loss')
ax.plot(history['val_loss'],   label='Val Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE Loss')
ax.set_title(f'BMS MLP {HIDDEN_SIZES} — Training History')
ax.legend()
ax.grid(True)
plt.tight_layout()
curve_path = os.path.join(SAVE_DIR, f'{MODEL_NAME}_training_curve.png')
plt.savefig(curve_path, dpi=150)
print(f"\nTraining curve saved → {curve_path}")
