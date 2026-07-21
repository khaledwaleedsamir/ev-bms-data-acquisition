"""
Fine-tune the Panasonic-pretrained MLP on the collected BMS dataset.

Transfer learning strategy
--------------------------
  All layers unfrozen — warm-start from Panasonic weights, fine-tune everything.
  Constant-offset failure mode from the frozen-layer approach indicated the model
  needed full freedom to recalibrate the output level, not just the deeper layers.

The scaler is refit on BMS training data (not reused from Panasonic). The Panasonic
scaler was trained on -20 to 25°C; BMS data sits at ~31°C mean with only 3.6°C std,
making temperature a constant outlier under the old scaler and useless as a feature.
Pack values are converted to per-cell before scaling:
  cell_voltage = pack_voltage / N_SERIES
  cell_current = pack_current / N_PARALLEL

Split (run-based, no data leakage across runs)
-----------------------------------------------
  Train : discharge + charge runs from file1 and file2 (majority)
  Val   : held-out runs from file1/file2 + speed profile run
  Test  : file2_run_015, file3_run_001, file3_run_002  (unseen conditions)

Outputs (saved to soc_estimation/mlp/outputs/)
  mlp_panasonic_finetuned.pth            best fine-tuned weights
  mlp_panasonic_finetuned_training_curve.png
  mlp_panasonic_finetuned_test_results.png
  mlp_panasonic_finetuned_test_metrics.csv

Run from the repository root:
  python -m soc_estimation.mlp.finetune_mlp_panasonic
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import Dataset, DataLoader

from soc_estimation.mlp.mlp import MLP_SOC, ModelManager

# ──────────────────────────── CONFIG ─────────────────────────────────────────
HDF5_PATH      = r'dataset\all_data\h5_files\hoverboard_bms_dataset.h5'
SAVE_DIR       = r'soc_estimation\mlp\outputs'
PRETRAINED_WEIGHTS = os.path.join(SAVE_DIR, 'mlp_panasonic.pth')
MODEL_NAME     = 'mlp_panasonic_finetuned'

FEATURE_COLS   = ['Voltage [V]', 'Current [A]', 'Temperature [degC]',
                  'Cycle Charge [Ah]', 'Cycle Capacity [Wh]']
TARGET_COL     = 'SOC [-]'

# Pack topology — convert to per-cell to match Panasonic training range
N_SERIES   = 10
N_PARALLEL =  3

HIDDEN_SIZES = [128, 64, 32, 16]
BATCH_SIZE   = 256
EPOCHS       = 200
PATIENCE     = 20
LR           = 1e-4   # small LR preserves Panasonic dynamics while correcting the offset

# Run-based split — test runs are held out entirely and never seen during training
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
    'file3_run_003_speed_profile_1'
]

TEST_RUNS = [
    'file2_run_013_80pct_speed_25kg_load_discharge',
    'file2_run_015_80pct_speed_discharge',
    'file3_run_002_prediction'
]
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(SAVE_DIR, exist_ok=True)


# ── 1. Load HDF5 data ─────────────────────────────────────────────────────────
print("=" * 60)
print("Loading BMS dataset from HDF5")
print("=" * 60)

run_dfs = []
with h5py.File(HDF5_PATH, 'r') as f:
    for run_name in f.keys():
        g         = f[run_name]['bms']
        ts_ms     = f[run_name]['timestamp_ms'][:].astype(np.float64)
        temp_vals = g['temp_values'][:]
        voltage   = g['voltage'][:].astype(np.float32)
        current   = g['current'][:].astype(np.float32)

        # Coulomb counting on per-cell values to match Panasonic single-cell ranges
        dt_s               = np.diff(ts_ms / 1000.0, prepend=ts_ms[0] / 1000.0).astype(np.float32)
        cell_i             = current / N_PARALLEL
        cell_v             = voltage / N_SERIES
        cycle_charge_ah    = np.cumsum(cell_i * dt_s / 3600.0)
        cycle_capacity_wh  = np.cumsum(cell_v * cell_i * dt_s / 3600.0)

        df = pd.DataFrame({
            'Voltage [V]':          voltage                                    / N_SERIES,
            'Current [A]':          current                                    / N_PARALLEL,
            'Temperature [degC]':   temp_vals.mean(axis=1).astype(np.float32),
            'Cycle Charge [Ah]':    cycle_charge_ah,
            'Cycle Capacity [Wh]':  cycle_capacity_wh,
            'SOC [-]':              g['battery_level'][:].astype(np.float32)  / 100.0,
            'run_name':             run_name,
        })
        run_dfs.append(df)

full_df = pd.concat(run_dfs, ignore_index=True)
print(f"Loaded {len(run_dfs)} runs  ({len(full_df):,} samples total)")


# ── 2. Run-based split ────────────────────────────────────────────────────────
train_df = full_df[full_df['run_name'].isin(TRAIN_RUNS)].copy()
val_df   = full_df[full_df['run_name'].isin(VAL_RUNS)].copy()
test_df  = full_df[full_df['run_name'].isin(TEST_RUNS)].copy()

all_assigned = set(TRAIN_RUNS) | set(VAL_RUNS) | set(TEST_RUNS)
unassigned   = [r for r in full_df['run_name'].unique() if r not in all_assigned]
if unassigned:
    print(f"\n[INFO] Runs not assigned to any split (excluded): {unassigned}")

print(f"\nSplit summary")
print(f"  Train : {len(train_df):>9,} samples  ({len(TRAIN_RUNS)} runs)")
print(f"  Val   : {len(val_df):>9,} samples  ({len(VAL_RUNS)} runs)")
print(f"  Test  : {len(test_df):>9,} samples  ({len(TEST_RUNS)} runs)")


# ── 3. Fit scaler on BMS training data ───────────────────────────────────────
from sklearn.preprocessing import StandardScaler

scaler_X = StandardScaler()
X_train  = scaler_X.fit_transform(train_df[FEATURE_COLS].values.astype(np.float32))
y_train  = train_df[TARGET_COL].values.reshape(-1, 1).astype(np.float32)

X_val    = scaler_X.transform(val_df[FEATURE_COLS].values.astype(np.float32))
y_val    = val_df[TARGET_COL].values.reshape(-1, 1).astype(np.float32)

X_test   = scaler_X.transform(test_df[FEATURE_COLS].values.astype(np.float32))
y_test   = test_df[TARGET_COL].values.reshape(-1, 1).astype(np.float32)

scaler_path = os.path.join(SAVE_DIR, f'{MODEL_NAME}_scalers.pkl')
joblib.dump({'scaler_X': scaler_X, 'scaler_y': None}, scaler_path)
print(f"\nBMS scaler fit on training data and saved → {scaler_path}")
print(f"  Voltage mean/std    : {scaler_X.mean_[0]:.3f} V / {scaler_X.scale_[0]:.3f}")
print(f"  Current mean/std    : {scaler_X.mean_[1]:.3f} A / {scaler_X.scale_[1]:.3f}")
print(f"  Temperature mean/std: {scaler_X.mean_[2]:.3f} °C / {scaler_X.scale_[2]:.3f}")


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


# ── 5. Load pretrained model and apply transfer learning setup ────────────────
print(f"\nLoading pretrained weights from {PRETRAINED_WEIGHTS}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = MLP_SOC(input_size=len(FEATURE_COLS), hidden_sizes=HIDDEN_SIZES, output_size=1)  # input_size=5
model.load_state_dict(torch.load(PRETRAINED_WEIGHTS, map_location='cpu'))

# Freeze first block: Linear(5→64) + ReLU + Dropout
for layer in model.network[:3]:
    for param in layer.parameters():
        param.requires_grad = False

frozen_params    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nParameter summary")
print(f"  Frozen     : {frozen_params:,}  (first block — Linear(5→64))")
print(f"  Trainable  : {trainable_params:,}")

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=LR
)
criterion = torch.nn.MSELoss()
manager   = ModelManager(model, device=device, optimizer=optimizer, criterion=criterion)
manager.scaler_X = scaler_X   # attach for predict() calls later


# ── 6. Fine-tune ──────────────────────────────────────────────────────────────
model_path = os.path.join(SAVE_DIR, f'{MODEL_NAME}.pth')
print(f"\nFine-tuning on {device}  →  {model_path}")
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


# ── 8. Per-run test plots ─────────────────────────────────────────────────────
test_run_dfs = [test_df[test_df['run_name'] == r] for r in TEST_RUNS if r in test_df['run_name'].values]
per_run_metrics = []

fig, axes = plt.subplots(len(test_run_dfs), 1, figsize=(12, 3.5 * len(test_run_dfs)), squeeze=False)
for ax, run_df in zip(axes[:, 0], test_run_dfs):
    run_name = run_df['run_name'].iloc[0]
    X_run    = scaler_X.transform(run_df[FEATURE_COLS].values.astype(np.float32))
    y_true   = run_df[TARGET_COL].values

    # X_run is already scaled; detach scaler so predict() doesn't double-scale
    manager.scaler_X = None
    y_pred = manager.predict(X_run)
    manager.scaler_X = scaler_X

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2   = r2_score(y_true, y_pred)
    per_run_metrics.append({'run': run_name, 'MAE_%': mae*100, 'RMSE_%': rmse*100, 'R2': r2})

    ax.plot(y_true * 100, label='True SOC (BMS)',  color='steelblue',  lw=1.2)
    ax.plot(y_pred * 100, label='Predicted SOC',    color='orangered', lw=1.2, linestyle='--')
    ax.set_title(f'{run_name}   MAE={mae*100:.1f}%  RMSE={rmse*100:.1f}%  R²={r2:.3f}', fontsize=9)
    ax.set_ylabel('SOC (%)')
    ax.legend(fontsize=8)
    ax.grid(True, lw=0.4)

axes[-1, 0].set_xlabel('Sample index')
plt.suptitle('Fine-tuned Panasonic MLP (warm-start, all layers) → BMS test runs', fontsize=11, y=1.01)
plt.tight_layout()
results_plot = os.path.join(SAVE_DIR, f'{MODEL_NAME}_test_results.png')
plt.savefig(results_plot, dpi=150, bbox_inches='tight')
print(f"\nTest plot saved → {results_plot}")

metrics_df = pd.DataFrame(per_run_metrics)
print("\n" + "=" * 60)
print("Per-run metrics")
print("=" * 60)
print(metrics_df.to_string(index=False, float_format=lambda x: f'{x:.2f}'))
csv_path = os.path.join(SAVE_DIR, f'{MODEL_NAME}_test_metrics.csv')
metrics_df.to_csv(csv_path, index=False)
print(f"Metrics CSV saved → {csv_path}")


# ── 9. Training curve ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(history['train_loss'], label='Train Loss')
ax.plot(history['val_loss'],   label='Val Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE Loss')
ax.set_title('Fine-tuning History — warm-start (Panasonic → BMS, all layers)')
ax.legend()
ax.grid(True)
plt.tight_layout()
curve_path = os.path.join(SAVE_DIR, f'{MODEL_NAME}_training_curve.png')
plt.savefig(curve_path, dpi=150)
print(f"Training curve saved → {curve_path}")
