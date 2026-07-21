"""
Adapter-based transfer learning: Panasonic MLP → BMS pack dataset.

Architecture
------------
  [Raw BMS pack features]
        │
  ┌─────▼──────────────────┐
  │  Adapter  (trainable)  │   x + [Linear(5→32) → ReLU → Dropout(0.1) → Linear(32→16) → ReLU → Dropout(0.1) → Linear(16→5)]
  └─────┬──────────────────┘    residual: adapter learns the correction on top of the identity
        │  5 values in Panasonic-expected feature space
  ┌─────▼──────────────────┐
  │  Panasonic MLP (frozen) │   128→64→32→16→1 + Sigmoid
  └─────┬──────────────────┘
        │
      SOC ∈ [0, 1]

No manual pack-to-cell scaling.  The adapter learns the full domain shift
(pack level → single-cell range + distribution) entirely from data.

The Panasonic network is always kept in eval() mode so its Dropout layers
never activate — even during adapter training.

Outputs (saved to soc_estimation/mlp/outputs/)
  mlp_adapter.pth                      best adapter weights only
  mlp_adapter_bms_scalers.pkl          BMS StandardScaler (fit on raw pack features)
  mlp_adapter_training_curve.png
  mlp_adapter_test_results.png
  mlp_adapter_test_metrics.csv

Run from the repository root:
  python -m soc_estimation.mlp.adapter_mlp_panasonic
"""

import os
import copy
import time
import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn as nn
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from soc_estimation.mlp.mlp import MLP_SOC

# ──────────────────────────── CONFIG ─────────────────────────────────────────
HDF5_PATH          = r'dataset\all_data\h5_files\hoverboard_bms_dataset.h5'
SAVE_DIR           = r'soc_estimation\mlp\outputs'
PRETRAINED_WEIGHTS = os.path.join(SAVE_DIR, 'mlp_panasonic.pth')
MODEL_NAME         = 'mlp_adapter'

FEATURE_COLS = ['Voltage [V]', 'Current [A]', 'Temperature [degC]',
                'Cycle Charge [Ah]', 'Cycle Capacity [Wh]']
TARGET_COL   = 'SOC [-]'

# Adapter architecture: two hidden layers for more capacity
ADAPTER_HIDDEN = [32, 16]

# Pack capacity for Coulomb-counting SOC ground truth
# NCR18650B nominal 3350 mAh × 3P = 10.05 Ah
Q_PACK_AH = 3.35 * 3  # 10.05 Ah

# Panasonic backbone architecture (must match pretrained weights exactly)
PANASONIC_HIDDEN = [128, 64, 32, 16]

BATCH_SIZE = 256
EPOCHS     = 200
PATIENCE   = 20
LR         = 1e-3

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
    'file3_run_003_speed_profile_1',
]

TEST_RUNS = [
    'file2_run_013_80pct_speed_25kg_load_discharge',
    'file2_run_015_80pct_speed_discharge',
    'file3_run_002_prediction',
]
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(SAVE_DIR, exist_ok=True)


# ── Combined model ────────────────────────────────────────────────────────────

class _Adapter(nn.Module):
    def __init__(self, n_features: int, hidden_sizes: list):
        super().__init__()
        layers = []
        prev = n_features
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.1)]
            prev = h
        layers.append(nn.Linear(prev, n_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.net(x)  # residual: adapter learns the correction, not the full mapping


class AdapterSOCNet(nn.Module):
    """Trainable adapter prepended to a fully frozen Panasonic backbone."""

    def __init__(self, adapter: nn.Module, backbone: nn.Module):
        super().__init__()
        self.adapter     = adapter
        self.backbone    = backbone
        self.output_bias = nn.Parameter(torch.zeros(1))  # single scalar offset correction
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x):
        soc = self.backbone(self.adapter(x))
        return torch.clamp(soc + self.output_bias, 0.0, 1.0)

    def train(self, mode: bool = True):
        # Keep backbone always in eval so its Dropout never activates
        super().train(mode)
        self.backbone.eval()
        return self


# ── Dataset helper ────────────────────────────────────────────────────────────

class _PairDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── 1. Load HDF5 — raw pack-level features, no manual scaling ─────────────────
print("=" * 60)
print("Loading BMS dataset (raw pack-level features)")
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

        # Coulomb-counting SOC: anchor initial value from BMS first sample,
        # then integrate pack current forward.
        # Sign convention: positive current = discharge (Daly BMS default).
        soc_init = float(g['battery_level'][0]) / 100.0
        delta_soc = np.cumsum(current * dt_s / 3600.0) / Q_PACK_AH
        soc_cc    = np.clip(soc_init - delta_soc, 0.0, 1.0).astype(np.float32)

        df = pd.DataFrame({
            'Voltage [V]':         voltage,
            'Current [A]':         current,
            'Temperature [degC]':  temp_vals.mean(axis=1).astype(np.float32),
            'Cycle Charge [Ah]':   cycle_charge_ah,
            'Cycle Capacity [Wh]': cycle_capacity_wh,
            'SOC [-]':             soc_cc,
            'run_name':            run_name,
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


# ── 3. Fit BMS scaler on raw pack training features ───────────────────────────
scaler_X = StandardScaler()
X_train  = scaler_X.fit_transform(train_df[FEATURE_COLS].values.astype(np.float32))
y_train  = train_df[TARGET_COL].values.reshape(-1, 1).astype(np.float32)

X_val    = scaler_X.transform(val_df[FEATURE_COLS].values.astype(np.float32))
y_val    = val_df[TARGET_COL].values.reshape(-1, 1).astype(np.float32)

X_test   = scaler_X.transform(test_df[FEATURE_COLS].values.astype(np.float32))
y_test   = test_df[TARGET_COL].values.reshape(-1, 1).astype(np.float32)

scaler_path = os.path.join(SAVE_DIR, f'{MODEL_NAME}_bms_scalers.pkl')
joblib.dump({'scaler_X': scaler_X, 'scaler_y': None}, scaler_path)
print(f"\nBMS scaler (pack-level) saved → {scaler_path}")
print(f"  Voltage  mean/std : {scaler_X.mean_[0]:.2f} V  / {scaler_X.scale_[0]:.3f}")
print(f"  Current  mean/std : {scaler_X.mean_[1]:.2f} A  / {scaler_X.scale_[1]:.3f}")
print(f"  Temp     mean/std : {scaler_X.mean_[2]:.2f} °C / {scaler_X.scale_[2]:.3f}")
print(f"  Cyc.Chg  mean/std : {scaler_X.mean_[3]:.3f} Ah / {scaler_X.scale_[3]:.3f}")
print(f"  Cyc.Cap  mean/std : {scaler_X.mean_[4]:.3f} Wh / {scaler_X.scale_[4]:.3f}")


# ── 4. DataLoaders ────────────────────────────────────────────────────────────
train_loader = DataLoader(_PairDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(_PairDataset(X_val,   y_val),   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(_PairDataset(X_test,  y_test),  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


# ── 5. Build combined model ───────────────────────────────────────────────────
print(f"\nLoading Panasonic backbone from {PRETRAINED_WEIGHTS}")
device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

backbone = MLP_SOC(input_size=len(FEATURE_COLS), hidden_sizes=PANASONIC_HIDDEN, output_size=1)
backbone.load_state_dict(torch.load(PRETRAINED_WEIGHTS, map_location='cpu'))

adapter  = _Adapter(n_features=len(FEATURE_COLS), hidden_sizes=ADAPTER_HIDDEN)
model    = AdapterSOCNet(adapter, backbone).to(device)

frozen_params    = sum(p.numel() for p in model.backbone.parameters())
trainable_params = sum(p.numel() for p in model.adapter.parameters()) + model.output_bias.numel()
print(f"\nParameter summary")
print(f"  Backbone (frozen)   : {frozen_params:,}")
print(f"  Adapter  (trainable): {trainable_params:,}  (adapter + 1 output bias)")

optimizer = torch.optim.Adam(
    list(model.adapter.parameters()) + [model.output_bias], lr=LR
)
criterion = nn.MSELoss()


# ── 6. Training loop ──────────────────────────────────────────────────────────
def _run_epoch(model, loader, optimizer, criterion, device, training: bool):
    model.train(training)
    total_loss, all_preds, all_targets = 0.0, [], []

    with torch.set_grad_enabled(training):
        for X_batch, y_batch in tqdm(loader, desc="Train" if training else "Val", leave=False):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            preds   = model(X_batch)
            loss    = criterion(preds, y_batch)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss   += loss.item() * X_batch.size(0)
            all_preds.append(preds.detach().cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())

    all_preds   = np.concatenate(all_preds).flatten()
    all_targets = np.concatenate(all_targets).flatten()
    avg_loss    = total_loss / len(loader.dataset)
    return {
        'loss': avg_loss,
        'mae':  mean_absolute_error(all_targets, all_preds),
        'rmse': mean_squared_error(all_targets, all_preds) ** 0.5,
        'r2':   r2_score(all_targets, all_preds),
    }


model_path  = os.path.join(SAVE_DIR, f'{MODEL_NAME}.pth')
best_val_loss   = float('inf')
best_adapter_wts = copy.deepcopy(model.adapter.state_dict())
patience_counter = 0
history = {'train_loss': [], 'val_loss': []}
start_time = time.time()

print(f"\nTraining adapter on {device}  →  {model_path}")
print("=" * 60)

for epoch in range(1, EPOCHS + 1):
    train_m = _run_epoch(model, train_loader, optimizer, criterion, device, training=True)
    val_m   = _run_epoch(model, val_loader,   optimizer, criterion, device, training=False)

    history['train_loss'].append(train_m['loss'])
    history['val_loss'].append(val_m['loss'])

    if val_m['loss'] < best_val_loss:
        best_val_loss    = val_m['loss']
        best_adapter_wts = copy.deepcopy(model.adapter.state_dict())
        torch.save({
            'adapter': best_adapter_wts,
            'output_bias': model.output_bias.data.clone(),
        }, model_path)
        patience_counter = 0
    else:
        patience_counter += 1

    print(
        f"Epoch [{epoch}/{EPOCHS}] | "
        f"Train Loss: {train_m['loss']:.6f} | "
        f"Val Loss: {val_m['loss']:.6f} | "
        f"MAE: {val_m['mae']:.4f} | "
        f"RMSE: {val_m['rmse']:.4f} | "
        f"R2: {val_m['r2']:.4f}"
    )

    if patience_counter >= PATIENCE:
        print(f"\nEarly stopping after {epoch} epochs.")
        break

model.adapter.load_state_dict(best_adapter_wts)
with torch.no_grad():
    model.output_bias.copy_(torch.load(model_path, map_location=device)['output_bias'])
print(f"\nTraining completed in {time.time() - start_time:.1f}s  |  Best val loss: {best_val_loss:.6f}")
print(f"Learned output bias: {model.output_bias.item()*100:+.2f} %  (systematic offset the adapter corrected)")


# ── 7. Test-set evaluation ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Test-set results")
print("=" * 60)
test_m = _run_epoch(model, test_loader, optimizer, criterion, device, training=False)
print(f"  Loss  (MSE) : {test_m['loss']:.6f}")
print(f"  MAE         : {test_m['mae']*100:.2f} %")
print(f"  RMSE        : {test_m['rmse']*100:.2f} %")
print(f"  R²          : {test_m['r2']:.4f}")


# ── 8. Per-run test plots ─────────────────────────────────────────────────────
model.eval()
test_run_dfs    = [test_df[test_df['run_name'] == r] for r in TEST_RUNS if r in test_df['run_name'].values]
per_run_metrics = []

fig, axes = plt.subplots(len(test_run_dfs), 1, figsize=(12, 3.5 * len(test_run_dfs)), squeeze=False)
for ax, run_df in zip(axes[:, 0], test_run_dfs):
    run_name = run_df['run_name'].iloc[0]
    X_run    = scaler_X.transform(run_df[FEATURE_COLS].values.astype(np.float32))
    y_true   = run_df[TARGET_COL].values

    with torch.no_grad():
        x_t    = torch.tensor(X_run, dtype=torch.float32).to(device)
        y_pred = model(x_t).cpu().numpy().flatten()

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2   = r2_score(y_true, y_pred)
    per_run_metrics.append({'run': run_name, 'MAE_%': mae*100, 'RMSE_%': rmse*100, 'R2': r2})

    ax.plot(y_true * 100, label='True SOC (Coulomb)',  color='steelblue',  lw=1.2)
    ax.plot(y_pred * 100, label='Predicted SOC',    color='orangered', lw=1.2, linestyle='--')
    ax.set_title(f'{run_name}   MAE={mae*100:.1f}%  RMSE={rmse*100:.1f}%  R²={r2:.3f}', fontsize=9)
    ax.set_ylabel('SOC (%)')
    ax.legend(fontsize=8)
    ax.grid(True, lw=0.4)

axes[-1, 0].set_xlabel('Sample index')
plt.suptitle('Adapter + Frozen Panasonic MLP → BMS test runs', fontsize=11, y=1.01)
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
ax.set_title('Adapter training history (frozen Panasonic backbone)')
ax.legend()
ax.grid(True)
plt.tight_layout()
curve_path = os.path.join(SAVE_DIR, f'{MODEL_NAME}_training_curve.png')
plt.savefig(curve_path, dpi=150)
print(f"Training curve saved → {curve_path}")
