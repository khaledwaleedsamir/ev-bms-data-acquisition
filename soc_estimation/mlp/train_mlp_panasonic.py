"""
Train the MLP SOC model on the Panasonic 18650PF benchmark dataset.

Split strategy
--------------
  Train : Cycle_1/2/3, US06, HWFET, UDDS, LA92  (all 5 temperatures)
  Val   : Cycle_4                                 (all 5 temperatures)
  Test  : NN drive cycle                          (all 5 temperatures)

After training, saved artifacts:
  outputs/mlp_panasonic.pth          model weights (best val loss)
  outputs/mlp_panasonic_scalers.pkl  StandardScaler for features
  outputs/mlp_panasonic_training_curve.png

Run from the repository root:
  python -m soc_estimation.mlp.train_mlp_panasonic
"""

import os
import sys
import numpy as np
import torch
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

from soc_estimation.mlp.mlp import MLP_SOC, ModelManager
from soc_estimation.panasonic_loader import load_all_drive_cycles, split_by_cycle

# ──────────────────────────── CONFIG ─────────────────────────────────────────
PANASONIC_ROOT = r'dataset\Panasonic 18650PF Data'
SAVE_DIR       = r'soc_estimation\mlp\outputs'
MODEL_NAME     = 'mlp_panasonic'

FEATURE_COLS  = ['Voltage [V]', 'Current [A]', 'Temperature [degC]',
                 'Cycle Charge [Ah]', 'Cycle Capacity [Wh]']
TARGET_COL    = 'SOC [-]'

HIDDEN_SIZES  = [128, 64, 32, 16]
BATCH_SIZE    = 256
EPOCHS        = 200
PATIENCE      = 20
LR            = 1e-3
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(SAVE_DIR, exist_ok=True)

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("=" * 60)
print("Loading Panasonic 18650PF drive-cycle data")
print("=" * 60)
all_dfs = load_all_drive_cycles(PANASONIC_ROOT)

train_df, val_df, test_df = split_by_cycle(all_dfs)
print(f"\nSplit summary")
print(f"  Train : {len(train_df):>9,} samples")
print(f"  Val   : {len(val_df):>9,} samples")
print(f"  Test  : {len(test_df):>9,} samples  (NN drive cycle)")

# ── 2. Feature / target arrays ────────────────────────────────────────────────
X_train = train_df[FEATURE_COLS].values.astype(np.float32)
y_train = train_df[TARGET_COL].values.reshape(-1, 1).astype(np.float32)
X_val   = val_df[FEATURE_COLS].values.astype(np.float32)
y_val   = val_df[TARGET_COL].values.reshape(-1, 1).astype(np.float32)
X_test  = test_df[FEATURE_COLS].values.astype(np.float32)
y_test  = test_df[TARGET_COL].values.reshape(-1, 1).astype(np.float32)

# ── 3. Scaling (features only; y stays in [0,1] — Sigmoid already bounds it) ─
scaler_X = StandardScaler()
X_train_s = scaler_X.fit_transform(X_train)
X_val_s   = scaler_X.transform(X_val)
X_test_s  = scaler_X.transform(X_test)

scaler_path = os.path.join(SAVE_DIR, f'{MODEL_NAME}_scalers.pkl')
joblib.dump({'scaler_X': scaler_X, 'scaler_y': None}, scaler_path)
print(f"\nScaler saved  → {scaler_path}")

# ── 4. DataLoaders ────────────────────────────────────────────────────────────
class _PairDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(_PairDataset(X_train_s, y_train), batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(_PairDataset(X_val_s,   y_val),   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(_PairDataset(X_test_s,  y_test),  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ── 5. Model + manager ────────────────────────────────────────────────────────
device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model     = MLP_SOC(input_size=len(FEATURE_COLS), hidden_sizes=HIDDEN_SIZES, output_size=1)  # input_size=5
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = torch.nn.MSELoss()
manager   = ModelManager(model, device=device, optimizer=optimizer, criterion=criterion)

# ── 6. Training ───────────────────────────────────────────────────────────────
model_path = os.path.join(SAVE_DIR, f'{MODEL_NAME}.pth')
print(f"\nTraining on {device}  (model will be saved to {model_path})")
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
print("Test-set results  (NN drive cycle, all 5 temperatures)")
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
ax.set_title('Panasonic MLP — Training History')
ax.legend()
ax.grid(True)
plt.tight_layout()
curve_path = os.path.join(SAVE_DIR, f'{MODEL_NAME}_training_curve.png')
plt.savefig(curve_path, dpi=150)
plt.show()
print(f"\nTraining curve saved → {curve_path}")
