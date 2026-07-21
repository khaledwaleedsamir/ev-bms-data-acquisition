"""
Evaluate the BMS-trained MLP on the held-out test runs.

Loads mlp_bms.pth + mlp_bms_scalers.pkl, runs inference on each test run,
and reports per-run MAE / RMSE / R² plus an overall summary.

Run from the repository root:
  python -m soc_estimation.mlp.test_mlp
"""

import os
import numpy as np
import pandas as pd
import h5py
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from soc_estimation.mlp.mlp import MLP_SOC, ModelManager

# ──────────────────────────── CONFIG ─────────────────────────────────────────
HDF5_PATH  = r'dataset\all_data\h5_files\hoverboard_bms_dataset.h5'
SAVE_DIR   = r'soc_estimation\mlp\outputs'
MODEL_NAME = 'mlp_bms'

FEATURE_COLS = ['Voltage [V]', 'Current [A]', 'Temperature [degC]',
                'Cycle Charge [Ah]', 'Cycle Capacity [Wh]']
TARGET_COL   = 'SOC [-]'

N_SERIES   = 10
N_PARALLEL =  3

HIDDEN_SIZES = [128, 64, 32]

TEST_RUNS = [
    'file2_run_013_80pct_speed_25kg_load_discharge',
    'file2_run_015_80pct_speed_discharge',
    'file3_run_002_prediction',
]
# ─────────────────────────────────────────────────────────────────────────────

model_path  = os.path.join(SAVE_DIR, f'{MODEL_NAME}.pth')
scaler_path = os.path.join(SAVE_DIR, f'{MODEL_NAME}_scalers.pkl')

# ── Load model ────────────────────────────────────────────────────────────────
model   = MLP_SOC(input_size=len(FEATURE_COLS), hidden_sizes=HIDDEN_SIZES, output_size=1)
manager = ModelManager(model, device='cpu')
manager.load_model_weights(model_path)
scalers = joblib.load(scaler_path)
manager.scaler_X = scalers['scaler_X']
manager.model.eval()


# ── Load test runs ────────────────────────────────────────────────────────────
run_dfs = []
with h5py.File(HDF5_PATH, 'r') as f:
    for run_name in TEST_RUNS:
        if run_name not in f:
            print(f"[WARNING] Run not found in HDF5, skipping: {run_name}")
            continue
        g         = f[run_name]['bms']
        ts_ms     = f[run_name]['timestamp_ms'][:].astype(np.float64)
        temp_vals = g['temp_values'][:]
        voltage   = g['voltage'][:].astype(np.float32)
        current   = g['current'][:].astype(np.float32)

        dt_s              = np.diff(ts_ms / 1000.0, prepend=ts_ms[0] / 1000.0).astype(np.float32)
        cycle_charge_ah   = np.cumsum(current * dt_s / 3600.0)
        cycle_capacity_wh = np.cumsum(voltage * current * dt_s / 3600.0)

        run_dfs.append(pd.DataFrame({
            'Voltage [V]':         voltage                                    / N_SERIES,
            'Current [A]':         current                                    / N_PARALLEL,
            'Temperature [degC]':  temp_vals.mean(axis=1).astype(np.float32),
            'Cycle Charge [Ah]':   cycle_charge_ah,
            'Cycle Capacity [Wh]': cycle_capacity_wh,
            'SOC [-]':             g['battery_level'][:].astype(np.float32)  / 100.0,
            'run_name':            run_name,
        }))

print(f"Loaded {len(run_dfs)} test runs")


# ── Per-run inference + metrics ───────────────────────────────────────────────
per_run_metrics = []
fig, axes = plt.subplots(len(run_dfs), 1, figsize=(12, 3.5 * len(run_dfs)), squeeze=False)

for ax, run_df in zip(axes[:, 0], run_dfs):
    run_name = run_df['run_name'].iloc[0]
    X_scaled = manager.scaler_X.transform(run_df[FEATURE_COLS].values.astype(np.float32))

    manager.scaler_X = None
    y_pred = manager.predict(X_scaled)
    manager.scaler_X = scalers['scaler_X']

    y_true = run_df[TARGET_COL].values
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
plt.suptitle(f'BMS MLP {HIDDEN_SIZES} — Test Results', fontsize=11, y=1.01)
plt.tight_layout()
plot_path = os.path.join(SAVE_DIR, f'{MODEL_NAME}_test_results.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"Plot saved → {plot_path}")


# ── Summary ───────────────────────────────────────────────────────────────────
metrics_df = pd.DataFrame(per_run_metrics)
print("\n" + "=" * 60)
print("Per-run metrics")
print("=" * 60)
print(metrics_df.to_string(index=False, float_format=lambda x: f'{x:.2f}'))

full_test_df = pd.concat(run_dfs, ignore_index=True)
X_all = manager.scaler_X.transform(full_test_df[FEATURE_COLS].values.astype(np.float32))
manager.scaler_X = None
y_all_pred = manager.predict(X_all)
manager.scaler_X = scalers['scaler_X']
y_all_true = full_test_df[TARGET_COL].values

print(f"\n--- Overall ---")
print(f"  MAE  : {mean_absolute_error(y_all_true, y_all_pred)*100:.2f} %")
print(f"  RMSE : {mean_squared_error(y_all_true, y_all_pred)**0.5*100:.2f} %")
print(f"  R²   : {r2_score(y_all_true, y_all_pred):.4f}")

csv_path = os.path.join(SAVE_DIR, f'{MODEL_NAME}_test_metrics.csv')
metrics_df.to_csv(csv_path, index=False)
print(f"Metrics CSV saved → {csv_path}")
