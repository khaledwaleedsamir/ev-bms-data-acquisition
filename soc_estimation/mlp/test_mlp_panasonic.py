"""
Cross-dataset test: evaluate the Panasonic-trained MLP on the own collected dataset.

Loads model weights + scaler from outputs/mlp_panasonic.pth and runs inference
on every run in the local HDF5 file, reporting per-run and overall metrics.

The ground truth is the BMS-reported battery_level, which may have a systematic
offset vs the Coulomb-counting SOC used during Panasonic training.

Run from the repository root:
  python -m soc_estimation.mlp.test_mlp_panasonic
"""

import os
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from soc_estimation.mlp.mlp import MLP_SOC, ModelManager

# ──────────────────────────── CONFIG ─────────────────────────────────────────
HDF5_PATH  = r'dataset\all_data\h5_files\hoverboard_bms_dataset.h5'
SAVE_DIR   = r'soc_estimation\mlp\outputs'
MODEL_NAME = 'mlp_panasonic'

FEATURE_COLS = ['Voltage [V]', 'Current [A]', 'Temperature [degC]',
                'Cycle Charge [Ah]', 'Cycle Capacity [Wh]']
TARGET_COL   = 'SOC [-]'

# Pack topology — Panasonic model was trained on single-cell measurements.
# The Daly BMS reports pack-level voltage and current, so convert to per-cell
# before inference so the inputs land in the same range the model was trained on.
#   cell_voltage = pack_voltage / N_SERIES      (e.g. 36V pack → 3.6V/cell)
#   cell_current = pack_current / N_PARALLEL    (e.g. 9A pack  → 3A/cell)
N_SERIES   = 10   # cells in series  (10S → ~25–42 V pack)
N_PARALLEL =  3   # cells in parallel (3P)
# ─────────────────────────────────────────────────────────────────────────────

model_path  = os.path.join(SAVE_DIR, f'{MODEL_NAME}.pth')
scaler_path = os.path.join(SAVE_DIR, f'{MODEL_NAME}_scalers.pkl')

# ── Load model ────────────────────────────────────────────────────────────────
model   = MLP_SOC(input_size=len(FEATURE_COLS), hidden_sizes=[128, 64, 32, 16], output_size=1)
manager = ModelManager(model, device='cpu')
manager.load_model_weights(model_path)
scalers = joblib.load(scaler_path)
manager.scaler_X = scalers['scaler_X']
manager.model.eval()

# ── Load HDF5 data ────────────────────────────────────────────────────────────
run_dfs = []
with h5py.File(HDF5_PATH, 'r') as f:
    for run_name in f.keys():
        g         = f[run_name]['bms']
        ts_ms     = f[run_name]['timestamp_ms'][:].astype(np.float64)
        temp_vals = g['temp_values'][:]
        voltage   = g['voltage'][:].astype(np.float32)
        current   = g['current'][:].astype(np.float32)

        # Coulomb counting on per-cell values to match Panasonic single-cell ranges
        dt_s              = np.diff(ts_ms / 1000.0, prepend=ts_ms[0] / 1000.0).astype(np.float32)
        cell_i            = current / N_PARALLEL
        cell_v            = voltage / N_SERIES
        cycle_charge_ah   = np.cumsum(cell_i * dt_s / 3600.0)
        cycle_capacity_wh = np.cumsum(cell_v * cell_i * dt_s / 3600.0)

        df = pd.DataFrame({
            # Divide by pack topology to get per-cell values matching Panasonic training range
            'Voltage [V]':         voltage                                    / N_SERIES,
            'Current [A]':         current                                    / N_PARALLEL,
            'Temperature [degC]':  temp_vals.mean(axis=1).astype(np.float32),
            'Cycle Charge [Ah]':   cycle_charge_ah,
            'Cycle Capacity [Wh]': cycle_capacity_wh,
            'SOC [-]':             g['battery_level'][:].astype(np.float32) / 100.0,
            'run_name':            run_name,
        })
        run_dfs.append(df)

full_df = pd.concat(run_dfs, ignore_index=True)
print(f"Loaded {len(run_dfs)} runs  ({len(full_df):,} samples total)")

# ── Per-run inference + metrics ───────────────────────────────────────────────
per_run_metrics = []
fig_rows = len(run_dfs)
fig, axes = plt.subplots(fig_rows, 1, figsize=(12, 3 * fig_rows), squeeze=False)

for ax, run_df in zip(axes[:, 0], run_dfs):
    run_name = run_df['run_name'].iloc[0]
    X = run_df[FEATURE_COLS].values
    y_true = run_df[TARGET_COL].values

    y_pred = manager.predict(X)  # already [0, 1]

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2   = r2_score(y_true, y_pred)
    per_run_metrics.append({'run': run_name, 'MAE_%': mae*100, 'RMSE_%': rmse*100, 'R2': r2})

    ax.plot(y_true * 100, label='True SOC (BMS)', color='steelblue',  lw=1.2)
    ax.plot(y_pred * 100, label='Predicted SOC',   color='orangered', lw=1.2, linestyle='--')
    ax.set_title(f'{run_name}   MAE={mae*100:.1f}%  RMSE={rmse*100:.1f}%  R²={r2:.3f}', fontsize=9)
    ax.set_ylabel('SOC (%)')
    ax.legend(fontsize=8)
    ax.grid(True, lw=0.4)

axes[-1, 0].set_xlabel('Sample index')
plt.suptitle('Panasonic-trained MLP → Own dataset (cross-test)', fontsize=11, y=1.01)
plt.tight_layout()
plot_path = os.path.join(SAVE_DIR, f'{MODEL_NAME}_cross_test.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.show()

# ── Summary table ─────────────────────────────────────────────────────────────
metrics_df = pd.DataFrame(per_run_metrics)
print("\n" + "=" * 60)
print("Per-run metrics  (ground truth = BMS battery_level)")
print("=" * 60)
print(metrics_df.to_string(index=False, float_format=lambda x: f'{x:.2f}'))

y_all_true = full_df[TARGET_COL].values
y_all_pred = manager.predict(full_df[FEATURE_COLS].values)
print("\n--- Overall ---")
print(f"  MAE  : {mean_absolute_error(y_all_true, y_all_pred)*100:.2f} %")
print(f"  RMSE : {mean_squared_error(y_all_true, y_all_pred)**0.5*100:.2f} %")
print(f"  R²   : {r2_score(y_all_true, y_all_pred):.4f}")
print(f"\nPlot saved → {plot_path}")

# ── Save results CSV ──────────────────────────────────────────────────────────
csv_path = os.path.join(SAVE_DIR, f'{MODEL_NAME}_cross_test_metrics.csv')
metrics_df.to_csv(csv_path, index=False)
print(f"Metrics CSV saved → {csv_path}")
