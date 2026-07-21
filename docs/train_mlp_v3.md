# `train_mlp_v3.py` — Detailed Walkthrough

Trains an MLP that predicts **how much SOC will change** over several future
horizons (5/15/30/60 s), rather than predicting absolute SOC directly.

Reference: `soc_estimation/mlp/train_mlp_v3.py`. Depends on
`soc_estimation/mlp/mlp.py` (model + training loop) and
`soc_estimation/soc_relabel.py` (ground-truth SOC label).

---

## 1. Why "delta SOC" instead of absolute SOC (lines 1–24)

v1/v2 predicted absolute `SOC(t+h)`. Problem: SOC barely moves in 5–60 s, so a
trivial **persistence baseline** (`SOC(t+h) = SOC(t)`) already scores ~0.03 pp
MAE — the model was mostly learning to copy its input, which hides whether it's
learning anything real about battery dynamics.

v3 instead predicts the **signed change**:

```
target_k(t) = soc_cc(t + HORIZONS[k]) - soc_cc(t)      for k in HORIZONS
```

This makes the persistence baseline predict **zero change**, which is a much
harder bar to beat, and forces the model to actually use current draw, dV/dt,
etc. Two baselines are computed for comparison at evaluation time:

- **`zero`** — assume no change (persistence in delta form): `MAE = mean(|target|)`
- **`coulomb`** — assume the current at time *t* stays constant for the whole
  horizon and integrate it: `delta = I(t) * h / 3600 / NOMINAL_CAP_AH * 100`

The label used is `soc_cc` (OCV-anchored coulomb-counted SOC from
`soc_relabel.py`), **not** the Daly BMS's own `battery_level`, which is
quantized to 0.1 pp steps and drifts 10–20 pp on later runs.

---

## 2. Config block (lines 42–89)

| Name | Meaning |
|---|---|
| `H5_PATH` | Source HDF5 file with all runs |
| `SAVE_DIR` | Where model weights, scalers, plots, metrics get written (gitignored) |
| `HORIZONS = [5, 15, 30, 60]` | Seconds ahead to predict, in rows (data is ~1 Hz) |
| `HIDDEN_SIZES = [64, 32, 16]` | MLP hidden layer widths |
| `FEATURE_COLS` | The 6 input features (see §3) |
| `TRAIN_RUNS` / `VAL_RUNS` / `TEST_RUNS` | Hardcoded lists of HDF5 run names — the split is **by whole run**, not by row, so no run's data leaks across splits |

To change which runs are used, edit these three lists directly. To add/remove
a feature, edit `FEATURE_COLS` **and** `engineer_features()` together — see §3.

---

## 3. Feature engineering — `engineer_features()` (lines 93–101)

Input: a per-run DataFrame with `timestamp_ms`, `voltage`, `current` (already
has `soc_cc` attached by this point, from `load_runs`).

```python
dt_s        = time delta between rows, in seconds (gap-filled to 1s, floor 0.1s)
C_rate      = current / NOMINAL_CAP_AH                     # instantaneous C-rate
dV_dt       = d(voltage)/dt, clipped to [-1, 1] V/s          # voltage slope
I_mean_10s  = rolling 10-sample mean of current              # smoothed load
I_mean_30s  = rolling 30-sample mean of current               # smoothed load, longer window
```

`delta_voltage` is **not** computed here — it comes straight from the BMS
(`bms/delta_voltage`, max-min cell voltage spread), loaded in `load_run()`.

**Final feature vector (6 features, `FEATURE_COLS`):**
`["voltage", "C_rate", "dV_dt", "delta_voltage", "I_mean_10s", "I_mean_30s"]`

All 6 are things you can compute in real time on the ESP32/PC during a live
run — no look-ahead.

---

## 4. Loading data — `load_run()`, `load_runs()` (lines 105–132)

- `load_run(f, run_name)`: pulls `timestamp_ms`, `battery_level`, `voltage`,
  `current`, `delta_voltage` out of one HDF5 run group into a flat DataFrame.
- `load_runs(h5_path, run_names)`:
  1. Opens the HDF5 file once, loops over the requested run names (warns and
     skips missing ones — handy if a run list references a run not present in
     a particular file).
  2. Calls `add_soc_cc(df)` to attach the `soc_cc` label (see §7).
  3. Drops rows where `soc_cc` is `NaN` — these are BLE dropout rows
     (`voltage <= DROPOUT_VOLTAGE`, i.e. logger read 0 while disconnected).
     Dropped **before** `engineer_features()` so the rolling means / `dV_dt`
     never see a fake zero-voltage sample.
  4. Runs `engineer_features()`.

  Returns a `dict[run_name -> DataFrame]`.

---

## 5. Building (X, Y) arrays — `build_arrays()` (lines 136–165)

For each run's DataFrame:

```python
max_h = 60                       # largest horizon
n     = len(df)
soc   = df["soc_cc"].values

X = features[0 : n - max_h]                                   # drop last max_h rows —
                                                                # no future data to compute target
Y[:, k] = soc[h : n - max_h + h] - soc[0 : n - max_h]          # shifted diff, per horizon k
```

This is the core trick: row `t` of `X` is paired with row `t` of `Y`, where
`Y[t, k]` looks `HORIZONS[k]` rows into the *future* of the *same run* to
compute the delta. Because it's a per-run slice (not a global shift), horizon
windows never cross a run boundary.

Runs shorter than `max_h` rows are skipped (can't form a single valid target).

Also returned: `I_now` (current at time *t*, for the coulomb baseline) and
`SOC_now` (absolute `soc_cc` at time *t*, only used later to reconstruct
absolute-SOC plots — never fed to the model).

All runs are then `np.vstack`-ed into one big `X_train`/`Y_train` matrix
(train/val/test built separately by calling `build_arrays()` three times).

---

## 6. Scaling — `StandardScaler` (lines 284–298)

```python
scaler_X = StandardScaler()
X_train_sc = scaler_X.fit_transform(X_train)   # fit mean/std on TRAIN ONLY
X_val_sc   = scaler_X.transform(X_val)         # reuse train's mean/std — no leakage
X_test_sc  = scaler_X.transform(X_test)

scaler_Y = StandardScaler()                    # separate scaler, same fit/transform pattern
Y_train_sc = scaler_Y.fit_transform(Y_train)
```

`StandardScaler` does **z-score standardization**, `x' = (x - mean) / std`,
computed per column — **not** min-max scaling. Output is centered at 0 with
unit variance, unbounded (see `soc_estimation/mlp/inspect_scaler.py` for a
worked example dumping raw vs. scaled values to CSV).

Targets (`Y`, the deltas) are scaled too, because raw deltas are tiny
(fractions of a percentage point) relative to typical NN weight/loss scales —
standardizing keeps MSE loss well-conditioned for Adam.

Both scalers are saved together to `scalers_v3.pkl` (`joblib.dump`) — needed
at inference time to replicate the exact same transform, and to inverse-scale
the model's raw output back into real percentage points (`predict_all()`, §8).

---

## 7. `soc_cc` label — `soc_relabel.py` (background)

Not part of `train_mlp_v3.py` itself, but essential to understand since it's
the training target's source:

1. Interpolates a rest-OCV-vs-SOC curve (from HPPC test data,
   `output_figures/hppc/hppc_results.csv`) to get **two SOC anchors** — one
   each at the start and end of a run — correcting for IR drop under load if
   the endpoint wasn't at rest.
2. Coulomb-integrates the logged current across the whole run
   (`cum`, in percentage points, clipping any single timestep gap to 5 s so a
   logging dropout doesn't get treated as continuous current draw).
3. Least-squares-fits the run's *starting* SOC so the coulomb trajectory
   matches both anchors, weighting the anchor closer to rest more heavily.
4. Rows where `voltage <= DROPOUT_VOLTAGE` (BLE dropout, logged as 0) get
   `NaN` — this is what `load_runs()` filters out before feature engineering.

Net effect: absolute-SOC accuracy is roughly ±2 pp, but **within-run deltas**
(exactly what v3 predicts) are much tighter since they come purely from
current integration.

---

## 8. Model, training, evaluation (lines 264–355)

```python
model = MLP_SOC(
    input_size=6,                 # len(FEATURE_COLS)
    hidden_sizes=[64, 32, 16],
    output_size=4,                # len(HORIZONS) — one delta output per horizon
    output_activation="none",     # NOT sigmoid — deltas are signed, can be +/- and >1
)
```

See `mlp.py`:
- `MLP_SOC`: stack of `Linear -> ReLU -> Dropout(0.2)` blocks per hidden layer,
  final `Linear` to `output_size`. `output_activation="sigmoid"` (the default,
  used by v1/v2 for absolute SOC in [0,1]) is turned **off** here since deltas
  aren't bounded to [0,1].
- `ModelManager.start_training()`: standard train/val loop with early
  stopping (`patience=30` epochs without val-loss improvement) and
  checkpointing the best model to `save_path` every time val loss improves.
  Optimizer: Adam, `lr=3e-4`. Loss: `MSELoss` on the **scaled** targets (so
  all 4 horizons contribute comparably to the loss regardless of their
  differing natural magnitude).

After training, `predict_all()` runs inference in eval mode and calls
`scaler_Y.inverse_transform()` to convert model output back to real
percentage-point deltas before computing metrics — so the printed MAE/RMSE/R²
in `evaluate()` are in actual "% SOC" units, comparable directly to the `zero`
and `coulomb` baselines.

`plot_predictions()` reconstructs an absolute-SOC curve purely for
visualization: `SOC(t) + predicted_delta` vs `SOC(t) + actual_delta` — both
anchored to the same true `SOC(t)`, so it's just a different view of the same
delta error, not a new/independent quantity.

---

## 9. Output artifacts (all in `soc_estimation/mlp/outputs/`, gitignored)

| File | Contents |
|---|---|
| `mlp_soc_v3_delta.pth` | Best model weights (by val loss) |
| `scalers_v3.pkl` | `{"scaler_X": ..., "scaler_Y": ...}` — needed for any future inference |
| `test_metrics_v3.csv` | Per-horizon MAE/RMSE/R² + both baselines, test set |
| `training_history_v3.png` | Train/val loss curves |
| `test_predictions_v3.png` | Actual vs. predicted absolute SOC, per horizon, first 2000 test samples |

---

## Common edit points

- **Add/remove a feature**: edit `FEATURE_COLS` (line 54) and, if it's derived
  rather than raw, add the computation to `engineer_features()` (line 93).
  Re-run — `input_size` is inferred from `len(FEATURE_COLS)` automatically.
- **Change horizons**: edit `HORIZONS` (line 47). `output_size` follows
  automatically via `len(HORIZONS)`.
- **Change train/val/test split**: edit the three run-name lists (lines
  58–89). Must be run names present in `H5_PATH`.
- **Change model capacity**: edit `HIDDEN_SIZES` (line 48).
- **Change label source**: swap out `add_soc_cc` / `soc_cc` for a different
  target column throughout `load_runs()` and `build_arrays()`.
