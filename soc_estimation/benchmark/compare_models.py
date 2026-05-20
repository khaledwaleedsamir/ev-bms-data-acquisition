"""
SOC Estimation – Automated Comparative Model Study
===================================================
* Same train / val / test run-splits as train_mlp.py
* Expanded feature set with automated selection (MI + RF SelectFromModel)
* Tunes 7–9 classical ML models via RandomizedSearchCV
* Loads the pre-trained MLP for a direct comparison
* Saves a results CSV and plots to soc_estimation/AutoML/outputs/
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import h5py
import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel, mutual_info_regression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[info] XGBoost not installed – skipping. (pip install xgboost)")

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("[info] LightGBM not installed – skipping. (pip install lightgbm)")

# ─────────────────────────────── Paths ────────────────────────────────────────
BASE        = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PATH   = os.path.join(BASE, "dataset", "all_data", "h5_files", "hoverboard_bms_dataset.h5")
MLP_WEIGHTS = os.path.join(BASE, "soc_estimation", "mlp", "outputs", "mlp_model_3_features-64-32-16-1.pth")
MLP_SCALERS = os.path.join(BASE, "soc_estimation", "mlp", "outputs", "scalers_3_features.pkl")
OUT_DIR     = os.path.join(BASE, "soc_estimation", "AutoML", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

CSV_FILES = [
    (os.path.join(BASE, "dataset", "all_data", "esp_run-001_discharge_80pct_speed.csv"),
     "esp_run-001_discharge_80pct_speed"),
    (os.path.join(BASE, "dataset", "all_data", "esp_run-002_charge.csv"),
     "esp_run-002_charge"),
    (os.path.join(BASE, "dataset", "all_data", "esp_run-003_discharge_80pct_speed.csv"),
     "esp_run-003_discharge_80pct_speed"),
]

# ──────────────────────── Run-based splits (identical to train_mlp.py) ────────
TRAIN_RUNS = [
    "file1_run_001", "file1_run_002", "file1_run_003",
    "file2_run_001_40pct_speed_15kg_load_discharge",
    "file2_run_002_40pct_speed_15kg_load_discharge",
    "file2_run_003_charge",
    "file2_run_009_40pct_speed_25kg_load_discharge",
    "file2_run_010_80pct_speed_25kg_load_discharge",
    "file2_run_011_80pct_speed_25kg_load_discharge",
    "file2_run_012_80pct_speed_25kg_load_discharge",
    "file2_run_013_80pct_speed_25kg_load_discharge",
    "file2_run_014_charge",
    "file2_run_016_charge", "file2_run_017_charge", "file2_run_018_charge",
    "esp_run-001_discharge_80pct_speed", "esp_run-002_charge",
]
VAL_RUNS = [
    "file1_run_004", "file1_run_005",
    "file2_run_004_80pct_speed_15kg_load_discharge",
    "file2_run_005_charge",
    "file2_run_006_60pct_speed_15kg_load_discharge",
    "file2_run_007_60pct_speed_15kg_load_discharge",
    "file2_run_008_charge",
    "file3_run_003_speed_profile_1",
]
TEST_RUNS = [
    "file2_run_015_80pct_speed_discharge",
    "file2_run_018_charge",
    "esp_run-003_discharge_80pct_speed",
]

# ─────────────────────────────── Features ─────────────────────────────────────
ALL_FEATURES = ["Voltage [V]", "Current [A]", "Temperature [degC]"]
MLP_FEATURES = ALL_FEATURES
TARGET = "SOC [-]"


# ─────────────────────────────── Data loading ─────────────────────────────────
def load_data() -> pd.DataFrame:
    run_dfs = []

    with h5py.File(DATA_PATH, "r") as f:
        for run_name in f.keys():
            g   = f[run_name]
            bms = g["bms"]
            tv  = bms["temp_values"][:]           # shape (N, 3)

            df = pd.DataFrame({
                "timestamp_ms":        g["timestamp_ms"][:],
                "SOC [-]":             bms["battery_level"][:] / 100.0,
                "Voltage [V]":         bms["voltage"][:],
                "Current [A]":         bms["current"][:],
                "Power [W]":           bms["power"][:],
                "Cycle Capacity [Wh]": bms["cycle_capacity"][:],
                "Cycle Charge [Ah]":   bms["cycle_charge"][:],
                "temp_1":              tv[:, 0],
                "temp_2":              tv[:, 1],
                "temp_3":              tv[:, 2],
                "run_name":            run_name,
            })
            df["Temperature [degC]"] = tv.mean(axis=1)
            df["temp_spread"]        = tv.max(axis=1) - tv.min(axis=1)
            run_dfs.append(df)

    for path, run_name in CSV_FILES:
        if not os.path.exists(path):
            print(f"  [warn] CSV not found, skipping: {path}")
            continue
        dc = pd.read_csv(path)
        df = pd.DataFrame({
            "timestamp_ms":        dc["esp_timestamp_ms"],
            "SOC [-]":             dc["bms_soc_pct"] / 100.0,
            "Voltage [V]":         dc["voltage_V"],
            "Current [A]":         dc["current_A"],
            "Power [W]":           dc["voltage_V"] * dc["current_A"],
            "Cycle Capacity [Wh]": dc["cycle_capacity_Wh"],
            "Cycle Charge [Ah]":   dc["cycle_charge_Ah"],
            "Temperature [degC]":  dc["temperature_degC"],
            "temp_1":              dc["temperature_degC"],
            "temp_2":              dc["temperature_degC"],
            "temp_3":              dc["temperature_degC"],
            "temp_spread":         0.0,   # single sensor – no spread available
            "run_name":            run_name,
        })
        run_dfs.append(df)

    return pd.concat(run_dfs, ignore_index=True)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Returns MAE / RMSE in SOC % and R²."""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    return {
        "MAE [%]":  mean_absolute_error(y_true, y_pred) * 100,
        "RMSE [%]": root_mean_squared_error(y_true, y_pred) * 100,
        "R²":       r2_score(y_true, y_pred),
    }


# ═════════════════════════════════ Main ═══════════════════════════════════════
print("=" * 65)
print("  SOC Estimation – Comparative Model Study")
print("=" * 65)

# ── 1. Load & split ───────────────────────────────────────────────────────────
print("\n[1] Loading data …")
bms_df = load_data()

train_df = bms_df[bms_df.run_name.isin(TRAIN_RUNS)].dropna(subset=ALL_FEATURES + [TARGET]).reset_index(drop=True)
val_df   = bms_df[bms_df.run_name.isin(VAL_RUNS)  ].dropna(subset=ALL_FEATURES + [TARGET]).reset_index(drop=True)
test_df  = bms_df[bms_df.run_name.isin(TEST_RUNS) ].dropna(subset=ALL_FEATURES + [TARGET]).reset_index(drop=True)

print(f"   Train : {len(train_df):,} samples across {train_df.run_name.nunique()} runs")
print(f"   Val   : {len(val_df):,} samples across {val_df.run_name.nunique()} runs")
print(f"   Test  : {len(test_df):,} samples across {test_df.run_name.nunique()} runs")

X_train = train_df[ALL_FEATURES].values.astype(np.float32)
y_train = train_df[TARGET].values.astype(np.float32)
X_val   = val_df[ALL_FEATURES].values.astype(np.float32)
y_val   = val_df[TARGET].values.astype(np.float32)
X_test  = test_df[ALL_FEATURES].values.astype(np.float32)
y_test  = test_df[TARGET].values.astype(np.float32)

# ── 2. Mutual information feature ranking ─────────────────────────────────────
print("\n[2] Mutual information feature ranking …")
mi_scores = mutual_info_regression(X_train, y_train, random_state=42)
mi_df = (
    pd.DataFrame({"Feature": ALL_FEATURES, "MI Score": mi_scores})
    .sort_values("MI Score", ascending=False)
    .reset_index(drop=True)
)
print(mi_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(mi_df["Feature"], mi_df["MI Score"], color="steelblue", edgecolor="k", linewidth=0.5)
ax.set_xlabel("Mutual Information Score")
ax.set_title("Feature Relevance – Mutual Information with SOC")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "01_feature_mi_scores.png"), dpi=150)
plt.close()
print(f"   Saved → 01_feature_mi_scores.png")

# ── 3. Automated feature selection (RF SelectFromModel, threshold = mean) ─────
print("\n[3] Automated feature selection …")
rf_sel = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
rf_sel.fit(X_train, y_train)

selector  = SelectFromModel(rf_sel, prefit=True, threshold="mean")
sel_mask  = selector.get_support()
sel_feats = [f for f, s in zip(ALL_FEATURES, sel_mask) if s]
dropped   = [f for f, s in zip(ALL_FEATURES, sel_mask) if not s]
print(f"   Selected ({len(sel_feats)}): {sel_feats}")
print(f"   Dropped  ({len(dropped)}): {dropped}")

# Feature importance plot
fi       = pd.Series(rf_sel.feature_importances_, index=ALL_FEATURES).sort_values()
thresh   = rf_sel.feature_importances_.mean()
colors   = ["teal" if fi[f] >= thresh else "lightcoral" for f in fi.index]

fig, ax = plt.subplots(figsize=(8, 4))
fi.plot.barh(ax=ax, color=colors, edgecolor="k", linewidth=0.5)
ax.axvline(thresh, color="red", linestyle="--", linewidth=1.2, label="Mean threshold")
ax.set_xlabel("Feature Importance")
ax.set_title("RF Feature Importance  (teal = selected, red = dropped)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "02_feature_importances.png"), dpi=150)
plt.close()
print(f"   Saved → 02_feature_importances.png")

# Apply selection then scale (scaler fitted on train only)
X_train_sel = selector.transform(X_train)
X_val_sel   = selector.transform(X_val)
X_test_sel  = selector.transform(X_test)

scaler      = StandardScaler()
X_train_sc  = scaler.fit_transform(X_train_sel)
X_val_sc    = scaler.transform(X_val_sel)
X_test_sc   = scaler.transform(X_test_sel)

joblib.dump(
    {"selector": selector, "scaler": scaler, "selected_features": sel_feats},
    os.path.join(OUT_DIR, "feature_pipeline.pkl"),
)

# ── 4. Model catalogue + hyperparameter grids ─────────────────────────────────
MODELS: dict[str, tuple] = {
    "Ridge": (
        Ridge(),
        {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]},
    ),
    "Lasso": (
        Lasso(max_iter=20_000),
        {"alpha": [1e-5, 1e-4, 1e-3, 0.01, 0.1, 1.0]},
    ),
    "SVR": (
        SVR(kernel="rbf"),
        {"C": [0.1, 1, 10, 100], "epsilon": [0.001, 0.005, 0.01, 0.05], "gamma": ["scale", "auto"]},
    ),
    "KNN": (
        KNeighborsRegressor(n_jobs=-1),
        {"n_neighbors": [3, 5, 7, 10, 15, 20, 30], "weights": ["uniform", "distance"], "p": [1, 2]},
    ),
    "Random Forest": (
        RandomForestRegressor(random_state=42, n_jobs=-1),
        {"n_estimators": [100, 200, 300, 500],
         "max_depth": [None, 10, 20, 30],
         "min_samples_leaf": [1, 2, 4],
         "max_features": ["sqrt", "log2", 0.5]},
    ),
    "Extra Trees": (
        ExtraTreesRegressor(random_state=42, n_jobs=-1),
        {"n_estimators": [100, 200, 300, 500],
         "max_depth": [None, 10, 20],
         "min_samples_leaf": [1, 2, 4],
         "max_features": ["sqrt", "log2", 0.5]},
    ),
    "HistGBM": (
        HistGradientBoostingRegressor(random_state=42),
        {"max_iter": [100, 200, 300, 500],
         "max_depth": [3, 5, 7, None],
         "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
         "l2_regularization": [0.0, 0.1, 1.0],
         "min_samples_leaf": [10, 20, 30]},
    ),
}

if HAS_XGB:
    MODELS["XGBoost"] = (
        XGBRegressor(random_state=42, n_jobs=-1, tree_method="hist", verbosity=0),
        {"n_estimators": [100, 200, 300, 500],
         "max_depth": [3, 5, 7, 9],
         "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
         "subsample": [0.7, 0.8, 0.9, 1.0],
         "colsample_bytree": [0.6, 0.8, 1.0],
         "reg_alpha": [0, 0.1, 1.0],
         "reg_lambda": [0.5, 1.0, 2.0]},
    )

if HAS_LGBM:
    MODELS["LightGBM"] = (
        LGBMRegressor(random_state=42, n_jobs=-1, verbosity=-1),
        {"n_estimators": [100, 200, 300, 500],
         "max_depth": [-1, 5, 10, 15],
         "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
         "num_leaves": [31, 63, 127],
         "subsample": [0.7, 0.8, 1.0],
         "reg_alpha": [0, 0.1, 1.0],
         "reg_lambda": [0.5, 1.0, 2.0]},
    )

# ── 5. Train & evaluate all models ────────────────────────────────────────────
print(f"\n[4] Training {len(MODELS)} models (RandomizedSearchCV cv=3, n_iter=30) …\n")

results      = []
test_preds   = {}   # model_name → np.ndarray (length = len(test_df))

for name, (estimator, param_grid) in MODELS.items():
    print(f"  ── {name:<18}", end=" ", flush=True)
    search = RandomizedSearchCV(
        estimator, param_grid,
        n_iter=30, cv=3,
        scoring="neg_mean_absolute_error",
        random_state=42, n_jobs=-1, refit=True,
    )
    search.fit(X_train_sc, y_train)
    best = search.best_estimator_

    val_pred  = np.clip(best.predict(X_val_sc),  0.0, 1.0)
    test_pred = np.clip(best.predict(X_test_sc), 0.0, 1.0)

    val_m  = compute_metrics(y_val,  val_pred)
    test_m = compute_metrics(y_test, test_pred)

    results.append({
        "Model":        name,
        "Features":     ", ".join(sel_feats),
        "Best Params":  str(search.best_params_),
        **{f"Val {k}":  v for k, v in val_m.items()},
        **{f"Test {k}": v for k, v in test_m.items()},
    })
    test_preds[name] = test_pred

    print(f"val MAE = {val_m['MAE [%]']:5.2f}%   test MAE = {test_m['MAE [%]']:5.2f}%   R² = {test_m['R²']:.4f}")

# ── 6. MLP evaluation (pre-trained weights, original 5 features + its scaler) ─
print(f"\n  ── {'MLP (pre-trained)':<18}", end=" ", flush=True)
mlp_ok = os.path.exists(MLP_WEIGHTS) and os.path.exists(MLP_SCALERS)

if mlp_ok:
    try:
        sys.path.insert(0, BASE)
        import torch
        from soc_estimation.mlp.mlp import MLP_SOC, ModelManager

        mlp_model = MLP_SOC(input_size=len(MLP_FEATURES), hidden_sizes=[64, 32, 16], output_size=1)
        mlp_mgr   = ModelManager(mlp_model, device="cpu")
        mlp_mgr.load_model_weights(MLP_WEIGHTS)
        mlp_mgr.load_scalers(MLP_SCALERS)
        mlp_mgr.model.eval()

        mlp_val_pred  = np.clip(mlp_mgr.predict(val_df[MLP_FEATURES].values.astype(np.float32)),  0.0, 1.0)
        mlp_test_pred = np.clip(mlp_mgr.predict(test_df[MLP_FEATURES].values.astype(np.float32)), 0.0, 1.0)

        val_m  = compute_metrics(y_val,  mlp_val_pred)
        test_m = compute_metrics(y_test, mlp_test_pred)

        results.append({
            "Model":        "MLP (pre-trained)",
            "Features":     ", ".join(MLP_FEATURES),
            "Best Params":  "hidden=[64,32,16], lr=1e-4, Adam, MSE, patience=20",
            **{f"Val {k}":  v for k, v in val_m.items()},
            **{f"Test {k}": v for k, v in test_m.items()},
        })
        test_preds["MLP (pre-trained)"] = mlp_test_pred
        print(f"val MAE = {val_m['MAE [%]']:5.2f}%   test MAE = {test_m['MAE [%]']:5.2f}%   R² = {test_m['R²']:.4f}")

    except Exception as exc:
        print(f"\n  [warn] MLP evaluation failed: {exc}")
else:
    print(f"[skip] weights not found at {MLP_WEIGHTS}")

# ── 7. Results table ──────────────────────────────────────────────────────────
results_df = pd.DataFrame(results).sort_values("Test MAE [%]").reset_index(drop=True)

print("\n" + "=" * 65)
print("  Test Set Results (sorted by MAE)")
print("=" * 65)
display = results_df[["Model", "Test MAE [%]", "Test RMSE [%]", "Test R²", "Val MAE [%]", "Val R²"]]
print(display.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

csv_path = os.path.join(OUT_DIR, "results_table.csv")
results_df.to_csv(csv_path, index=False)
print(f"\nFull results → {csv_path}")

# ── 8. Plots ──────────────────────────────────────────────────────────────────
print("\n[5] Generating plots …")

# 8a. Metric comparison bar chart (MAE, RMSE, R²)
metric_df = results_df.set_index("Model")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, col, color, ascending in zip(
    axes,
    ["Test MAE [%]", "Test RMSE [%]", "Test R²"],
    ["steelblue", "coral", "mediumseagreen"],
    [True, True, False],
):
    vals = metric_df[col].sort_values(ascending=ascending)
    bars = ax.barh(vals.index, vals.values, color=color, edgecolor="k", linewidth=0.5)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    ax.set_xlabel(col)
    ax.set_title(col)
    ax.invert_yaxis()
    ax.grid(axis="x", linewidth=0.3)

plt.suptitle("Model Comparison on Test Set", fontweight="bold", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "03_model_comparison_bars.png"), dpi=150)
plt.close()
print("   Saved → 03_model_comparison_bars.png")

# 8b. Predicted vs actual SOC for each test run, one subplot per model
n_models = len(test_preds)
ncols    = 3
nrows    = (n_models + ncols - 1) // ncols

for run_name in TEST_RUNS:
    run_mask = (test_df["run_name"] == run_name).values
    if run_mask.sum() == 0:
        continue

    y_run    = y_test[run_mask]
    x_axis   = np.arange(len(y_run))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3.5), squeeze=False)
    axes_flat = axes.flatten()

    for i, (model_name, pred_full) in enumerate(test_preds.items()):
        ax       = axes_flat[i]
        pred_run = pred_full[run_mask]
        m        = compute_metrics(y_run, pred_run)

        ax.plot(x_axis, y_run  * 100, "b-",  linewidth=1.2, label="Actual")
        ax.plot(x_axis, pred_run * 100, "r--", linewidth=1.2, label="Predicted")
        ax.set_title(
            f"{model_name}\n"
            f"MAE={m['MAE [%]']:.2f}%  RMSE={m['RMSE [%]']:.2f}%  R²={m['R²']:.3f}",
            fontsize=9,
        )
        ax.set_xlabel("Sample index")
        ax.set_ylabel("SOC (%)")
        ax.legend(fontsize=7)
        ax.grid(linewidth=0.3)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    safe_name = run_name.replace("/", "_")
    plt.suptitle(f"Test run: {run_name}", fontweight="bold", fontsize=11)
    plt.tight_layout()
    fname = f"04_predictions_{safe_name}.png"
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=150)
    plt.close()
    print(f"   Saved → {fname}")

# 8c. All models overlaid on the full test set (concatenated)
fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(y_test * 100, "k-", linewidth=1.5, label="Actual SOC", zorder=10)

cmap   = plt.cm.tab10
colors = [cmap(i % 10) for i in range(n_models)]
for (model_name, pred), color in zip(test_preds.items(), colors):
    ax.plot(pred * 100, "--", linewidth=1.0, color=color, label=model_name, alpha=0.8)

# Vertical separators between test runs
boundary = 0
for run_name in TEST_RUNS[:-1]:
    boundary += (test_df["run_name"] == run_name).sum()
    ax.axvline(boundary, color="grey", linestyle=":", linewidth=1.0)
    ax.text(boundary + 2, 102, run_name.split("_")[2], fontsize=7, color="grey", rotation=90, va="top")

ax.set_xlabel("Sample index (all test runs concatenated)")
ax.set_ylabel("SOC (%)")
ax.set_title("All Models – Full Test Set", fontweight="bold")
ax.legend(fontsize=8, ncol=3, loc="lower left")
ax.grid(linewidth=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "05_all_models_test_overlay.png"), dpi=150)
plt.close()
print("   Saved → 05_all_models_test_overlay.png")

print(f"\nAll outputs written to: {OUT_DIR}")
print("Done.")
