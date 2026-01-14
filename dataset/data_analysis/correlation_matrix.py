import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# constant cols to drop
cols_to_drop = [
    "bms/cell_count",
    "bms/cycles",
    "bms/temp_sensors",
    "bms/battery_charging",
    "timestamp_ms",
    # drop all cell voltages
    "bms/cell_voltages_0",
    "bms/cell_voltages_1",
    "bms/cell_voltages_2",
    "bms/cell_voltages_3",
    "bms/cell_voltages_4",
    "bms/cell_voltages_5",
    "bms/cell_voltages_6",
    "bms/cell_voltages_7",
    "bms/cell_voltages_8",
    "bms/cell_voltages_9",
    "time_s"
    # "hoverboard/hb_board_temp",
    # "hoverboard/hb_measured_voltage",
    # "hoverboard/hb_speedL_meas",
    # "hoverboard/hb_speedR_meas"
]

df = pd.read_excel(r"C:\Users\assas\Desktop\NU\Experimental Setup\ev-bms-data-acquisition\dataset\run_004_80pct_speed_15kg_load_discharge.xlsx")
# Keep only numeric columns
df_numeric = df.select_dtypes(include="number")
df_filtered = df_numeric.drop(columns=cols_to_drop, errors="ignore")
corr_matrix = df_filtered.corr(method="pearson")

print(corr_matrix)

plt.figure(figsize=(14, 12))

plt.imshow(corr_matrix)
plt.colorbar(fraction=0.046, pad=0.04)

plt.xticks(
    ticks=np.arange(len(corr_matrix.columns)),
    labels=corr_matrix.columns,
    rotation=90,
    fontsize=9
)
plt.yticks(
    ticks=np.arange(len(corr_matrix.columns)),
    labels=corr_matrix.columns,
    fontsize=9
)

plt.title("Pearson Correlation Matrix", fontsize=14)
plt.tight_layout()
plt.show()

# dV_dt = df["dV_pack_dt"].dropna().values

# # Full autocorrelation
# autocorr_full = np.correlate(dV_dt, dV_dt, mode='full')

# # Lags array
# lags = np.arange(-len(dV_dt)+1, len(dV_dt))

# plt.figure(figsize=(10,5))
# plt.plot(lags, autocorr_full)
# plt.title("Full Autocorrelation of dV/dt")
# plt.xlabel("Lag")
# plt.ylabel("Autocorrelation")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# x = df["bms/current"].dropna().values
# y = df["bms/voltage"].dropna().values

# # Make signals zero-mean for meaningful correlation
# x = x - np.mean(x)
# y = y - np.mean(y)

# cross_corr = np.correlate(x, y, mode='full')
# lags = np.arange(-len(x)+1, len(x))

# plt.figure(figsize=(10,5))
# plt.plot(lags, cross_corr)
# plt.title("Cross-Correlation between Current and Voltage")
# plt.xlabel("Lag")
# plt.ylabel("Cross-Correlation")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# cell_voltage_cols = [
#     col for col in df.columns
#     if "cell" in col.lower() and "voltage" in col.lower()
# ]

# print("Detected cell voltage columns:")
# print(cell_voltage_cols)

# df["V_cell_mean"]  = df[cell_voltage_cols].mean(axis=1)
# df["V_cell_min"]   = df[cell_voltage_cols].min(axis=1)
# df["V_cell_max"]   = df[cell_voltage_cols].max(axis=1)
# df["V_cell_std"]   = df[cell_voltage_cols].std(axis=1)
# df["time_s"] = df["timestamp_ms"] / 1000.0
# df["dV_pack_dt"] = df["bms/voltage"].diff() / df["time_s"].diff()
# df["dI_pack_dt"] = df["bms/current"].diff() / df["time_s"].diff()

# df.to_excel(r"C:\Users\assas\Desktop\NU\Experimental Setup\ev-bms-data-acquisition\dataset\runs_excel\run_001_with_cell_voltage_features_dv_di.xlsx", index=False)