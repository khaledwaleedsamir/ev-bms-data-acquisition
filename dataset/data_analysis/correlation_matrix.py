import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

excel_file_name = "run_005_charge_new_features"

# cols to drop
cols_to_drop = [
    "bms/cell_count",
    "bms/cycles",
    "bms/temp_sensors",
    "bms/battery_charging",
    "timestamp_ms",
    "time_s",
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
    "bms/temp_values_1",
    "hoverboard/hb_board_temp",
    "hoverboard/hb_measured_voltage",
    "hoverboard/hb_speedL_meas",
    "hoverboard/hb_speedR_meas"
]

# Load data
df = pd.read_excel(fr"C:\Users\assas\Desktop\NU\Experimental Setup\ev-bms-data-acquisition\dataset\new_runs_excel\{excel_file_name}.xlsx")
# Print columns available
print("Columns in dataset:")
print(df.columns.tolist())
# Keep only numeric columns
df_numeric = df.select_dtypes(include="number")
# Drop unwanted columns if they exist
df_filtered = df_numeric.drop(columns=cols_to_drop, errors="ignore")
# Clean column names
df_filtered.columns = df_filtered.columns.str.replace(r"^hoverboard/", "", regex=True)
# Correlation matrix
corr_matrix = df_filtered.corr(method="pearson")
# Print correlation matrix in console
print(corr_matrix)


# ---- Plot ----
fig, ax = plt.subplots(figsize=(14, 12), constrained_layout=True)

im = ax.imshow(corr_matrix)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# X-axis labels (fully visible)
ax.set_xticks(np.arange(len(corr_matrix.columns)))
ax.set_xticklabels(
    corr_matrix.columns,
    rotation=90,
    ha="right",
    rotation_mode="anchor",
    fontsize=9
)

# Y-axis labels
ax.set_yticks(np.arange(len(corr_matrix.columns)))
ax.set_yticklabels(corr_matrix.columns, fontsize=9)

# Title
ax.set_title(
    f"Pearson Correlation Matrix ({excel_file_name})",
    fontsize=14,
    pad=20
)
plt.show()