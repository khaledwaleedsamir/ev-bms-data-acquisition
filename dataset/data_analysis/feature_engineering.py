import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

excel_file_name = "run_004_80pct_speed_15kg_load_discharge"

# Load data
df = pd.read_excel(fr"C:\Users\assas\Desktop\NU\Experimental Setup\ev-bms-data-acquisition\dataset\new_runs_excel\{excel_file_name}.xlsx")
# Print columns available
print("Columns in dataset:")
print(df.columns.tolist())

# Extract cell voltage columns
cell_voltage_cols = [
    col for col in df.columns
    if "cell" in col.lower() and "voltage" in col.lower()
]
print("Detected cell voltage columns:")
print(cell_voltage_cols)

# create new features based on cell voltages
df["V_cell_mean"]  = df[cell_voltage_cols].mean(axis=1) # average cell voltage
df["V_cell_min"]   = df[cell_voltage_cols].min(axis=1)  # minimum cell voltage
df["V_cell_max"]   = df[cell_voltage_cols].max(axis=1)  # maximum cell voltage
df["V_cell_std"]   = df[cell_voltage_cols].std(axis=1)  # standard deviation of cell voltages

# create time-based features
df["time_s"] = df["timestamp_ms"] / 1000.0
df["dV_pack_dt"] = df["bms/voltage"].diff() / df["time_s"].diff() # rate of change of pack voltage
df["dI_pack_dt"] = df["bms/current"].diff() / df["time_s"].diff() # rate of change of pack current
df["dSOC_dt"] = df["bms/battery_level"].diff() / df["time_s"].diff() # rate of change of SOC

# Save the new dataframe with features
df.to_excel(fr"C:\Users\assas\Desktop\NU\Experimental Setup\ev-bms-data-acquisition\dataset\new_runs_excel\{excel_file_name}_new_features.xlsx", index=False)
print(f"Saved new features to {excel_file_name}_new_features.xlsx")