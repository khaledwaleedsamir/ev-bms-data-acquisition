import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# Folder containing Excel files
data_folder = r"C:\Users\assas\Desktop\NU\Experimental Setup\ev-bms-data-acquisition\dataset\test_excels"
excel_files = glob.glob(os.path.join(data_folder, "*.xlsx"))
print("Found Excel files:", excel_files)

plt.figure(figsize=(10, 6))

for file in excel_files:
    # Read Excel file
    df = pd.read_excel(file)

    # ---- SELECT COLUMNS ----
    time_ms = df["timestamp_ms"]
    current = -df["bms/current"]

    # ---- CONVERT ms TIMESTAMP ----
    # Convert to datetime
    time_dt = pd.to_datetime(time_ms, unit="ms")

    # Convert to elapsed time in seconds
    time_s = (time_dt - time_dt.iloc[0]).dt.total_seconds()
    print(time_s)
    print(current)

    # ---- LEGEND LABEL FROM FILENAME ----
    filename = os.path.basename(file).lower()

    if "80pct" in filename:
        speed = "80% Speed"
    elif "50pct" in filename:
        speed = "50% Speed"
    else:
        speed = "Unknown Speed"

    if "noload" in filename:
        load = "No Load"
    elif "rollers_weights" in filename:
        load = "Rollers + Weights"
    elif "rollers" in filename:
        load = "Rollers Only"
    else:
        load = "Unknown Load"

    label = f"{speed} – {load}"

    # ---- PLOT ----
    plt.plot(time_s, current, label=label)

# ---- FORMATTING ----
plt.xlabel("Time (s)")
plt.ylabel("Current (A)")
plt.title("Current vs Time at Different Speeds and Loads")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()

plt.show()