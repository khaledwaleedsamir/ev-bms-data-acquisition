import pandas as pd
import numpy as np
import glob
import os
from datetime import datetime, timedelta

# ===================== SETTINGS =====================

DATA_FOLDER = r"C:\Users\assas\Desktop\NU\Experimental Setup\ev-bms-data-acquisition\dataset\new_runs_excel"   # folder containing Excel files
RUN_KEYWORD = "80pct_speed_25kg_load_discharge"

TIME_COLUMN = "time_string"
OUTPUT_FILE = "merged_80pct_speed_25kg_load_discharge.xlsx"

SAMPLE_PERIOD_SEC = 1  # fixed sampling rate

# =========================================================


def parse_time_string(t):
    """
    Handles:
    - b'10:50:33.28017' (bytes)
    - "b'10:50:33.28017'" (string from Excel)
    - "10:50:33.28017"
    """

    if pd.isna(t):
        return np.nan

    # Convert bytes -> string
    if isinstance(t, bytes):
        t = t.decode()

    # Remove leading b' and trailing '
    t = str(t).strip()

    if t.startswith("b'") and t.endswith("'"):
        t = t[2:-1]

    # Parse time
    dt = datetime.strptime(t, "%H:%M:%S.%f")

    return dt.hour * 3600 + dt.minute * 60 + dt.second


def seconds_to_time_string(sec):
    """
    Converts seconds since midnight -> HH:MM:SS.000000
    """
    sec = int(sec)
    t = timedelta(seconds=sec)
    total_seconds = int(t.total_seconds())

    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60

    return f"{h:02d}:{m:02d}:{s:02d}.000000"


def load_and_prepare(filepath):
    df = pd.read_excel(filepath)

    # Convert time_string to seconds
    df["_time_sec"] = df[TIME_COLUMN].apply(parse_time_string)

    # Sort by time
    df = df.sort_values("_time_sec").reset_index(drop=True)
    return df


# Find all matching files
files = sorted(glob.glob(os.path.join(DATA_FOLDER, f"*{RUN_KEYWORD}*.xlsx")))

if len(files) < 2:
    raise ValueError("Need at least two files to concatenate.")

print("Files found:")
for f in files:
    print(" -", os.path.basename(f))

# Load first file
merged_df = load_and_prepare(files[0])

# Process remaining files
for file in files[1:]:
    df_next = load_and_prepare(file)

    last_time = merged_df["_time_sec"].iloc[-1]
    next_start = df_next["_time_sec"].iloc[0]

    gap_seconds = next_start - last_time - SAMPLE_PERIOD_SEC

    if gap_seconds > 0:
        print(f"Filling gap of {gap_seconds} seconds before {os.path.basename(file)}")

        gap_times = np.arange(
            last_time + SAMPLE_PERIOD_SEC,
            next_start,
            SAMPLE_PERIOD_SEC
        )

        gap_df = pd.DataFrame({"_time_sec": gap_times})

        # Fill all other columns with zeros
        for col in merged_df.columns:
            if col not in [TIME_COLUMN, "_time_sec"]:
                gap_df[col] = 0

        merged_df = pd.concat([merged_df, gap_df], ignore_index=True)

    # Append actual data
    merged_df = pd.concat([merged_df, df_next], ignore_index=True)

# Convert seconds back to time_string
merged_df[TIME_COLUMN] = merged_df["_time_sec"].apply(seconds_to_time_string)

# Final cleanup
merged_df = merged_df.sort_values("_time_sec").reset_index(drop=True)
merged_df = merged_df.drop(columns=["_time_sec"])

# Save
merged_df.to_excel(OUTPUT_FILE, index=False)

print(f"\nMerged file saved as: {OUTPUT_FILE}")
