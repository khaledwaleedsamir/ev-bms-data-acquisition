import pandas as pd

# === LOAD ORIGINAL EXCEL FILE ===
df = pd.read_excel(
    r"C:\Users\assas\Desktop\NU\Experimental Setup\ev-bms-data-acquisition\dataset\runs_excel\run_002.xlsx"
)

# === TIME COLUMN OR SAMPLE INDEX ===
if "timestamp" in df.columns:
    time_col = "timestamp"
else:
    df["time"] = range(len(df))
    time_col = "time"

# === DOWNSAMPLE ===
df_downsampled = df.iloc[::100, :].reset_index(drop=True)

# === SAVE TO NEW EXCEL FILE ===
df_downsampled.to_excel(
    r"C:\Users\assas\Desktop\NU\Experimental Setup\ev-bms-data-acquisition\dataset\runs_excel\run_002_downsampled.xlsx",
    index=False
)

print("Downsampled Excel created successfully!")
print(f"Original rows: {len(df)}, Downsampled rows: {len(df_downsampled)}")
