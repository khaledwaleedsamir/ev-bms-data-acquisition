import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Downsample an Excel file.")
parser.add_argument("input", help="Path to input Excel file")
parser.add_argument("output", help="Path to output Excel file")
parser.add_argument("--ratio", type=int, default=10, help="Downsampling ratio (default: 10)")
args = parser.parse_args()

df = pd.read_excel(args.input)

if "timestamp" not in df.columns:
    df["time"] = range(len(df))

df_downsampled = df.iloc[::args.ratio, :].reset_index(drop=True)

df_downsampled.to_excel(args.output, index=False)

print("Downsampled Excel created successfully!")
print(f"Original rows: {len(df)}, Downsampled rows: {len(df_downsampled)}")