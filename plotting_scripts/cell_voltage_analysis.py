import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Config ────────────────────────────────────────────────────────────────────
csv_file = r"C:\Users\assas\Desktop\NU\Experimental Setup\ev-bms-data-acquisition\discharge_cell_voltages.csv"

CELL_COLUMNS = [f"cell_{i}_V" for i in range(10, 0, -1)]   # cell_10_V … cell_1_V

plt.rcParams.update({
    "font.family":         "serif",
    "font.serif":          ["Times New Roman", "DejaVu Serif"],
    "font.size":           11,
    "axes.titlesize":      12,
    "axes.labelsize":      11,
    "axes.linewidth":      0.8,
    "axes.grid":           True,
    "grid.linestyle":      "--",
    "grid.linewidth":      0.5,
    "grid.alpha":          0.6,
    "legend.fontsize":     10,
    "legend.framealpha":   0.9,
    "legend.edgecolor":    "0.8",
    "xtick.direction":     "in",
    "ytick.direction":     "in",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "lines.linewidth":     1.4,
    "figure.dpi":          150,
})

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(csv_file)
df.dropna(subset=["esp_timestamp_ms"], inplace=True)
df.reset_index(drop=True, inplace=True)

time_sec = (df["esp_timestamp_ms"] - df["esp_timestamp_ms"].iloc[0]) / 1000

# ── Average loop time ─────────────────────────────────────────────────────────
avg_loop_ms = df["loop_time_ms"].dropna().mean()
print(f"Average loop time: {avg_loop_ms:.2f} ms")

# ── Plot 1: Pack Voltage ──────────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(10, 4))

ax1.plot(time_sec, df["pack_voltage_V"].dropna(), color="#1f77b4", label="Pack Voltage")
ax1.set_title("Pack Voltage over Time")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Voltage (V)")
ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax1.legend()
fig1.tight_layout()

# ── Plot 2: Individual Cell Voltages ──────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(10, 5))

for col in CELL_COLUMNS:
    label = col.replace("_V", "").replace("_", " ").title()
    ax2.plot(time_sec, df[col].dropna(), label=label)

ax2.set_title("Individual Cell Voltages over Time")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Voltage (V)")
ax2.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax2.legend(ncol=2, fontsize=9)
fig2.tight_layout()

plt.show()
