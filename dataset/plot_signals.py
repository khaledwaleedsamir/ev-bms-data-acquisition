import pandas as pd
import matplotlib.pyplot as plt

def plot_signals(df, time_col, signals, title="Plot", xlabel="Time", ylabel="Value", ylim=None, legend=True, figsize=(12,6)):
    """
    Generic multi-signal plotting function.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with time and signals.
    time_col : str
        Column name representing time.
    signals : list[str] or str
        One or more column names to plot.
    title : str
        Plot title.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    ylim : tuple or None
        Y-axis limits (ymin, ymax).
    legend : bool
        Whether to show legend.
    figsize : tuple
        Figure size.
    """
    
    if isinstance(signals, str):
        signals = [signals]

    plt.figure(figsize=figsize)

    for col in signals:
        plt.plot(df[time_col], df[col], label=col)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)

    if ylim:
        plt.ylim(*ylim)

    if legend and len(signals) > 1:
        plt.legend()

    plt.tight_layout()

df = pd.read_excel(r"C:\Users\assas\Desktop\NU\Experimental Setup\ev-bms-data-acquisition\dataset\runs_excel\run_001_downsampled.xlsx")
time_col = "timestamp" if "timestamp" in df.columns else "time"
if "timestamp" not in df.columns:
    df["time"] = range(len(df))

# Column lists
cell_cols = [f"bms/cell_voltages_{i}" for i in range(10)]
temp_cols = [f"bms/temp_values_{i}" for i in range(3)]

# === Replacing your individual plots ===

plot_signals(df, time_col, "bms/voltage",
             title="Voltage vs Time",
             ylabel="Voltage (V)")

plot_signals(df, time_col, "bms/current",
             title="Current vs Time",
             ylabel="Current (A)",
             ylim=(-5, 5))

plot_signals(df, time_col, "bms/battery_level",
             title="State of Charge vs Time",
             ylabel="SoC (%)")

plot_signals(df, time_col, cell_cols,
             title="Cell Voltages Over Time (All 10 Cells)",
             ylabel="Voltage (V)")

plot_signals(df, time_col, "hoverboard/hb_board_temp",
             title="Hoverboard Temperature vs Time",
             ylabel="Temperature (°C)")

plot_signals(df, time_col, temp_cols,
             title="Battery Temperature Sensors Over Time",
             ylabel="Temperature (°C)")

plot_signals(df, time_col, ["hoverboard/hb_speedR_meas", "hoverboard/hb_speedL_meas"],
             title="Hoverboard Wheel Speeds vs Time",
             ylabel="Speed (RPM)")

plt.show()