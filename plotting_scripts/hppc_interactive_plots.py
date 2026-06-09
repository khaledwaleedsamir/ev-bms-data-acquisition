"""
Interactive publication-quality plots for the full HPPC test.

Generates 7 HTML figures (one per signal group) saved to output_figures/hppc/.
Each opens in the default browser with full zoom / pan / hover.

Signals plotted:
  1. Pack voltage
  2. Current
  3. Temperature (3 BMS sensors)
  4. SOC
  5. Cell voltages (10 cells)
  6. Hoverboard speed (L + R)
  7. Hoverboard board temperature
"""

import os
import webbrowser
import pandas as pd
import plotly.graph_objects as go

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(ROOT, "dataset", "all_data", "csv", "hppc_full.csv")
OUT_DIR  = os.path.join(ROOT, "output_figures", "hppc")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading CSV …", end=" ", flush=True)
df = pd.read_csv(CSV_PATH)
t  = df["time_s"]
print(f"{len(df):,} samples")

# ── Shared style ───────────────────────────────────────────────────────────────
TEMPLATE   = "plotly_white"
FONT_FAMILY = "Arial, sans-serif"
FONT_SIZE   = 14
AXIS_COLOR  = "#333333"
GRID_COLOR  = "#e0e0e0"

def base_layout(title, xlab, ylab):
    return dict(
        template=TEMPLATE,
        title=dict(text=title, font=dict(family=FONT_FAMILY, size=16, color=AXIS_COLOR)),
        xaxis=dict(
            title=dict(text=xlab, font=dict(family=FONT_FAMILY, size=FONT_SIZE)),
            tickfont=dict(family=FONT_FAMILY, size=12),
            showgrid=True, gridcolor=GRID_COLOR, gridwidth=1,
            showline=True, linecolor=AXIS_COLOR, linewidth=1.5,
            mirror=True, ticks="outside", ticklen=5,
        ),
        yaxis=dict(
            title=dict(text=ylab, font=dict(family=FONT_FAMILY, size=FONT_SIZE)),
            tickfont=dict(family=FONT_FAMILY, size=12),
            showgrid=True, gridcolor=GRID_COLOR, gridwidth=1,
            showline=True, linecolor=AXIS_COLOR, linewidth=1.5,
            mirror=True, ticks="outside", ticklen=5,
        ),
        legend=dict(
            font=dict(family=FONT_FAMILY, size=12),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor=AXIS_COLOR, borderwidth=1,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=70, r=30, t=60, b=60),
        hovermode="x unified",
    )

def save_and_open(fig, filename):
    path = os.path.join(OUT_DIR, filename)
    fig.write_html(path, include_plotlyjs="cdn")
    webbrowser.open(f"file:///{path.replace(os.sep, '/')}")
    print(f"  saved -> {path}")

LINE_W = 1.5   # line width for all traces

# ── Qualitative palettes ───────────────────────────────────────────────────────
# 10-color for cell voltages (colorblind-friendly Tableau-10)
CELL_COLORS = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
    "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
]
# 3-color warm for temperature sensors
TEMP_COLORS  = ["#e15759", "#f28e2b", "#4e79a7"]
# Speed: two blues
SPEED_COLORS = ["#4e79a7", "#f28e2b"]

# ══════════════════════════════════════════════════════════════════════════════
# 1. Pack Voltage
# ══════════════════════════════════════════════════════════════════════════════
print("Plot 1/7 — Pack Voltage")
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=t, y=df["bms/voltage"],
    mode="lines", name="Pack voltage",
    line=dict(color="#4e79a7", width=LINE_W),
))
fig.update_layout(**base_layout(
    "Battery Pack Voltage — Full HPPC Test",
    "Sample (s)", "Voltage (V)"
))
save_and_open(fig, "01_pack_voltage.html")

# ══════════════════════════════════════════════════════════════════════════════
# 2. Current
# ══════════════════════════════════════════════════════════════════════════════
print("Plot 2/7 — Current")
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=t, y=df["bms/current"],
    mode="lines", name="Current",
    line=dict(color="#e15759", width=LINE_W),
))
fig.add_hline(y=0, line=dict(color="#888888", width=1, dash="dash"))
fig.update_layout(**base_layout(
    "Pack Current — Full HPPC Test",
    "Sample (s)", "Current (A)"
))
save_and_open(fig, "02_current.html")

# ══════════════════════════════════════════════════════════════════════════════
# 3. Temperature (3 BMS sensors)
# ══════════════════════════════════════════════════════════════════════════════
print("Plot 3/7 — Temperature")
fig = go.Figure()
for i, color in enumerate(TEMP_COLORS):
    fig.add_trace(go.Scatter(
        x=t, y=df[f"bms/temp_values_{i}"],
        mode="lines", name=f"Sensor {i+1}",
        line=dict(color=color, width=LINE_W),
    ))
fig.update_layout(**base_layout(
    "BMS Temperature Sensors — Full HPPC Test",
    "Sample (s)", "Temperature (°C)"
))
save_and_open(fig, "03_temperature.html")

# ══════════════════════════════════════════════════════════════════════════════
# 4. SOC
# ══════════════════════════════════════════════════════════════════════════════
print("Plot 4/7 — SOC")
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=t, y=df["bms/battery_level"],
    mode="lines", name="SOC",
    line=dict(color="#59a14f", width=LINE_W),
))
fig.update_layout(**base_layout(
    "State of Charge — Full HPPC Test",
    "Sample (s)", "SOC (%)"
))
fig.update_yaxes(range=[0, 105])
save_and_open(fig, "04_soc.html")

# ══════════════════════════════════════════════════════════════════════════════
# 5. Cell Voltages (10 cells)
# ══════════════════════════════════════════════════════════════════════════════
print("Plot 5/7 — Cell Voltages")
fig = go.Figure()
for i, color in enumerate(CELL_COLORS):
    fig.add_trace(go.Scatter(
        x=t, y=df[f"bms/cell_voltages_{i}"],
        mode="lines", name=f"Cell {i+1}",
        line=dict(color=color, width=LINE_W),
    ))
fig.update_layout(**base_layout(
    "Individual Cell Voltages — Full HPPC Test",
    "Sample (s)", "Cell Voltage (V)"
))
save_and_open(fig, "05_cell_voltages.html")

# ══════════════════════════════════════════════════════════════════════════════
# 6. Hoverboard Speed (L + R)
# ══════════════════════════════════════════════════════════════════════════════
print("Plot 6/7 — Hoverboard Speed")
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=t, y=df["hoverboard/hb_speedR_meas"],
    mode="lines", name="Speed R",
    line=dict(color=SPEED_COLORS[0], width=LINE_W),
))
fig.add_trace(go.Scatter(
    x=t, y=df["hoverboard/hb_speedL_meas"],
    mode="lines", name="Speed L",
    line=dict(color=SPEED_COLORS[1], width=LINE_W),
))
fig.update_layout(**base_layout(
    "Hoverboard Motor Speed — Full HPPC Test",
    "Sample (s)", "Speed (RPM)"
))
save_and_open(fig, "06_hb_speed.html")

# ══════════════════════════════════════════════════════════════════════════════
# 7. Hoverboard Board Temperature
# ══════════════════════════════════════════════════════════════════════════════
print("Plot 7/7 — Hoverboard Board Temperature")
# Mask sensor glitch values outside physically plausible range
hb_temp = df["hoverboard/hb_board_temp"].copy()
hb_temp[(hb_temp < 0) | (hb_temp > 60)] = float("nan")
hb_temp = hb_temp.interpolate(method="linear", limit_direction="both")
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=t, y=hb_temp,
    mode="lines", name="Board temp",
    line=dict(color="#b07aa1", width=LINE_W),
))
fig.update_layout(**base_layout(
    "Hoverboard Board Temperature — Full HPPC Test",
    "Sample (s)", "Temperature (°C)"
))
save_and_open(fig, "07_hb_board_temp.html")

print(f"\nAll figures saved to {OUT_DIR}")
