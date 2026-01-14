"""
Running hoverboard at a configurable constant speed to discharge the battery
WITH real-time BMS plotting
"""

from dataset.dataset_utils import (
    init_run_dynamic, append_row,
    get_timestamp, get_date_string, get_time_string
)
from drivers.hoverboard_controller import HoverboardController
from drivers.bms_reader import BMSReader

import threading
import time
from collections import deque

# -------- Qt / Plotting --------
import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer
import pyqtgraph as pg

######################################## CONFIGS ########################################

hdf5_file = "dataset/hoverboard_bms_dataset2_test.h5"

run_name = "run_009_40pct_speed_30kg_load_discharge"
run_metadata = {
    "description": "Running hoverboard at 0.4 full speed with rollers resistance and 30kg load to discharge the battery.",
    "date": get_date_string(),
    "battery_pack": "Lithium-Ion 10Ah",
    "battery_age": "new",
    "Logging rate": "1 sample/sec",
    "Hoverboard Speed": "40% of full speed",
    "Hoverboard Load": "30kg + rollers resistance"
}

FULL_SPEED = 580
speed = int(FULL_SPEED * 0.4)
stop_soc = 40.0

hb_com_port = "COM5"
hb_baud_rate = 115200
bms_name = "EGIKE_STATION_1"

LOG_HZ = 1
sample_interval = 1.0 / LOG_HZ

######################################## DATA BUFFERS ########################################

MAX_POINTS = 600  # 10 minutes @ 1 Hz

time_buf = deque(maxlen=MAX_POINTS)
soc_buf = deque(maxlen=MAX_POINTS)
volt_buf = deque(maxlen=MAX_POINTS)
curr_buf = deque(maxlen=MAX_POINTS)
speed_buf = deque(maxlen=MAX_POINTS)
bms_temp1_buf = deque(maxlen=MAX_POINTS)
bms_temp2_buf = deque(maxlen=MAX_POINTS)
bms_temp3_buf = deque(maxlen=MAX_POINTS)
hb_board_temp_buf = deque(maxlen=MAX_POINTS)

start_time = time.time()

######################################## INIT SAMPLES ########################################

hb_init_sample = {
    "hb_speedR_meas": 0,
    "hb_speedL_meas": 0,
    "hb_measured_voltage": 0.0,
    "hb_board_temp": 0.0
}

bms_init_sample = {
    "battery_charging": False,
    "battery_level": 0.0,
    "voltage": 0.0,
    "current": 0.0,
    "cycle_charge": 0,
    "temp_sensors": 0,
    "temp_values": [0, 0, 0],
    "power": 0.0,
    "cycle_capacity": 0.0,
    "cycles": 0,
    "delta_voltage": 0.0,
    "temperature": 0.0,
    "cell_count": 0,
    "cell_voltages": [0.0] * 10
}

######################################## THREAD STATE ########################################

stop_flag = threading.Event()
last_hb = hb_init_sample
last_bms = bms_init_sample

######################################## LOGGER THREAD ########################################

def data_logger(hoverboard, bms_reader):
    global last_hb, last_bms

    while not stop_flag.is_set():
        timestamp = get_timestamp()
        time_string = get_time_string()

        hb_feedback = hoverboard.get_feedback()
        if hb_feedback:
            last_hb = hb_feedback

        bms_sample = bms_reader.get_latest()
        if bms_sample:
            last_bms = bms_sample

        # ---- Console ----
        print(last_hb)
        print(last_bms)

        # ---- HDF5 ----
        append_row(
            hdf5_file, run_name,
            timestamp, time_string,
            last_hb, last_bms
        )

        # ---- Plot buffers ----
        t = time.time() - start_time
        time_buf.append(t)
        soc_buf.append(last_bms.get("battery_level", 0.0))
        volt_buf.append(last_bms.get("voltage", 0.0))
        curr_buf.append(last_bms.get("current", 0.0))
        # ---- Speed buffer (average L/R) ----
        speed_l = last_hb.get("hb_speedL_meas", 0)
        speed_r = last_hb.get("hb_speedR_meas", 0)
        speed_buf.append((abs(speed_l) + abs(speed_r)) / 2)
        # ---- BMS and hoverboard temps ----
        temp_values = last_bms.get("temp_values", [0,0,0])
        bms_temp1_buf.append(temp_values[0])
        bms_temp2_buf.append(temp_values[1])
        bms_temp3_buf.append(temp_values[2])
        hb_board_temp_buf.append(last_hb.get("hb_board_temp", 0))

        # ---- Stop condition ----
        if last_bms.get("battery_level", 100) <= stop_soc:
            print(f"Reached stop SOC ({stop_soc}%), stopping run.")
            stop_flag.set()
            break

        time.sleep(sample_interval)

######################################## MAIN ########################################

# ---- HDF5 init ----
init_run_dynamic(
    hdf5_file, run_name,
    run_metadata,
    hb_init_sample, bms_init_sample
)

# ---- Hardware init ----
hoverboard = HoverboardController(
    serial_port=hb_com_port,
    baud_rate=hb_baud_rate,
    print_feedback=False
)
hoverboard.start_threads()

bms_reader = BMSReader(device_name=bms_name)
bms_reader.start()

print(f"Hoverboard started on {hb_com_port}")
print(f"BMS Reader started for device {bms_name}")

hoverboard.ramp_speed(speed)
print(f"Hoverboard ramped to speed {speed}")

while bms_reader.get_latest() is None:
    print("Waiting for BMS Bluetooth Connection...")
    time.sleep(1)

# ---- Start logger thread ----
threading.Thread(
    target=data_logger,
    args=(hoverboard, bms_reader),
    daemon=True
).start()

######################################## QT PLOTTING ########################################

app = QApplication(sys.argv)

pg.setConfigOption("background", "w")
pg.setConfigOption("foreground", "k")

win = pg.GraphicsLayoutWidget(title="BMS & Hoverboard Real-Time Monitor")
win.resize(1400, 700)

# ----- Row 1 -----
soc_plot = win.addPlot(title="SOC (%)")
soc_plot.setLabel("left", "SOC", units="%")
soc_plot.showGrid(x=True, y=True)
soc_curve = soc_plot.plot(pen=pg.mkPen("b", width=2))
soc_plot.addLine(y=stop_soc, pen=pg.mkPen("r", style=pg.QtCore.Qt.DashLine))

volt_plot = win.addPlot(title="Voltage (V)")
volt_plot.setLabel("left", "Voltage", units="V")
volt_plot.showGrid(x=True, y=True)
volt_curve = volt_plot.plot(pen=pg.mkPen("g", width=2))

curr_plot = win.addPlot(title="Current (A)")
curr_plot.setLabel("left", "Current", units="A")
curr_plot.showGrid(x=True, y=True)
curr_curve = curr_plot.plot(pen=pg.mkPen("m", width=2))

speed_plot = win.addPlot(title="Speed")
speed_plot.setLabel("left", "Speed")
speed_plot.setLabel("bottom", "Time", units="s")
speed_plot.showGrid(x=True, y=True)
speed_curve = speed_plot.plot(pen=pg.mkPen("k", width=2))

# ----- Next row -----
win.nextRow()

bms_temp1_plot = win.addPlot(title="BMS Temp 1")
bms_temp1_plot.setLabel("left", "°C")
bms_temp1_plot.showGrid(x=True, y=True)
bms_temp1_curve = bms_temp1_plot.plot(pen=pg.mkPen("r", width=2))

bms_temp2_plot = win.addPlot(title="BMS Temp 2")
bms_temp2_plot.setLabel("left", "°C")
bms_temp2_plot.showGrid(x=True, y=True)
bms_temp2_curve = bms_temp2_plot.plot(pen=pg.mkPen("g", width=2))

bms_temp3_plot = win.addPlot(title="BMS Temp 3")
bms_temp3_plot.setLabel("left", "°C")
bms_temp3_plot.showGrid(x=True, y=True)
bms_temp3_curve = bms_temp3_plot.plot(pen=pg.mkPen("b", width=2))

hb_board_temp_plot = win.addPlot(title="HB Board Temp")
hb_board_temp_plot.setLabel("left", "°C")
hb_board_temp_plot.showGrid(x=True, y=True)
hb_board_temp_curve = hb_board_temp_plot.plot(pen=pg.mkPen("k", width=2))

win.show()

def update_plot():
    soc_curve.setData(time_buf, soc_buf)
    volt_curve.setData(time_buf, volt_buf)
    curr_curve.setData(time_buf, curr_buf)
    speed_curve.setData(time_buf, speed_buf)
    bms_temp1_curve.setData(time_buf, bms_temp1_buf)
    bms_temp2_curve.setData(time_buf, bms_temp2_buf)
    bms_temp3_curve.setData(time_buf, bms_temp3_buf)
    hb_board_temp_curve.setData(time_buf, hb_board_temp_buf)

plot_timer = QTimer()
plot_timer.timeout.connect(update_plot)
plot_timer.start(200)  # 5 Hz plot refresh

######################################## CLEAN EXIT ########################################

def shutdown():
    stop_flag.set()
    hoverboard.ramp_speed(0)
    hoverboard.close()
    bms_reader.stop()

app.aboutToQuit.connect(shutdown)

sys.exit(app.exec())
