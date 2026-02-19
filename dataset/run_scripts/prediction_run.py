"""
Running hoverboard at a configurable speed to discharge the battery
WITH real-time BMS plotting and SOC prediction using the trained MLP model.
"""

from dataset.dataset_utils import (
    init_run_dynamic, append_row,
    get_timestamp, get_date_string, get_time_string
)
from drivers.hoverboard_controller import HoverboardController
from drivers.bms_reader import BMSReader
from soc_estimation.mlp.mlp import MLP, ModelManager
import threading
import os
import time
from collections import deque
import signal
import numpy as np
import h5py
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------- Qt / Plotting --------
import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer
import pyqtgraph as pg

######################################## CONFIGS ########################################
hdf5_file = "dataset/hoverboard_bms_prediction.h5"
run_type = "discharge" # "discharge" or "charge"
run_name = "run_001_test_prediction"

run_metadata = {
    "description": "Testing prediction accuracy of the MLP model in a discharging scenario.",
    "date": get_date_string(),
    "battery_pack": "Lithium-Ion 10Ah",
    "battery_age": "new",
    "Logging rate": "1 sample/sec",
    "Hoverboard Speed": "N/A",
    "Hoverboard Load": "N/A"
}

FULL_SPEED = 580
speed = int(FULL_SPEED * 0.4)
stop_soc = 40.0

hb_com_port = "COM5"
hb_baud_rate = 115200
bms_name = "EGIKE_STATION_1"

LOG_HZ = 1
sample_interval = 1.0 / LOG_HZ
##################################### LOAD MLP MODEL #####################################
model = MLP(input_size=4, hidden_sizes=[64, 32, 16], output_size=1)
mlp_manager = ModelManager(model, device='cpu')
save_path = r'C:\Users\assas\Desktop\NU\Experimental Setup\ev-bms-data-acquisition\soc_estimation\mlp\outputs'
mlp_manager.load_model_weights(f"{save_path}\\final_best_mlp_model.pth")
mlp_manager.load_scalers(f"{save_path}\\final_scalers.pkl")
mlp_manager.model.eval()

######################################## DATA BUFFERS ########################################

MAX_POINTS = 600  # 10 minutes @ 1 Hz

time_buf = deque(maxlen=MAX_POINTS)
soc_buf = deque(maxlen=MAX_POINTS)
pred_soc_buf = deque(maxlen=MAX_POINTS)
volt_buf = deque(maxlen=MAX_POINTS)
curr_buf = deque(maxlen=MAX_POINTS)
speed_buf = deque(maxlen=MAX_POINTS)
bms_temp1_buf = deque(maxlen=MAX_POINTS)
bms_temp2_buf = deque(maxlen=MAX_POINTS)
bms_temp3_buf = deque(maxlen=MAX_POINTS)
hb_board_temp_buf = deque(maxlen=MAX_POINTS)

# Unbounded — for end-of-run metrics
all_soc = []
all_pred_soc = []

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
        
        if run_type == "discharge":
            hb_feedback = hoverboard.get_feedback()
            if hb_feedback:
                last_hb = hb_feedback

        bms_sample = bms_reader.get_latest()
        if bms_sample:
            last_bms = bms_sample

        # ---- Console ----
        data_for_mlp = [
            last_bms.get("voltage", 0.0),
            last_bms.get("current", 0.0),
            np.mean(last_bms.get("temp_values", [0,0,0])),
            last_bms.get("cycle_charge", 0)
        ]
        predicted_soc = mlp_manager.predict(data_for_mlp)[0] * 100  # Scale back to percentage
        print(last_hb)
        print(last_bms)
        print(f"Predicted SOC: {predicted_soc:.2f}%")

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
        all_soc.append(last_bms.get("battery_level", 0.0))
        pred_soc_buf.append(predicted_soc)
        all_pred_soc.append(predicted_soc) 
        volt_buf.append(last_bms.get("voltage", 0.0))
        curr_buf.append(last_bms.get("current", 0.0))
        # ---- Speed buffer (average L/R) ----
        speed_l = last_hb.get("hb_speedL_meas", 0)
        speed_r = last_hb.get("hb_speedR_meas", 0)
        speed_buf.append((speed_l - speed_r) / 2)
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
if run_type == "discharge":
    hoverboard = HoverboardController(
        serial_port=hb_com_port,
        baud_rate=hb_baud_rate,
        print_feedback=False
    )
    hoverboard.start_threads()
    print(f"Hoverboard started on {hb_com_port}")
else:
    hoverboard = None

bms_reader = BMSReader(device_name=bms_name)
bms_reader.start()
print(f"BMS Reader started for device {bms_name}")

if run_type == "discharge":
    hoverboard.ramp_speed(speed)
    print(f"Hoverboard ramped to speed {speed}")

# ---- Start sinusoidal speed control ----
# sin_stop_event = threading.Event()
# print("Starting sinusoidal speed control...")

# threading.Thread(
#     target=hoverboard.sinusoidal_speed,
#     kwargs=dict(
#         amplitude=speed,       # swings from -400 to +400
#         frequency=0.02,      # one full cycle every ~50s
#         update_interval=0.05,
#         max_speed=FULL_SPEED,
#         stop_event=sin_stop_event
#     ),
#     daemon=True
# ).start()


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
soc_plot = win.addPlot(title="SOC (%) vs Time")
soc_plot.setLabel("left", "SOC", units="%")
soc_plot.showGrid(x=True, y=True)
soc_plot.addLegend()
soc_curve = soc_plot.plot(pen=pg.mkPen("b", width=2), name="Actual SOC")
pred_soc_curve = soc_plot.plot(
    pen=pg.mkPen("r", width=2, style=pg.QtCore.Qt.DashLine),
    name="Predicted SOC"
)
soc_plot.addLine(y=stop_soc, pen=pg.mkPen("r", style=pg.QtCore.Qt.DashLine))

volt_plot = win.addPlot(title="Voltage vs Time")
volt_plot.setLabel("left", "Voltage", units="V")
volt_plot.showGrid(x=True, y=True)
volt_curve = volt_plot.plot(pen=pg.mkPen("g", width=2))

curr_plot = win.addPlot(title="Current (A) vs Time")
curr_plot.setLabel("left", "Current", units="A")
curr_plot.showGrid(x=True, y=True)
curr_curve = curr_plot.plot(pen=pg.mkPen("m", width=2))

speed_plot = win.addPlot(title="Speed (vs Time)")
speed_plot.setLabel("left", "Speed")
speed_plot.setLabel("bottom", "Time", units="s")
speed_plot.showGrid(x=True, y=True)
speed_curve = speed_plot.plot(pen=pg.mkPen("k", width=2))

# ----- Next row -----
win.nextRow()

bms_temp1_plot = win.addPlot(title="BMS Temp 1")
bms_temp1_plot.setLabel("left", "BMS Temp 1 °C")
bms_temp1_plot.showGrid(x=True, y=True)
bms_temp1_curve = bms_temp1_plot.plot(pen=pg.mkPen("r", width=2))

bms_temp2_plot = win.addPlot(title="BMS Temp 2")
bms_temp2_plot.setLabel("left", "BMS Temp 2 °C")
bms_temp2_plot.showGrid(x=True, y=True)
bms_temp2_curve = bms_temp2_plot.plot(pen=pg.mkPen("g", width=2))

bms_temp3_plot = win.addPlot(title="BMS Temp 3")
bms_temp3_plot.setLabel("left", "BMS Temp 3 °C")
bms_temp3_plot.showGrid(x=True, y=True)
bms_temp3_curve = bms_temp3_plot.plot(pen=pg.mkPen("b", width=2))



hb_board_temp_plot = win.addPlot(title="HB Board Temp")
hb_board_temp_plot.setLabel("left", "HB Board Temp °C")
hb_board_temp_plot.showGrid(x=True, y=True)
hb_board_temp_curve = hb_board_temp_plot.plot(pen=pg.mkPen("k", width=2))

win.show()

def update_plot():
    soc_curve.setData(time_buf, soc_buf)
    pred_soc_curve.setData(time_buf, pred_soc_buf) 
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
    print("Shutting down...")
    stop_flag.set()
    #sin_stop_event.set()
    hoverboard.ramp_speed(0)
    hoverboard.close()
    bms_reader.stop()
    actual    = np.array(all_soc)
    predicted = np.array(all_pred_soc)
    if len(actual) > 1 and len(predicted) > 1:
        # Align lengths in case of any race condition
        min_len = min(len(actual), len(predicted))
        actual, predicted = actual[:min_len], predicted[:min_len]

        r2  = r2_score(actual, predicted)
        mse = mean_squared_error(actual, predicted)
        mae = mean_absolute_error(actual, predicted)

        print("\n========== SOC Prediction Metrics ==========")
        print(f"  Samples evaluated : {min_len}")
        print(f"  R²                : {r2:.4f}")
        print(f"  MSE               : {mse:.4f}")
        print(f"  MAE               : {mae:.4f}")
        print("=============================================\n")
        # save soc and predicted soc to csv for further analysis
        os.makedirs(fr"C:\Users\assas\Desktop\NU\Experimental Setup\ev-bms-data-acquisition\dataset\csv_files", exist_ok=True)
        df_results = pd.DataFrame({
            "Actual_SOC": actual,
            "Predicted_SOC": predicted
        })
        df_results.to_csv(fr"C:\Users\assas\Desktop\NU\Experimental Setup\ev-bms-data-acquisition\dataset\csv_files\{run_name}.csv", index=False)
    else:
        print("Not enough data to compute metrics.")

def handle_signal(sig, frame):
    shutdown()
    sys.exit(0)

signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal) 

app.aboutToQuit.connect(shutdown)
sys.exit(app.exec())
