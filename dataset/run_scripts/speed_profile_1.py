"""
Running hoverboard at a configurable speed profile to discharge the battery
WITH real-time BMS plotting and SOC prediction using the trained MLP model.
"""
from dataset.dataset_utils  import (
    init_run_dynamic, append_row,
    get_timestamp, get_date_string, get_time_string
)
from drivers.hoverboard_controller import HoverboardController
from drivers.bms_reader import BMSReader
from soc_estimation.mlp.mlp import MLP_SOC, ModelManager
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

# -------- Speed profile generation --------
def run_speed_profile(hb: HoverboardController, speed_vector, hold_time=5.0, stop_event=None):
    """
    Runs a speed profile in a separate thread using blocking ramp_speed.
    Each speed is ramped and then held for the remaining time.
    """
    for target_speed in speed_vector:
        if stop_event and stop_event.is_set():
            break
        start_time = time.time()
        # Ramp to the target speed
        print(f"Ramping to speed {target_speed}...")
        hb.ramp_speed(target_speed)
        elapsed_time = time.time() - start_time
        
        # Hold the speed for the remaining time
        time.sleep(max(0, hold_time - elapsed_time))

speed_vector = [31, 399, 86, 481, 543, 572, 112, 77, 40, 259, 243, 368, 172, 291, 247, 298, 438, 554, 
                406, 445, 529, 322, 365, 461, 492, 263, 46, 134, 514, 110, 322, 180, 500, 326, 260, 10, 
                79, 119, 138, 264, 488, 438, 428, 219, 77, 376, 567, 64, 281, 200, 536, 424, 138, 399, 
                70, 89, 25, 161, 536, 88, 289, 217, 490, 319, 563, 103, 109, 33, 220, 183, 79, 552, 209,
                125, 572, 513, 208, 471, 392, 269, 171, 172, 544, 512, 516, 160, 524, 212, 31, 395, 183, 
                120, 72, 571, 76, 178, 545, 388, 529, 443, 128, 286, 435, 310, 302, 250, 487, 266, 409, 
                423, 562, 214, 68, 322, 277, 257, 339, 552, 40, 324, 227, 80, 384, 125, 402, 472, 154, 
                470, 99, 96, 528, 40, 74, 276, 68, 319, 430, 574, 254, 379, 175, 5, 301, 326, 186, 411, 
                123, 284, 137, 224, 477, 570, 292, 493, 523, 149, 216, 495, 138, 211, 554, 220, 567, 123,
                293, 365, 401, 208, 566, 119, 120, 427, 12, 262, 183, 247, 215, 404, 547, 368, 11, 483, 
                135, 398, 105, 484, 74, 365, 81, 524, 335, 566, 346, 279, 272, 228, 214, 151, 549, 511, 
                216, 28, 75, 321, 152, 533, 8, 495, 171, 376, 523, 62, 364, 387, 324, 309, 457, 235, 384,
                430, 166, 251, 383, 219, 563, 170, 559, 420, 141, 511, 432, 226, 475, 246, 228, 434, 102, 
                263, 448, 398, 409, 108, 450, 438, 553, 94, 301, 402, 114, 366, 215, 565, 126, 159, 449, 
                482, 320, 376, 468, 251, 61, 48, 275, 558, 432, 372, 347, 267, 184, 538, 430, 424, 229, 
                442, 110, 32, 434, 515, 346, 374, 425, 469, 136, 475, 486, 502, 385, 567, 430, 315, 273, 
                326, 383, 539, 20, 519, 398, 201, 0, 300, 198, 82, 149, 348, 358, 138, 357, 522, 73, 284, 
                502, 41, 326, 21, 501, 408, 451, 49, 307, 445, 492, 45, 486, 455, 83, 535, 421, 41, 490, 
                504, 29, 529, 39, 580, 104, 195, 550, 493, 519, 113, 271, 151, 255, 11, 403, 473, 204, 
                518, 266, 24, 575, 21, 425, 311, 481, 168, 391, 263, 139, 425]

######################################## CONFIGS ########################################
hdf5_file = "dataset/hoverboard_bms_prediction.h5"
run_type = "discharge" # "discharge" or "charge"
run_name = "run_003_speed_profile_1"

run_metadata = {
    "description": "Testing SOC prediction accuracy of the MLP model in discharge on a speed profile.",
    "date": get_date_string(),
    "battery_pack": "Lithium-Ion 10Ah",
    "battery_age": "new",
    "Logging rate": "1 sample/sec",
    "Hoverboard Speed": "N/A",
    "Hoverboard Load": "N/A"
}

FULL_SPEED = 580
speed = int(FULL_SPEED * 0.8)
stop_soc = 40.0

hb_com_port = "COM5"
hb_baud_rate = 115200
bms_name = "EGIKE_STATION_1"

LOG_HZ = 1
sample_interval = 1.0 / LOG_HZ
##################################### LOAD MLP MODEL #####################################
model = MLP_SOC(input_size=4, hidden_sizes=[32, 16], output_size=1)
mlp_manager = ModelManager(model, device='cpu')
save_path = r'C:\Users\assas\Desktop\NU\Experimental Setup\ev-bms-data-acquisition\soc_estimation\mlp\outputs'
mlp_manager.load_model_weights(f"{save_path}\\mlp_model.pth")
mlp_manager.load_scalers(f"{save_path}\\scalers.pkl")
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

        # ---- prediction ----
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
    print("Starting hoverboard speed profile thread...")
    threading.Thread(
        target=run_speed_profile,
        args=(hoverboard, speed_vector, 5.0, stop_flag),
        daemon=True
    ).start()

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
    if run_type == "discharge":
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
