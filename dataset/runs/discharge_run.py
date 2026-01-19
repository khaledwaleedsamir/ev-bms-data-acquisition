"""
Docstring for dataset.runs.discharge_run.py
Description: Running hoverboard at a configurable constant speed to discharge the battery.
"""
from dataset.dataset_utils import init_run, append_row, get_timestamp, get_date_string, get_time_string, init_run_dynamic
from drivers.hoverboard_controller import HoverboardController
from drivers.bms_reader import BMSReader
import threading
import time

######################################## CONFIGS ########################################

# HDF5 file parameters
hdf5_file = "dataset/hoverboard_bms_dataset2.h5"

# run parameters
run_name = "run_013_80pct_speed_25kg_load_discharge"
run_metadata = {
    "description": "Running hoverboard at 0.8 full speed with rollers resistance and 25kg load to discharge the battery.",
    "date": get_date_string(),
    "battery_pack": "Lithium-Ion 10Ah",
    "battery_age": "new",
    "Logging rate": "1 sample/sec",
    "Hoverboard Speed": "80% of full speed",
    "Hoverboard Load": "25kg + rollers resistance"
}
FULL_SPEED = 580                # full speed value for hoverboard
speed = int(FULL_SPEED*0.8)     # constant speed to maintain
stop_soc = 30.0                 # stop run when SOC reaches this value
hb_com_port = "COM5"            # Hoverboard COM port
hb_baud_rate = 115200           # Hoverboard baud rate
bms_name = "EGIKE_STATION_1"    # BMS device name
LOG_HZ = 1                      # Logging interval (Hz)
sample_interval = 1.0 / LOG_HZ  # seconds

######################################## END OF CONFIGS ########################################

# BMS and hoverboard samples for initialization (dummy values)
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
    "cell_voltages": [0.0]*10
}
# Stop flag for the logging thread
stop_flag = threading.Event()

# store the last known data
last_hb = hb_init_sample
last_bms = bms_init_sample
######################################## FUNCTIONS ########################################
def data_logger(hoverboard, bms_reader):
    global last_hb, last_bms
    while not stop_flag.is_set():
        timestamp = get_timestamp()
        time_string = get_time_string()

        # Get latest hoverboard feedback
        hb_feedback = hoverboard.get_feedback()
        if hb_feedback is not None:
            last_hb = hb_feedback
        hb_dict = last_hb  # use last-known if current is None

        # Get latest BMS sample
        bms_sample = bms_reader.get_latest()
        if bms_sample is not None:
            last_bms = bms_sample
        bms_dict = last_bms  # use last-known if current is None

        print(hb_dict)
        print(bms_dict)

        # Append row to HDF5
        append_row(hdf5_file, run_name, timestamp, time_string, hb_dict, bms_dict)

        # Stop run if SOC <= stop_soc
        if bms_dict and "battery_level" in bms_dict and bms_dict["battery_level"] <= stop_soc:
            print(f"Reached stop SOC ({stop_soc}%), stopping run.")
            stop_flag.set()
            break

        time.sleep(sample_interval)

######################################## MAIN RUN ########################################

# Initialize run in HDF5
init_run_dynamic(hdf5_file, run_name, run_metadata, hb_init_sample, bms_init_sample)

# Hoverboard and BMS initialization
hoverboard = HoverboardController(serial_port=hb_com_port, baud_rate=hb_baud_rate, print_feedback=False)
hoverboard.start_threads()
bms_reader = BMSReader(device_name=bms_name)
bms_reader.start()

# Print run info
print(f"Hoverboard started on {hb_com_port} at {hb_baud_rate} baud.")
print(f"BMS Reader started for device {bms_name}.")
print("Starting run:", run_name)
print("Run Description:", run_metadata["description"])


# Ramp hoverboard to target speed
hoverboard.ramp_speed(speed)
print("Starting Run...")
print(f"Hoverboard ramped to speed {speed}.")

# Wait for BMS connection
while bms_reader.get_latest() is None:
    print("Waiting for BMS Bluetooth Connection...")
    time.sleep(1)

# Start logger thread
logger_thread = threading.Thread(target=data_logger,args=(hoverboard, bms_reader))
logger_thread.start()
print("Simulation will stop when BMS SOC reaches", stop_soc, "%")

# # stop after 2 min (for testing)
# time.sleep(120)
# stop_flag.set()
# hoverboard.close()
# bms_reader.stop()