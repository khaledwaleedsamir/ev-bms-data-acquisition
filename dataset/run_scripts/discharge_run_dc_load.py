"""
Docstring for dataset.run_scripts.discharge_run_dc_load.py
Description: Constant-current discharge of the battery using the UTL8211+ DC
electronic load (drivers/utl8211.py) instead of the hoverboard motor.
"""
from dataset.dataset_utils import init_run_dynamic, append_row, get_timestamp, get_date_string, get_time_string
from drivers.bms_reader import BMSReader
from drivers.utl8211 import UTL8211, Mode
import threading
import time

######################################## CONFIGS ########################################

# HDF5 file parameters
hdf5_file = "dataset/all_data/h5_files/dc_load_bms_dataset_2ndlife.h5"

# run parameters
run_name = "run_001_5A_cc_discharge"
run_metadata = {
    "description": "Constant-current discharge at 5A on the UTL8211 DC electronic load.",
    "date": get_date_string(),
    "battery_pack": "Lithium-Ion, 42v, Nominal Capacity 10.5Ah, Current Capacity: Unknown",
    "battery_age": "2nd-life pack",
    "Logging rate": "1 sample/sec",
    "Load Mode": "Constant Current (CC)",
    "Load Current": "5A"
}

load_com_port  = "COM9"         # DC load COM port
load_baud_rate = 9600           # DC load baud rate
bms_name = "EGIKE_STATION_1"    # BMS device name

discharge_current = 5.0                    # A, constant CC-mode setpoint
current_protection_margin = 1.1            # OCP = margin * discharge_current

stop_soc = 20.0                 # stop run when SOC reaches this value
min_voltage_cutoff = 30.0       # V, abort run if pack voltage drops below this

LOG_HZ = 1                      # Logging interval (Hz)
sample_interval = 1.0 / LOG_HZ  # seconds

######################################## END OF CONFIGS ########################################

# Load and BMS samples for initialization (dummy values)
load_init_sample = {
    "load_target_current": 0.0,
    "load_meas_voltage":   0.0,
    "load_meas_current":   0.0,
    "load_meas_power":     0.0,
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
last_bms = bms_init_sample

######################################## FUNCTIONS ########################################
def data_logger(load, bms_reader):
    global last_bms
    while not stop_flag.is_set():
        timestamp = get_timestamp()
        time_string = get_time_string()

        # Get latest DC load measurements
        meas = load.measure_all()
        load_dict = {
            "load_target_current": discharge_current,
            "load_meas_voltage":   meas["voltage"],
            "load_meas_current":   meas["current"],
            "load_meas_power":     meas["power"],
        }

        # Get latest BMS sample
        bms_sample = bms_reader.get_latest()
        if bms_sample is not None:
            last_bms = bms_sample
        bms_dict = last_bms  # use last-known if current is None

        print(load_dict)
        print(bms_dict)

        # Append row to HDF5
        append_row(hdf5_file, run_name, timestamp, time_string, load_dict, bms_dict)

        # Stop run if SOC <= stop_soc
        if bms_dict and "battery_level" in bms_dict and bms_dict["battery_level"] <= stop_soc:
            print(f"Reached stop SOC ({stop_soc}%), stopping run.")
            stop_flag.set()
            break

        # Stop run if pack voltage <= min_voltage_cutoff
        if bms_dict and "voltage" in bms_dict and bms_dict["voltage"] <= min_voltage_cutoff:
            print(f"Reached min voltage cutoff ({min_voltage_cutoff}V), stopping run.")
            stop_flag.set()
            break

        time.sleep(sample_interval)

######################################## MAIN RUN ########################################

# Initialize run in HDF5
init_run_dynamic(hdf5_file, run_name, run_metadata, load_init_sample, bms_init_sample)

# DC load and BMS initialization
load = UTL8211(port=load_com_port, baudrate=load_baud_rate)
print(f"Load connected: {load.idn()}")

bms_reader = BMSReader(device_name=bms_name)
bms_reader.start()

# Print run info
print(f"DC load started on {load_com_port} at {load_baud_rate} baud.")
print(f"BMS Reader started for device {bms_name}.")
print("Starting run:", run_name)
print("Run Description:", run_metadata["description"])

# Wait for BMS connection
while bms_reader.get_latest() is None:
    print("Waiting for BMS Bluetooth Connection...")
    time.sleep(1)

# Configure and start the DC load in CC mode
load.set_mode(Mode.CURRENT)
load.set_current(0.0)
load.set_current_protection(discharge_current * current_protection_margin)
load.check_errors()
load.set_current(discharge_current)
load.input_on(True)
print(f"Load set to {discharge_current}A CC mode, input on.")

try:
    # Start logger thread
    logger_thread = threading.Thread(target=data_logger, args=(load, bms_reader))
    logger_thread.start()
    print("Run will stop when BMS SOC reaches", stop_soc, "%")

    logger_thread.join()
except KeyboardInterrupt:
    print("\nInterrupted. Stopping run...")
    stop_flag.set()
finally:
    load.set_current(0.0)
    load.input_on(False)
    load.close()
    bms_reader.stop()
    print("Cleanup complete.")
