"""
Charge run WITH live BMS streaming to MATLAB.

Combines charge_run.py (BMS-only charge logging to HDF5) and bms_to_matlab.py
(TCP/JSON BMS stream for MATLAB) into a single run: the existing HDF5
data_logger thread runs unchanged, while a second background thread streams
the latest BMS sample to MATLAB over TCP (newline-delimited JSON) at
MATLAB_LOG_HZ, independent of the HDF5 logging rate.

Usage
-----
1. Edit the CONFIGS section below (charge run settings + MATLAB TCP settings).
2. Run from the repo root:
       python -m dataset.run_scripts.charge_run_matlab
3. In MATLAB, run  bms_receiver.m  — it will connect to this server. It can
   connect/disconnect at any point during the run; the server keeps
   listening and resumes streaming on the next connection.
"""
from dataset.dataset_utils import append_row, get_timestamp, get_date_string, get_time_string, init_run_dynamic
from dataset.run_scripts.bms_to_matlab import _BMSEncoder, _sanitize
from drivers.bms_reader import BMSReader
import socket
import json
import threading
import time

######################################## CONFIGS ########################################

# HDF5 file parameters
hdf5_file = "dataset/all_data/h5_files/drive_cycle_data_final.h5"

# run parameters
run_name = "run004_full_charge_after_US06"
run_metadata = {
    "description": "cont. charging the battery to 100% SOC after US06 drive cycle profile discharge.",
    "date": get_date_string(),
    "battery_pack": "Lithium-Ion, 42V, 10.2Ah",
    "battery_age": "new",
    "Logging rate": "1 sample/sec"
}
speed = None                    # constant speed to maintain
stop_soc = 101.0                # stop run when SOC reaches this value
hb_com_port = None              # Hoverboard COM port
hb_baud_rate = None             # Hoverboard baud rate
bms_name = "EGIKE_STATION_1"    # BMS device name
LOG_HZ = 1                      # Logging interval (Hz)
sample_interval = 1.0 / LOG_HZ  # seconds

# --- MATLAB TCP stream ---
TCP_HOST      = "0.0.0.0"   # listen on all interfaces (127.0.0.1 for localhost only)
TCP_PORT      = 5005         # MATLAB connects to this port
MATLAB_LOG_HZ = 1            # samples per second to send to MATLAB

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
# Stop flag for the logging threads
stop_flag = threading.Event()

# store the last known data
last_hb = hb_init_sample
last_bms = bms_init_sample
######################################## FUNCTIONS ########################################
def data_logger(bms_reader):
    global last_hb, last_bms
    hb_dict = last_hb
    while not stop_flag.is_set():
        timestamp = get_timestamp()
        time_string = get_time_string()
        # Get latest BMS sample
        bms_sample = bms_reader.get_latest()
        if bms_sample is not None:
            last_bms = bms_sample
        bms_dict = last_bms  # use last-known if current is None
        print(bms_dict)
        # Append row to HDF5
        append_row(hdf5_file, run_name, timestamp, time_string, hb_dict, bms_dict)

        # Stop run if target SOC is reached
        # if bms_dict and "battery_level" in bms_dict and bms_dict["battery_level"] >= stop_soc:
        #     print(f"Reached stop SOC ({stop_soc}%), stopping run.")
        #     stop_flag.set()
        #     break
        time.sleep(sample_interval)


def matlab_server_thread(bms_reader: BMSReader) -> None:
    """Background thread: stream the latest BMS sample to MATLAB over TCP
    (newline-delimited JSON) until stop_flag is set. Runs independently of
    the HDF5 data_logger thread and its logging rate."""
    interval = 1.0 / MATLAB_LOG_HZ

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((TCP_HOST, TCP_PORT))
        srv.listen(1)
        srv.settimeout(1.0)  # periodic wakeups so stop_flag is checked while waiting for a connection
        print(f"[matlab] TCP server listening on {TCP_HOST}:{TCP_PORT} — waiting for MATLAB…")

        while not stop_flag.is_set():
            try:
                conn, addr = srv.accept()
            except socket.timeout:
                continue

            print(f"[matlab] MATLAB connected from {addr}")
            last_sent = None
            try:
                with conn:
                    while not stop_flag.is_set():
                        t0 = time.monotonic()

                        sample = bms_reader.get_latest()
                        if sample is not None:
                            last_sent = sample

                        if last_sent is not None:
                            payload = _sanitize(last_sent)
                            payload["timestamp"] = time.time()  # Unix seconds
                            line = json.dumps(payload, cls=_BMSEncoder) + "\n"
                            conn.sendall(line.encode("utf-8"))

                        elapsed = time.monotonic() - t0
                        time.sleep(max(0.0, interval - elapsed))
            except (BrokenPipeError, ConnectionResetError, OSError) as exc:
                print(f"[matlab] MATLAB disconnected ({exc}). Waiting for reconnect…")

######################################## MAIN RUN ########################################

# Initialize run in HDF5
init_run_dynamic(hdf5_file, run_name, run_metadata, hb_init_sample, bms_init_sample)

# BMS initialization
bms_reader = BMSReader(device_name=bms_name)
bms_reader.start()
print(f"BMS Reader started for device {bms_name}.")
print("Starting run:", run_name)
print("Run Description:", run_metadata["description"])

# Wait for BMS connection
while bms_reader.get_latest() is None:
    print("Waiting for BMS Bluetooth Connection...")
    time.sleep(1)

# Start logger thread
logger_thread = threading.Thread(target=data_logger, args=(bms_reader,))
logger_thread.start()
print("Simulation will stop when BMS SOC reaches", stop_soc, "%")

# Start MATLAB streaming thread
matlab_thread = threading.Thread(target=matlab_server_thread, args=(bms_reader,), daemon=True)
matlab_thread.start()

try:
    logger_thread.join()
except KeyboardInterrupt:
    print("\nInterrupted. Stopping run...")
finally:
    stop_flag.set()
    logger_thread.join(timeout=2.0)
    bms_reader.stop()
    matlab_thread.join(timeout=2.0)
    print("Cleanup complete.")
