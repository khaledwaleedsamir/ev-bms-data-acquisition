"""
Smoke test for hoverboard + BMS + HDF5 pipeline.

Purpose:
- Verify hoverboard connection
- Verify BMS connection
- Verify HDF5 run creation
- Verify data logging works end-to-end

This test runs for a short, fixed duration at very low speed.
"""

from dataset.dataset_utils import (
    init_run_dynamic,
    append_row,
    get_timestamp,
    get_date_string,
    get_time_string,
)
from drivers.hoverboard_controller import HoverboardController
from drivers.bms_reader import BMSReader
import threading
import time

######################################## CONFIGS ########################################

hdf5_file = "dataset/hoverboard_bms_smoke_test.h5"

run_name = "smoke_test_hoverboard_bms"
run_metadata = {
    "description": "Smoke test run to verify hoverboard, BMS, and logging pipeline.",
    "date": get_date_string(),
    "test_type": "smoke_test",
    "duration_sec": 60,
    "logging_rate": "1 sample/sec",
}

FULL_SPEED = 580
SMOKE_TEST_SPEED = int(FULL_SPEED * 0.2)  # 20% speed (safe)
LOG_HZ = 1
sample_interval = 1.0 / LOG_HZ
TEST_DURATION_SEC = 60

hb_com_port = "COM5"
hb_baud_rate = 115200
bms_name = "EGIKE_STATION_1"

######################################## INIT SAMPLES ########################################

hb_init_sample = {
    "hb_speedR_meas": 0,
    "hb_speedL_meas": 0,
    "hb_measured_voltage": 0.0,
    "hb_board_temp": 0.0,
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
    "cell_voltages": [0.0] * 10,
}

######################################## GLOBALS ########################################

stop_flag = threading.Event()
last_hb = hb_init_sample
last_bms = bms_init_sample

######################################## FUNCTIONS ########################################

def data_logger(hoverboard, bms_reader):
    global last_hb, last_bms

    start_time = time.time()

    while not stop_flag.is_set():
        timestamp = get_timestamp()
        time_string = get_time_string()

        hb_feedback = hoverboard.get_feedback()
        if hb_feedback is not None:
            last_hb = hb_feedback

        bms_sample = bms_reader.get_latest()
        if bms_sample is not None:
            last_bms = bms_sample

        print("HB:", last_hb)
        print("BMS:", last_bms)

        append_row(
            hdf5_file,
            run_name,
            timestamp,
            time_string,
            last_hb,
            last_bms,
        )

        if time.time() - start_time >= TEST_DURATION_SEC:
            print("Smoke test duration reached. Stopping.")
            stop_flag.set()
            break

        time.sleep(sample_interval)

######################################## MAIN ########################################

print("Initializing smoke test run...")

init_run_dynamic(
    hdf5_file,
    run_name,
    run_metadata,
    hb_init_sample,
    bms_init_sample,
)

hoverboard = HoverboardController(
    serial_port=hb_com_port,
    baud_rate=hb_baud_rate,
    print_feedback=False,
)
hoverboard.start_threads()

bms_reader = BMSReader(device_name=bms_name)
bms_reader.start()

print(f"Hoverboard connected on {hb_com_port}")
print(f"BMS reader started for {bms_name}")

print("Ramping hoverboard to low smoke-test speed...")
hoverboard.ramp_speed(SMOKE_TEST_SPEED)

while bms_reader.get_latest() is None:
    print("Waiting for BMS Bluetooth connection...")
    time.sleep(1)

logger_thread = threading.Thread(
    target=data_logger, args=(hoverboard, bms_reader)
)
logger_thread.start()

logger_thread.join()

print("Stopping hoverboard and BMS...")
hoverboard.close()
bms_reader.stop()

print("Smoke test completed successfully.")
