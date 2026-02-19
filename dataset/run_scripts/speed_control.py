"""
Running hoverboard with live keyboard speed control
Console logging only (no HDF5)
"""

from drivers.hoverboard_controller import HoverboardController
from drivers.bms_reader import BMSReader
from dataset.dataset_utils import get_timestamp, get_time_string
import threading
import time
import msvcrt 

######################################## CONFIGS ########################################

FULL_SPEED = 580                # hoverboard max speed
hb_com_port = "COM5"
hb_baud_rate = 115200
bms_name = "EGIKE_STATION_1"
LOG_HZ = 1
sample_interval = 1.0 / LOG_HZ
stop_soc = 40.0

######################################## STATE ########################################

stop_flag = threading.Event()
speed_lock = threading.Lock()
current_speed_pct = 20  # start at 20%

last_hb = {}
last_bms = {}

######################################## FUNCTIONS ########################################

def pct_to_speed(pct):
    return int((pct / 100.0) * FULL_SPEED)

def keyboard_control(hoverboard):
    global current_speed_pct

    print("\nKeyboard Controls:")
    print("  w = increase speed")
    print("  s = decrease speed")
    print("  0–9 = set speed (0–90%)")
    print("  x = stop hoverboard")
    print("  q = quit program\n")

    while not stop_flag.is_set():
        if msvcrt.kbhit():
            key = msvcrt.getch().decode("utf-8").lower()

            with speed_lock:
                if key == "w":
                    current_speed_pct = min(100, current_speed_pct + 5)
                elif key == "s":
                    current_speed_pct = max(0, current_speed_pct - 5)
                elif key.isdigit():
                    current_speed_pct = int(key) * 10
                elif key == "x":
                    current_speed_pct = 0
                elif key == "q":
                    print("Quit command received.")
                    stop_flag.set()
                    break

                speed_value = pct_to_speed(current_speed_pct)
                hoverboard.ramp_speed(speed_value)

                print(f"[SPEED] {current_speed_pct}% → {speed_value}")

        time.sleep(0.05)

def data_logger(hoverboard, bms_reader):
    global last_hb, last_bms

    while not stop_flag.is_set():
        timestamp = get_timestamp()
        time_str = get_time_string()

        hb_feedback = hoverboard.get_feedback()
        if hb_feedback:
            last_hb = hb_feedback

        bms_sample = bms_reader.get_latest()
        if bms_sample:
            last_bms = bms_sample

        print("\n-------------------------------")
        print(f"Time: {time_str}")
        print("Hoverboard:", last_hb)
        print("BMS:", last_bms)

        if last_bms and last_bms.get("battery_level", 100) <= stop_soc:
            print(f"\nSOC reached {stop_soc}%. Stopping run.")
            stop_flag.set()
            break

        time.sleep(sample_interval)

######################################## MAIN ########################################

hoverboard = HoverboardController(
    serial_port=hb_com_port,
    baud_rate=hb_baud_rate,
    print_feedback=False
)
hoverboard.start_threads()

bms_reader = BMSReader(device_name=bms_name)
bms_reader.start()

print(f"Hoverboard started on {hb_com_port}")
print(f"BMS reader started: {bms_name}")

# Wait for BMS
while bms_reader.get_latest() is None:
    print("Waiting for BMS connection...")
    time.sleep(1)

# Set initial speed
hoverboard.ramp_speed(pct_to_speed(current_speed_pct))
print(f"Initial speed set to {current_speed_pct}%")

# Start threads
threading.Thread(target=data_logger, args=(hoverboard, bms_reader), daemon=True).start()
threading.Thread(target=keyboard_control, args=(hoverboard,), daemon=True).start()

# Main wait loop
try:
    while not stop_flag.is_set():
        time.sleep(0.5)
except KeyboardInterrupt:
    stop_flag.set()

print("Stopping hoverboard...")
hoverboard.ramp_speed(0)
hoverboard.close()
bms_reader.stop()
print("Shutdown complete.")
