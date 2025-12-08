from scripts.hoverboard_controller import HoverboardController
from scripts.bms_reader import BMSReader
from dataset.dataset_utils import init_run, append_row
import time

hoverboard = HoverboardController(serial_port="COM5", baud_rate=115200, print_feedback=False)
hoverboard.start_threads()

bms_reader = BMSReader(device_name="EGIKE_STATION_1")
bms_reader.start()

print("Collecting data... Press Ctrl+C to stop.")
print("Starting hoverboard in 3 seconds...")
time.sleep(3)
hoverboard.ramp_speed(500)

try:
    while True:
        hb_feedback = hoverboard.get_feedback()
        bms_sample = bms_reader.get_latest()
        if hb_feedback:
            print("Hoverboard data:", hb_feedback)
        if bms_sample:
            print("BMS data:", repr(bms_sample).replace(", ", ",\n\t"))
        time.sleep(1)
except KeyboardInterrupt:
    print("\nCtrl+C detected! Shutting down gracefully...")
finally:
    # Ramp speed down to 0 before closing
    print("Stopping hoverboard...")
    hoverboard.ramp_speed(0, step=20)
    hoverboard.close()
    print("Stopping BMS reader...")
    bms_reader.stop()
    print("Shutdown complete.")

