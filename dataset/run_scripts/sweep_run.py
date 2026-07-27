"""
Speed/SOC efficiency sweep for Model 2's calibration table -- collects
Wh/km data across multiple speeds at multiple SOC checkpoints in ONE
continuous discharge, for the 0kg + rollers_resistance condition (no added
load yet -- see docs/range_controller.md's data-coverage table, this
condition currently has exactly one measured run, at 80% speed only).

Protocol (agreed before writing this script):
  - 9 SOC checkpoints, 100% down to 20% in 10-point steps.
  - At each checkpoint, a CALIBRATION BLOCK: every speed in
    CALIBRATION_SPEEDS_PCT held for DWELL_S seconds, in order.
  - Between checkpoints, TRANSIT at TRANSIT_SPEED_PCT (the fastest
    candidate) until the next checkpoint's target SOC is reached, then the
    next calibration block starts. Transit speed is the fastest candidate
    because the ENERGY needed to cross a given SOC gap is fixed by the pack
    capacity regardless of speed (SOC is literally a measure of energy
    remaining) -- so higher power during transit only buys back wall-clock
    time, it doesn't change how much of the pack gets used.
  - Rest before the first checkpoint and after the last, for a clean
    rest-OCV anchor at both ends -- same requirement soc_relabel.py's
    dual-anchor coulomb counting already relies on for every existing run.

Live SOC (for driving checkpoint/transit transitions and the safety floor)
comes straight from the BMS's own reported battery_level. The actual data
LABEL used to build the efficiency table is soc_relabel.py's offline
OCV-anchored coulomb count (anchored on the rest periods at both ends of
this run), independent of whatever triggers phase transitions live -- so the
live value only needs to be a coarse "have we crossed this checkpoint yet"
signal. This also matches deployment, where Model 1 (the SOC-estimation MLP)
supplies real-time SOC, not a coulomb counter.

Unlike controller_run.py, there is no optimizer here -- this drives a fixed,
deterministic schedule and logs. Speed changes always go through
HoverboardController.ramp_speed(), never a direct jump, per the project's
"must ramp" decision (docs/range_controller.md).

Run more than once, on separate charge cycles, before trusting the
resulting table cells -- this condition currently has zero repeat-run
history, so a single sweep can't distinguish "true value" from "one day's
fluke." Rotate CALIBRATION_SPEEDS_PCT's order between repeats (e.g. reverse
it) so no one speed is systematically last-in-block (and therefore
measured at a slightly lower true SOC than the others) every time.

KNOWN GAP: build_efficiency_dataset.py's STEADY_SPEED_RUNS currently assumes
one fixed speed for an entire run. This script logs multiple speed segments
inside a single HDF5 run (commanded_speed_pct is logged per row, so the
segments are fully recoverable) -- but the ingestion side needs a small
update to split by that per-row logged speed before this run's data can
feed into efficiency_summary.csv. Not yet done.

Run from the repo root: python -m dataset.run_scripts.sweep_run
"""
import signal
import sys
import threading
import time

from dataset.dataset_utils import (
    append_row,
    get_date_string,
    get_time_string,
    get_timestamp,
    init_run_dynamic,
)
from drivers.bms_reader import BMSReader
from drivers.hoverboard_controller import HoverboardController
from range_controller.constants import FULL_SPEED
from soc_estimation.soc_relabel import DROPOUT_VOLTAGE

######################################## CONFIGS ########################################
CONDITION_LOAD_KG = 0
CONDITION_ROLLERS_RESISTANCE = True

SOC_CHECKPOINTS = [100, 90, 80, 70, 60, 50, 40, 30, 20]   # visited in this order
CALIBRATION_SPEEDS_PCT = [20, 40, 60, 80, 100]            # order used within each block -- rotate on repeat sweeps
DWELL_S = 3 * 60                                          # per-speed dwell inside a calibration block
TRANSIT_SPEED_PCT = 100                                   # fastest candidate -- minimizes wall-clock time between checkpoints

REST_BEFORE_S = 20 * 60      # clean high-SOC OCV anchor
REST_AFTER_S = 20 * 60       # clean low-SOC OCV anchor
SAFETY_FLOOR_SOC_PCT = 15.0  # hardware interlock -- stops regardless of checkpoint schedule if tripped

LOG_HZ = 1
LOG_INTERVAL_S = 1.0 / LOG_HZ
TRANSIT_POLL_S = 2.0

hdf5_file = "dataset/all_data/h5_files/range_controller_validation.h5"
run_name = f"sweep_0kg_rollers_{get_date_string()}"

hb_com_port = "COM3"
hb_baud_rate = 115200
bms_name = "EGIKE_STATION_1"

run_metadata = {
    "description": "Speed/SOC efficiency sweep -- 0kg load + rollers resistance, "
                    f"checkpoints={SOC_CHECKPOINTS}, speeds={CALIBRATION_SPEEDS_PCT}, "
                    f"dwell={DWELL_S}s, transit_speed={TRANSIT_SPEED_PCT}%.",
    "date": get_date_string(),
    "run_type": "speed_soc_sweep",
    "condition_load_kg": CONDITION_LOAD_KG,
    "condition_rollers_resistance": str(CONDITION_ROLLERS_RESISTANCE),
    "soc_checkpoints": str(SOC_CHECKPOINTS),
    "calibration_speeds_pct": str(CALIBRATION_SPEEDS_PCT),
    "dwell_s": DWELL_S,
    "transit_speed_pct": TRANSIT_SPEED_PCT,
}
###########################################################################################


def pct_to_raw_speed(pct):
    return int(pct / 100.0 * FULL_SPEED)


def live_soc(bms_reader):
    """BMS-reported battery_level -- see module docstring for why this is
    fine here (only drives coarse phase transitions, never the final data
    label) even though it's the quantized/drifting value the rest of this
    project deliberately avoids for anything accuracy-sensitive."""
    sample = bms_reader.get_latest()
    if sample is None:
        return None
    return sample.get("battery_level")


######################################## STATE ########################################

stop_flag = threading.Event()
state_lock = threading.Lock()

current_commanded_speed_pct = 0
current_phase = "rest"   # "rest" | "transit" | "calibration"
samples_logged = 0
safety_tripped = False


def set_phase(hoverboard, speed_pct, phase):
    global current_commanded_speed_pct, current_phase
    with state_lock:
        current_commanded_speed_pct = speed_pct
        current_phase = phase
    hoverboard.ramp_speed(pct_to_raw_speed(speed_pct))


def rest_hold_median_sample(bms_reader, duration_s, label):
    print(f"Resting {duration_s / 60:.0f} min for a clean OCV anchor ({label})...")
    voltages, currents = [], []
    t_end = time.time() + duration_s
    while time.time() < t_end:
        s = bms_reader.get_latest()
        if s and s.get("voltage") is not None:
            voltages.append(s["voltage"])
            currents.append(s.get("current", 0.0) or 0.0)
        time.sleep(1.0)
    voltages.sort()
    currents.sort()
    return voltages[len(voltages) // 2], currents[len(currents) // 2]


######################################## CONTROL (main thread) ########################################

def hold_speed(hoverboard, bms_reader, speed_pct, duration_s, phase_label):
    global safety_tripped

    set_phase(hoverboard, speed_pct, phase_label)

    elapsed = 0.0
    while elapsed < duration_s and not stop_flag.is_set():
        time.sleep(0.5)
        elapsed += 0.5
        soc = live_soc(bms_reader)
        if soc is not None and soc <= SAFETY_FLOOR_SOC_PCT:
            print(f"SAFETY STOP: live SOC {soc:.1f}% <= floor {SAFETY_FLOOR_SOC_PCT}%.")
            safety_tripped = True
            stop_flag.set()
            return
        bms_sample = bms_reader.get_latest()
        if bms_sample and bms_sample.get("voltage") is not None and bms_sample["voltage"] <= DROPOUT_VOLTAGE:
            print("BMS dropout detected (voltage below threshold) -- stopping for safety.")
            safety_tripped = True
            stop_flag.set()
            return


def transit_to_checkpoint(hoverboard, bms_reader, target_soc_pct):
    global safety_tripped

    set_phase(hoverboard, TRANSIT_SPEED_PCT, "transit")
    print(f"Transit at {TRANSIT_SPEED_PCT}% until SOC <= {target_soc_pct}%...")

    while not stop_flag.is_set():
        soc = live_soc(bms_reader)
        if soc is not None and soc <= target_soc_pct:
            return
        if soc is not None and soc <= SAFETY_FLOOR_SOC_PCT:
            print(f"SAFETY STOP: live SOC {soc:.1f}% <= floor {SAFETY_FLOOR_SOC_PCT}%.")
            safety_tripped = True
            stop_flag.set()
            return
        bms_sample = bms_reader.get_latest()
        if bms_sample and bms_sample.get("voltage") is not None and bms_sample["voltage"] <= DROPOUT_VOLTAGE:
            print("BMS dropout detected (voltage below threshold) -- stopping for safety.")
            safety_tripped = True
            stop_flag.set()
            return
        time.sleep(TRANSIT_POLL_S)


def run_sweep(hoverboard, bms_reader):
    for i, checkpoint in enumerate(SOC_CHECKPOINTS):
        if stop_flag.is_set():
            return
        if i > 0:
            transit_to_checkpoint(hoverboard, bms_reader, checkpoint)
            if stop_flag.is_set():
                return

        soc_now = live_soc(bms_reader)
        soc_str = f"{soc_now:.1f}%" if soc_now is not None else "unknown"
        print(f"\n=== Checkpoint {checkpoint}% (BMS-reported SOC {soc_str}) -- calibration block ===")
        for speed in CALIBRATION_SPEEDS_PCT:
            if stop_flag.is_set():
                return
            print(f"  dwelling {speed}% for {DWELL_S / 60:.0f} min...")
            hold_speed(hoverboard, bms_reader, speed, DWELL_S, f"calibration_{checkpoint}pct_{speed}pct")


######################################## LOGGER THREAD ########################################

def data_logger(hoverboard, bms_reader):
    global samples_logged

    last_hb = {"hb_speedR_meas": 0, "hb_speedL_meas": 0, "hb_measured_voltage": 0.0, "hb_board_temp": 0.0}
    last_bms = None

    while not stop_flag.is_set():
        timestamp = get_timestamp()
        time_string = get_time_string()
        now = time.time()

        hb_feedback = hoverboard.get_feedback()
        if hb_feedback:
            last_hb = hb_feedback
        bms_sample = bms_reader.get_latest()
        if bms_sample:
            last_bms = bms_sample

        if last_bms is None:
            time.sleep(LOG_INTERVAL_S)
            continue

        with state_lock:
            commanded_speed = current_commanded_speed_pct
            phase = current_phase

        bms_for_log = dict(last_bms)
        bms_for_log["commanded_speed_pct"] = float(commanded_speed)
        bms_for_log["phase"] = phase

        append_row(hdf5_file, run_name, timestamp, time_string, last_hb, bms_for_log)
        samples_logged += 1

        if samples_logged % 10 == 0:
            soc = last_bms.get("battery_level")
            soc_str = f"{soc:.1f}%" if soc is not None else "unknown"
            print(f"[LOG] t={samples_logged}s  phase={phase}  speed={commanded_speed}%  SOC={soc_str}")

        elapsed = time.time() - now
        time.sleep(max(0.0, LOG_INTERVAL_S - elapsed))


######################################## MAIN ########################################

def main():
    print(f"Sweep: 0kg + rollers_resistance | checkpoints {SOC_CHECKPOINTS} | speeds {CALIBRATION_SPEEDS_PCT}")

    bms_reader = BMSReader(device_name=bms_name)
    bms_reader.start()
    print(f"Connecting to BMS '{bms_name}'...")
    while bms_reader.get_latest() is None:
        time.sleep(1)
    print("BMS connected.")

    hoverboard = HoverboardController(serial_port=hb_com_port, baud_rate=hb_baud_rate, print_feedback=False)
    hoverboard.start_threads()
    print(f"Hoverboard connected on {hb_com_port}.")
    while hoverboard.get_feedback() is None:
        time.sleep(0.5)

    # ---- Rest before start: clean high-SOC OCV anchor for soc_relabel.py's
    # offline labeling. rest_v/rest_i are printed only as a sanity check
    # (confirms the pack actually settled -- near-zero current, stable
    # voltage); nothing here is used to derive a live SOC value. ----
    rest_v, rest_i = rest_hold_median_sample(bms_reader, REST_BEFORE_S, "start")
    start_soc = live_soc(bms_reader)
    start_soc_str = f"{start_soc:.1f}%" if start_soc is not None else "unknown"
    print(f"Rest reading: V={rest_v:.2f}, I={rest_i:.2f} -- BMS-reported starting SOC = {start_soc_str}")

    # ---- HDF5 init ----
    first_bms = bms_reader.get_latest()
    hb_init_sample = hoverboard.get_feedback()
    bms_init_sample = dict(first_bms)
    bms_init_sample["commanded_speed_pct"] = 0.0
    bms_init_sample["phase"] = "rest"
    init_run_dynamic(hdf5_file, run_name, run_metadata, hb_init_sample, bms_init_sample)

    # ---- Start logger thread ----
    logger_thread = threading.Thread(target=data_logger, args=(hoverboard, bms_reader), daemon=True)
    logger_thread.start()

    def shutdown(*_args):
        print("\nShutting down...")
        stop_flag.set()
        time.sleep(0.5)
        hoverboard.ramp_speed(0)
        hoverboard.close()
        final_soc = live_soc(bms_reader)
        final_soc_str = f"{final_soc:.1f}%" if final_soc is not None else "unknown"
        print("\n========== SWEEP SUMMARY ==========")
        print(f"  Samples logged    : {samples_logged}")
        print(f"  Final BMS SOC     : {final_soc_str} (started {start_soc_str})")
        print(f"  Safety tripped    : {safety_tripped}")
        print("====================================\n")
        bms_reader.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # ---- Run the checkpoint/transit state machine (blocking, main thread) ----
    run_sweep(hoverboard, bms_reader)

    # ---- Rest after end: clean low-SOC OCV anchor (still logging) ----
    if not stop_flag.is_set():
        set_phase(hoverboard, 0, "rest")
        print(f"\nSweep complete. Resting {REST_AFTER_S / 60:.0f} min for the closing OCV anchor...")
        t_end = time.time() + REST_AFTER_S
        while time.time() < t_end and not stop_flag.is_set():
            time.sleep(1.0)

    shutdown()


if __name__ == "__main__":
    main()
