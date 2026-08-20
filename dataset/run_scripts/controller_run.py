"""
Real closed-loop validation run for Model 2's range controller -- Layer 3b
from docs/range_controller.md's validation plan. Runs EITHER a fixed
baseline speed OR the live optimizer, for a fixed target distance, so a
baseline run and a controller run can be compared directly afterward:
energy used, time taken, and final SOC to cover the same distance.

Two things this project didn't have before, built here:

  - Live distance tracking: RPM -> velocity -> integrated distance, same
    method as build_efficiency_dataset.py, run online instead of post-hoc.

  - Live SOC tracking: a single-anchor coulomb counter. soc_relabel.py's
    compute_soc_cc() is inherently offline -- it anchors on OCV at BOTH the
    start and the end of a run and fits between them. There is no "end" yet
    on a live run, so LiveSocTracker anchors once on a rest-OCV reading taken
    before the run starts, then integrates current forward from there with
    no end correction. Less accurate than the offline label over a long run,
    but it is the only thing that can run in real time.

Speed changes always go through HoverboardController.ramp_speed() -- never a
direct jump -- in both modes, per the "must ramp" decision in
docs/range_controller.md (rider balance/safety on a hoverboard).

Stops when TARGET_DISTANCE_KM is covered, or SAFETY_FLOOR_SOC_PCT is reached.
SAFETY_FLOOR_SOC_PCT is a hardware interlock, independent of the optimizer's
own OPTIMIZER_FLOOR_SOC_PCT (a planning parameter, not a safety mechanism) --
keep the hardware floor below the optimizer's floor so the optimizer's own
conservatism kicks in first under normal operation.

Run from the repo root: python -m dataset.run_scripts.controller_run

Before the first real 5 km run, do a short dry run (e.g. TARGET_DISTANCE_KM =
0.2) to confirm the control loop, logging, and shutdown all behave -- this
script has not been tested against real hardware.
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
from range_controller.constants import FULL_SPEED, NOMINAL_CAP_AH, WHEEL_CIRCUM_M
from range_controller.efficiency_model import EfficiencyLookup
from range_controller.optimizer import OptimizerParams, optimize_speed
from range_controller.state import Condition, ControllerState
from soc_estimation.soc_relabel import DROPOUT_VOLTAGE, soc_from_vi

######################################## CONFIGS ########################################
MODE = "controller"                  # "baseline" or "controller"
BASELINE_SPEED_PCT = 20              # only used when MODE == "baseline" -- most efficient speed per
                                      # efficiency_summary.csv for this condition (3.81 Wh/km, cheapest
                                      # of all 5 candidates), per docs/range_controller.md's rule: the
                                      # baseline should be the speed the efficiency model itself calls best
CANDIDATE_SPEEDS = [20, 40, 60, 80, 100]  # only used when MODE == "controller" -- all 5 now measured
                                           # for this condition (sweep_run.py, 2026-07-28)
CONDITION = Condition(load_kg=0, rollers_resistance=True)

TARGET_DISTANCE_KM = 5.0
SAFETY_FLOOR_SOC_PCT = 15.0          # hardware interlock -- always stops here regardless of mode
OPTIMIZER_FLOOR_SOC_PCT = 20.0       # optimizer's own planning floor; keep above the safety floor --
                                      # matches this condition's lowest measured SOC bucket (20%), so the
                                      # optimizer won't plan into "no data" territory under normal operation
MARGIN_FRACTION = 0.10

REST_SAMPLE_S = 5.0                  # how long to sit still before the run to anchor live SOC
CONTROL_INTERVAL_S = 5.0             # how often the controller re-evaluates speed
LOG_HZ = 1
LOG_INTERVAL_S = 1.0 / LOG_HZ

hdf5_file = "dataset/all_data/h5_files/range_controller_validation.h5"
_mode_tag = f"baseline{BASELINE_SPEED_PCT}" if MODE == "baseline" else "controller"
run_name = f"run_{_mode_tag}_{TARGET_DISTANCE_KM}km".replace(".", "p")

hb_com_port = "COM3"
hb_baud_rate = 115200
bms_name = "EGIKE_STATION_1"

run_metadata = {
    "description": f"Layer 3b closed-loop validation: {MODE} mode, target {TARGET_DISTANCE_KM} km.",
    "date": get_date_string(),
    "mode": MODE,
    "baseline_speed_pct": BASELINE_SPEED_PCT if MODE == "baseline" else -1,
    "candidate_speeds": str(CANDIDATE_SPEEDS) if MODE == "controller" else "",
    "condition_load_kg": CONDITION.load_kg,
    "condition_rollers_resistance": str(CONDITION.rollers_resistance),
    "target_distance_km": TARGET_DISTANCE_KM,
}
###########################################################################################


def pct_to_raw_speed(pct):
    return int(pct / 100.0 * FULL_SPEED)


class LiveDistanceTracker:
    """RPM -> velocity -> integrated distance. Same conversion as
    build_efficiency_dataset.py's compute_row_level, run online."""

    def __init__(self):
        self.distance_km = 0.0

    def update(self, speed_l, speed_r, dt_s):
        rpm = (abs(speed_l) + abs(speed_r)) / 2.0
        velocity_mps = (rpm / 60.0) * WHEEL_CIRCUM_M
        self.distance_km += velocity_mps * dt_s / 1000.0
        return self.distance_km


class LiveSocTracker:
    """Single-anchor coulomb counter -- see module docstring."""

    def __init__(self, capacity_ah=NOMINAL_CAP_AH):
        self.capacity_ah = capacity_ah
        self.soc_pct = None

    def start(self, rest_voltage, rest_current):
        self.soc_pct = soc_from_vi(rest_voltage, rest_current)
        return self.soc_pct

    def update(self, current_a, dt_s):
        if self.soc_pct is None:
            raise RuntimeError("LiveSocTracker.start() must be called before update()")
        delta_pct = current_a * dt_s / 3600.0 / self.capacity_ah * 100.0
        self.soc_pct = max(0.0, min(100.0, self.soc_pct + delta_pct))
        return self.soc_pct


######################################## STATE ########################################

stop_flag = threading.Event()
state_lock = threading.Lock()

distance_tracker = LiveDistanceTracker()
soc_tracker = LiveSocTracker()
cumulative_energy_wh = 0.0
current_commanded_speed_pct = 0
last_optimizer_warning = None
samples_logged = 0


######################################## CONTROL THREAD ########################################

def control_loop(hoverboard):
    global current_commanded_speed_pct, last_optimizer_warning

    model = EfficiencyLookup() if MODE == "controller" else None
    params = OptimizerParams(
        candidate_speeds=CANDIDATE_SPEEDS,
        floor_soc_pct=OPTIMIZER_FLOOR_SOC_PCT,
        margin_fraction=MARGIN_FRACTION,
    )

    while not stop_flag.is_set():
        with state_lock:
            soc = soc_tracker.soc_pct
            remaining_km = max(0.0, TARGET_DISTANCE_KM - distance_tracker.distance_km)

        if MODE == "baseline":
            target_pct = BASELINE_SPEED_PCT
        else:
            cstate = ControllerState(soc_pct=soc, remaining_distance_km=remaining_km, condition=CONDITION)
            result = optimize_speed(cstate, model, params)
            target_pct = result.speed_pct
            if result.warning:
                last_optimizer_warning = result.warning
                print(f"[OPTIMIZER WARNING] {result.warning}")

        with state_lock:
            current_commanded_speed_pct = target_pct
        print(f"[CONTROL] SOC={soc:.1f}%  remaining={remaining_km:.2f}km  -> speed={target_pct}%")
        hoverboard.ramp_speed(pct_to_raw_speed(target_pct))  # blocking, but only within this thread

        slept = 0.0
        while slept < CONTROL_INTERVAL_S and not stop_flag.is_set():
            time.sleep(0.2)
            slept += 0.2


######################################## LOGGER THREAD ########################################

def data_logger(hoverboard, bms_reader):
    global cumulative_energy_wh, samples_logged

    last_hb = {"hb_speedR_meas": 0, "hb_speedL_meas": 0, "hb_measured_voltage": 0.0, "hb_board_temp": 0.0}
    last_bms = None
    prev_t = time.time()

    while not stop_flag.is_set():
        timestamp = get_timestamp()
        time_string = get_time_string()
        now = time.time()
        dt_s = now - prev_t
        prev_t = now

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
            distance_km = distance_tracker.update(last_hb["hb_speedL_meas"], last_hb["hb_speedR_meas"], dt_s)
            soc_pct = soc_tracker.update(last_bms.get("current", 0.0) or 0.0, dt_s)
            cumulative_energy_wh += abs(last_bms.get("power", 0.0) or 0.0) * dt_s / 3600.0
            energy_wh = cumulative_energy_wh
            commanded_speed = current_commanded_speed_pct

        bms_for_log = dict(last_bms)
        bms_for_log["commanded_speed_pct"] = float(commanded_speed)
        bms_for_log["live_soc_pct"] = float(soc_pct)
        bms_for_log["live_distance_km"] = float(distance_km)
        bms_for_log["cumulative_energy_wh"] = float(energy_wh)

        append_row(hdf5_file, run_name, timestamp, time_string, last_hb, bms_for_log)
        samples_logged += 1

        print(f"[LOG] t={samples_logged}s  dist={distance_km:.3f}km  SOC={soc_pct:.1f}%  "
              f"energy={energy_wh:.1f}Wh  speed={commanded_speed}%")

        # ---- Stop conditions ----
        if distance_km >= TARGET_DISTANCE_KM:
            print(f"Target distance {TARGET_DISTANCE_KM} km reached.")
            stop_flag.set()
            break
        if soc_pct <= SAFETY_FLOOR_SOC_PCT:
            print(f"SAFETY STOP: live SOC {soc_pct:.1f}% <= floor {SAFETY_FLOOR_SOC_PCT}%.")
            stop_flag.set()
            break
        if last_bms.get("voltage") is not None and last_bms["voltage"] <= DROPOUT_VOLTAGE:
            print("BMS dropout detected (voltage below threshold) -- stopping for safety.")
            stop_flag.set()
            break

        elapsed = time.time() - now
        time.sleep(max(0.0, LOG_INTERVAL_S - elapsed))


######################################## MAIN ########################################

def main():
    print(f"Mode: {MODE} | target distance: {TARGET_DISTANCE_KM} km | condition: {CONDITION}")

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

    # ---- Rest sample to anchor live SOC before moving ----
    print(f"Sitting still for {REST_SAMPLE_S:.0f}s to get a stable rest-OCV reading...")
    rest_voltages, rest_currents = [], []
    t_end = time.time() + REST_SAMPLE_S
    while time.time() < t_end:
        s = bms_reader.get_latest()
        if s and s.get("voltage") is not None:
            rest_voltages.append(s["voltage"])
            rest_currents.append(s.get("current", 0.0) or 0.0)
        time.sleep(0.5)
    rest_voltages.sort()
    rest_currents.sort()
    rest_v = rest_voltages[len(rest_voltages) // 2]
    rest_i = rest_currents[len(rest_currents) // 2]
    start_soc = soc_tracker.start(rest_v, rest_i)
    print(f"Rest reading: V={rest_v:.2f}, I={rest_i:.2f} -> starting SOC = {start_soc:.1f}%")

    # ---- HDF5 init ----
    first_bms = bms_reader.get_latest()
    hb_init_sample = hoverboard.get_feedback()
    bms_init_sample = dict(first_bms)
    bms_init_sample["commanded_speed_pct"] = 0.0
    bms_init_sample["live_soc_pct"] = float(start_soc)
    bms_init_sample["live_distance_km"] = 0.0
    bms_init_sample["cumulative_energy_wh"] = 0.0
    init_run_dynamic(hdf5_file, run_name, run_metadata, hb_init_sample, bms_init_sample)

    # ---- Start control + logger threads ----
    control_thread = threading.Thread(target=control_loop, args=(hoverboard,), daemon=True)
    logger_thread = threading.Thread(target=data_logger, args=(hoverboard, bms_reader), daemon=True)
    control_thread.start()
    logger_thread.start()

    def shutdown(*_args):
        print("\nShutting down...")
        stop_flag.set()
        time.sleep(0.5)
        hoverboard.ramp_speed(0)
        hoverboard.close()
        bms_reader.stop()
        with state_lock:
            final_soc = soc_tracker.soc_pct
            final_distance = distance_tracker.distance_km
            final_energy = cumulative_energy_wh
        print("\n========== RUN SUMMARY ==========")
        print(f"  Mode              : {MODE}")
        print(f"  Samples logged    : {samples_logged}")
        print(f"  Distance covered  : {final_distance:.3f} km (target {TARGET_DISTANCE_KM} km)")
        print(f"  Energy used       : {final_energy:.2f} Wh")
        print(f"  Final live SOC    : {final_soc:.1f}% (started {start_soc:.1f}%)")
        if last_optimizer_warning:
            print(f"  Last warning      : {last_optimizer_warning}")
        print("==================================\n")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    while not stop_flag.is_set():
        time.sleep(0.5)
    shutdown()


if __name__ == "__main__":
    main()
