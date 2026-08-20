"""
HPPC (Hybrid Pulse Power Characterization) run script — semi-automated,
using the UTL8211+ DC electronic load (drivers/utl8211.py) instead of the
hoverboard motor for the discharge pulse and step-discharge phases.

Sequence repeated at each SOC step (every soc_step %):
  1. OCV rest              -- load off, voltage settles (ocv_rest_duration s)
  2. Discharge pulse       -- CC load pulse for pulse_discharge_duration s
  3. Rest between pulses   -- load off for rest_between_pulses s
  4. Charger connect window -- audible alert; countdown while user connects charger
  5. Charge pulse          -- charger connected, load off, logged for pulse_charge_duration s
  6. Charger disconnect window -- audible alert; countdown while user disconnects charger
  7. Rest after charge     -- load off for rest_after_charge_pulse s
  8. Step discharge        -- constant CC load until SOC drops by soc_step %

Stops when SOC <= min_soc.

The load's serial port is not thread-safe, so all load.* calls (from both
the logger thread and the main HPPC control flow) go through load_lock.
"""

from dataset.dataset_utils import (
    init_run_dynamic, append_row,
    get_timestamp, get_date_string, get_time_string
)
from drivers.bms_reader import BMSReader
from drivers.utl8211 import UTL8211, Mode
import threading
import time
try:
    import winsound
    _HAS_WINSOUND = True
except ImportError:
    _HAS_WINSOUND = False

######################################## CONFIGS ########################################

hdf5_file   = "dataset/all_data/h5_files/dc_load_bms_HPPC_data_2ndlife.h5"
run_name    = "run_hppc_dc_load_01"
run_metadata = {
    "description": "HPPC test: UTL8211 CC discharge pulse + manual charger charge pulse, 10% SOC steps.",
    "date":          get_date_string(),
    "battery_pack":  "Lithium-Ion, 42v, Nominal Capacity 10.2Ah, Current Capacity: Unknown",
    "battery_age":   "2nd-life pack",
    "Logging rate":  "1 sample/sec",
    "protocol":      "HPPC (DC electronic load)",
}

load_com_port  = "COM9"
load_baud_rate = 9600
bms_name       = "EGIKE_STATION_1"
LOG_HZ       = 1
sample_interval = 1.0 / LOG_HZ

# --- Discharge pulse ---
pulse_discharge_current  = 3.0   # A, CC-mode setpoint for discharge pulse
pulse_discharge_duration = 10    # seconds

# --- Rest between discharge and charge pulse ---
rest_between_pulses = 40  # seconds

# --- Manual charge pulse window ---
# Total manual window = charger_connect_window + pulse_charge_duration + charger_disconnect_window
charger_connect_window    = 5    # seconds: time given to plug in the charger
pulse_charge_duration     = 10   # seconds: actual charge pulse (charger connected)
charger_disconnect_window = 5    # seconds: time given to unplug the charger

# --- Rest after full charge pulse sequence ---
rest_after_charge_pulse = 40  # seconds

# --- Step discharge parameters ---
step_discharge_current  = 9.0    # A, CC-mode setpoint used to discharge between steps
soc_step                = 10.0   # % SOC to drop per step
ocv_rest_duration       = 30 * 60  # seconds (30-minute OCV rest)

# --- Protection ---
current_protection_margin = 1.1  # OCP = margin * max(pulse_discharge_current, step_discharge_current)

# --- SOC limits ---
min_soc = 20   # stop HPPC when SOC falls to or below this value

######################################## END CONFIGS ########################################

# Phase IDs — stored in HDF5 at every sample for offline analysis
PHASE_OCV_REST           = 0
PHASE_DISCHARGE_PULSE    = 1
PHASE_REST_BETWEEN       = 2
PHASE_CHARGER_CONNECT    = 3  # manual window: user plugging in charger
PHASE_CHARGE_PULSE       = 4  # charger is connected, active charge pulse
PHASE_CHARGER_DISCONNECT = 5  # manual window: user unplugging charger
PHASE_REST_AFTER         = 6
PHASE_STEP_DISCHARGE     = 7
PHASE_DONE               = 8

PHASE_NAMES = {
    PHASE_OCV_REST:           "ocv_rest",
    PHASE_DISCHARGE_PULSE:    "discharge_pulse",
    PHASE_REST_BETWEEN:       "rest_between",
    PHASE_CHARGER_CONNECT:    "charger_connect",
    PHASE_CHARGE_PULSE:       "charge_pulse",
    PHASE_CHARGER_DISCONNECT: "charger_disconnect",
    PHASE_REST_AFTER:         "rest_after",
    PHASE_STEP_DISCHARGE:     "step_discharge",
    PHASE_DONE:               "done",
}

# Shared state
stop_flag    = threading.Event()
phase_lock   = threading.Lock()
load_lock    = threading.Lock()  # serializes all access to the load's serial port
current_phase = PHASE_OCV_REST
current_target_current = 0.0

# HDF5 schema templates
load_init_sample = {
    "load_target_current": 0.0,
    "load_meas_voltage":   0.0,
    "load_meas_current":   0.0,
    "load_meas_power":     0.0,
    "hppc_phase":          0,    # logged at every sample for offline phase identification
}

bms_init_sample = {
    "battery_charging": False,
    "battery_level":    0.0,
    "voltage":          0.0,
    "current":          0.0,
    "cycle_charge":     0,
    "temp_sensors":     0,
    "temp_values":      [0, 0, 0],
    "power":            0.0,
    "cycle_capacity":   0.0,
    "cycles":           0,
    "delta_voltage":    0.0,
    "temperature":      0.0,
    "cell_count":       0,
    "cell_voltages":    [0.0] * 10,
}

last_bms = dict(bms_init_sample)

######################################## LOAD HELPERS ########################################

def load_set_current(load, amps):
    """Thread-safe current setpoint update; also records it for logging."""
    global current_target_current
    with load_lock:
        load.set_current(amps)
    with phase_lock:
        current_target_current = amps

def load_input_on(load, on=True):
    with load_lock:
        load.input_on(on)

######################################## DATA LOGGER ########################################

def data_logger(load, bms_reader):
    global last_bms
    while not stop_flag.is_set():
        timestamp   = get_timestamp()
        time_string = get_time_string()

        with load_lock:
            meas = load.measure_all()
        with phase_lock:
            target = current_target_current
            phase  = current_phase
        load_dict = {
            "load_target_current": target,
            "load_meas_voltage":   meas["voltage"],
            "load_meas_current":   meas["current"],
            "load_meas_power":     meas["power"],
            "hppc_phase":          phase,
        }

        bms_sample = bms_reader.get_latest()
        if bms_sample is not None:
            last_bms = bms_sample
        bms_dict = last_bms

        phase_label = PHASE_NAMES.get(phase, "?")
        soc_val     = bms_dict.get("battery_level", "?")
        v_val       = bms_dict.get("voltage", "?")
        i_val       = bms_dict.get("current", "?")
        print(f"[{phase_label:<20}]  SOC={soc_val}%  V={v_val}V  I={i_val}A")

        append_row(hdf5_file, run_name, timestamp, time_string, load_dict, bms_dict)
        time.sleep(sample_interval)

######################################## HPPC CONTROLLER ########################################

def set_phase(phase):
    global current_phase
    with phase_lock:
        current_phase = phase
    print(f"\n>>> Phase: {PHASE_NAMES[phase]}")

def get_soc(bms_reader):
    sample = bms_reader.get_latest()
    if sample is not None:
        return sample.get("battery_level", None)
    return None

def _beep():
    if _HAS_WINSOUND:
        for _ in range(3):
            winsound.Beep(1000, 400)
            time.sleep(0.1)
    else:
        print("\a\a\a", end="", flush=True)

def _alert(message):
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  *** {message} ***")
    print(f"{bar}\n")
    _beep()

def _countdown(seconds, label=""):
    """Second-by-second countdown. Exits early on stop_flag."""
    for remaining in range(seconds, 0, -1):
        if stop_flag.is_set():
            break
        print(f"  {remaining:3d}s  {label}")
        time.sleep(1)

def rest(duration_s, label=""):
    """Blocking rest. Prints every 60 s; exits early on stop_flag."""
    print(f"  Resting {duration_s}s  {label}")
    end_time = time.time() + duration_s
    while time.time() < end_time and not stop_flag.is_set():
        remaining = int(end_time - time.time())
        print(f"    {remaining}s remaining ...")
        time.sleep(min(60.0, end_time - time.time()))

def manual_charge_pulse():
    """
    Guides the user through the manual charger-based charge pulse:
      1. Audible alert + countdown to plug in the charger.
      2. Charge pulse window (charger connected, data logged).
      3. Audible alert + countdown to unplug the charger.
    Load input must be off before calling this.
    """
    # Connect window
    set_phase(PHASE_CHARGER_CONNECT)
    _alert(f"CONNECT CHARGER NOW  —  {charger_connect_window}s to plug in")
    _countdown(charger_connect_window, "to connect charger")

    # Active charge pulse
    set_phase(PHASE_CHARGE_PULSE)
    print(f"  Charge pulse active — {pulse_charge_duration}s")
    _countdown(pulse_charge_duration, "charge pulse remaining")

    # Disconnect window
    set_phase(PHASE_CHARGER_DISCONNECT)
    _alert(f"DISCONNECT CHARGER NOW  —  {charger_disconnect_window}s to unplug")
    _countdown(charger_disconnect_window, "to disconnect charger")
    print("  Charger disconnected. Continuing run automatically...")

def run_hppc(load, bms_reader):
    soc = None
    while soc is None and not stop_flag.is_set():
        soc = get_soc(bms_reader)
        time.sleep(1)

    print(f"\nStarting HPPC.  SOC={soc:.1f}%  stop at={min_soc}%  step={soc_step}%")
    print(f"Manual charge window: {charger_connect_window}s connect + "
          f"{pulse_charge_duration}s pulse + {charger_disconnect_window}s disconnect "
          f"= {charger_connect_window + pulse_charge_duration + charger_disconnect_window}s total\n")

    next_soc_target = soc - soc_step

    while not stop_flag.is_set():
        soc = get_soc(bms_reader)
        if soc is None:
            time.sleep(1)
            continue
        if soc <= min_soc:
            print(f"SOC={soc:.1f}% reached min_soc={min_soc}%. HPPC complete.")
            break

        print(f"\n{'#'*60}")
        print(f"  HPPC step:  SOC={soc:.1f}%   next target: {next_soc_target:.1f}%")
        print(f"{'#'*60}")

        # 1. Discharge pulse
        set_phase(PHASE_DISCHARGE_PULSE)
        print(f"  Load current={pulse_discharge_current}A  duration={pulse_discharge_duration}s")
        load_set_current(load, pulse_discharge_current)
        load_input_on(load, True)
        time.sleep(pulse_discharge_duration)
        load_input_on(load, False)
        load_set_current(load, 0.0)

        # 2. Rest between pulses
        set_phase(PHASE_REST_BETWEEN)
        rest(rest_between_pulses, "(between discharge and charge pulse)")

        # 3. Manual charge pulse (load stays off throughout)
        manual_charge_pulse()

        # 4. Rest after charge pulse
        set_phase(PHASE_REST_AFTER)
        rest(rest_after_charge_pulse, "(after charge pulse)")

        # 5. Step discharge — drive until SOC drops by soc_step
        set_phase(PHASE_STEP_DISCHARGE)
        print(f"  Load current={step_discharge_current}A  until SOC <= {next_soc_target:.1f}%")
        load_set_current(load, step_discharge_current)
        load_input_on(load, True)
        while not stop_flag.is_set():
            soc = get_soc(bms_reader)
            if soc is None:
                time.sleep(1)
                continue
            if soc <= next_soc_target or soc <= min_soc:
                break
            time.sleep(1)
        load_input_on(load, False)
        load_set_current(load, 0.0)

        next_soc_target = max(next_soc_target - soc_step, min_soc)

        # 6. OCV rest before next step's pulses
        set_phase(PHASE_OCV_REST)
        soc = get_soc(bms_reader) or 0.0
        rest(ocv_rest_duration, f"(OCV rest, SOC≈{soc:.1f}%)")

    set_phase(PHASE_DONE)
    load_input_on(load, False)
    load_set_current(load, 0.0)
    stop_flag.set()
    print("HPPC run complete.")

######################################## MAIN ########################################

init_run_dynamic(hdf5_file, run_name, run_metadata, load_init_sample, bms_init_sample)

load = UTL8211(port=load_com_port, baudrate=load_baud_rate)
print(f"Load connected: {load.idn()}")

bms_reader = BMSReader(device_name=bms_name)
bms_reader.start()

print(f"DC load on {load_com_port} at {load_baud_rate} baud.")
print(f"BMS Reader started for '{bms_name}'.")
print(f"Run: {run_name}")

while bms_reader.get_latest() is None:
    print("Waiting for BMS Bluetooth connection...")
    time.sleep(1)

# Configure the DC load for CC-mode pulsing
load.set_mode(Mode.CURRENT)
load.set_current(0.0)
protection_current = max(pulse_discharge_current, step_discharge_current) * current_protection_margin
load.set_current_protection(protection_current)
load.check_errors()
load.input_on(False)

logger_thread = threading.Thread(target=data_logger, args=(load, bms_reader), daemon=True)
logger_thread.start()

try:
    run_hppc(load, bms_reader)
except KeyboardInterrupt:
    print("\nInterrupted. Stopping load...")
    stop_flag.set()
    load_input_on(load, False)
    load_set_current(load, 0.0)
finally:
    logger_thread.join(timeout=5)
    load.input_on(False)
    load.set_current(0.0)
    load.close()
    print("Cleanup complete.")
