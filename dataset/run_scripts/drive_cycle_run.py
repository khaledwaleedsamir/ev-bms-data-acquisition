"""
Panasonic 18650PF drive-cycle replay run script.

Replays a drive-cycle current profile from the Panasonic 18650PF dataset
(dataset/Panasonic 18650PF Data/.../25degC/Drive cycles/*.mat) on the
UTL8211+ DC electronic load (drivers/utl8211.py).

The load can only sink current, so the (brief) regen/charge portions of the
profile (positive current in the Panasonic sign convention) are clamped to
0A (idle) rather than reversed. The remaining discharge-current magnitude is
then min-max normalized to [0, max_load_current] (0 at idle, max_load_current
at the profile's own peak) so the full drive-cycle *shape* is preserved
without any clipping, within the load's 400W / 9A safety ceiling:

    load_current(t) = |discharge_current(t)| * (max_load_current / peak(|discharge_current|))

The source current profile is resampled onto a fixed control period
(control_period, from CONTROL_HZ) before replay, so the control/logging
loop runs at a steady rate instead of following the dataset's irregular
original sample spacing.

The run aborts early (load input off, current cut to 0) if BMS-reported
SOC or pack voltage drops below the configured cutoffs.
"""

import os
import time
import threading

import numpy as np

from dataset.dataset_utils import (
    init_run_dynamic, append_row,
    get_timestamp, get_date_string, get_time_string
)
from drivers.bms_reader import BMSReader
from drivers.utl8211 import UTL8211, Mode
from soc_estimation.panasonic_loader import load_mat_file, CYCLE_TYPES

######################################## CONFIGS ########################################

hdf5_file = "dataset/all_data/h5_files/drive_cycle_data_2ndlife.h5"
run_name  = "run_003_drivecycle_US06_6"
run_metadata = {
    "description":    "US06 drive-cycle current profile replayed on UTL8211 DC load.",
    "date":           get_date_string(),
    "battery_pack":   "Lithium-Ion, 42v, Nominal Capacity 10.5Ah, Current Capacity: Unknown",
    "battery_age":    "Second-life pack",
}

load_com_port  = "COM7"
load_baud_rate = 9600
bms_name       = "EGIKE_STATION_1"

# --- Drive cycle selection (25degC only) ---
# NOTE: each Panasonic .mat file is the single reference drive cycle (e.g. UDDS
# is ~23 minutes) looped back-to-back for hours to fully discharge the 2.9Ah
# reference cell. num_repeats controls how many loops of that single cycle to
# replay -- 1 repeat is the actual drive cycle duration.
panasonic_root = os.path.join("dataset", "Panasonic 18650PF Data", "Panasonic 18650PF Data")
cycle_name     = "US06"   # one of: Cycle_1, Cycle_2, Cycle_3, Cycle_4, US06, HWFTa, HWFTb, UDDS, LA92, NN
num_repeats    = 20        # how many back-to-back loops of the single cycle to replay

# --- Replay timing ---
CONTROL_HZ     = 1.0              # fixed control-loop / logging rate for replay (1 sample/sec)
control_period = 1.0 / CONTROL_HZ

# --- Protection ---
# UTL8211 load is rated 400W max. At 42V nominal pack voltage that's ~9.5A --
# derate to 9A as a hard safety ceiling; the profile is normalized to exactly
# this peak (see build_setpoint_profile), so no clipping is needed.
max_load_current           = 9.0   # A, profile peak after min-max normalization
current_protection_margin  = 1.1   # OCP = margin * min(peak current, max_load_current)

# --- Safety cutoffs (abort the run without raising) ---
min_soc_cutoff     = 15.0   # % BMS battery_level
min_voltage_cutoff = 30.0   # V pack voltage (BMS-reported)

######################################## END CONFIGS ########################################

stop_flag = threading.Event()

hb_init_sample = {  # repurposed "hoverboard" HDF5 slot -- holds DC load fields for this run type
    "load_target_current": 0.0,
    "load_meas_voltage":   0.0,
    "load_meas_current":   0.0,
    "load_meas_power":     0.0,
    "cycle_time_s":        0.0,
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

######################################## DRIVE CYCLE LOADING ########################################

def find_cycle_file(root, cycle_name):
    """Locate the single 25degC drive-cycle .mat file matching cycle_name.
    Combined files (e.g. US06_HWFET_UDDS_LA92) match more than one keyword
    in CYCLE_TYPES and are excluded, same rule as panasonic_loader.py."""
    if cycle_name not in CYCLE_TYPES:
        raise ValueError(f"cycle_name={cycle_name!r} not in known CYCLE_TYPES={CYCLE_TYPES}")

    folder = os.path.join(root, "25degC", "Drive cycles")
    matches = [
        fname for fname in sorted(os.listdir(folder))
        if fname.endswith(".mat")
        and cycle_name in fname
        and sum(1 for ct in CYCLE_TYPES if ct in fname) == 1
    ]
    if not matches:
        raise FileNotFoundError(f"No 25degC drive-cycle file found for cycle_name={cycle_name!r} in {folder}")
    if len(matches) > 1:
        raise FileNotFoundError(f"Multiple 25degC drive-cycle files match cycle_name={cycle_name!r}: {matches}")
    return os.path.join(folder, matches[0])


def detect_single_cycle_duration_s(t_src, i_src, min_period_s=60.0, max_search_s=3600.0, min_acf=0.5):
    """
    Autocorrelation-based estimate of how long one repetition of the
    drive-cycle current profile lasts, for .mat files that loop the same
    cycle back-to-back to fully discharge the reference cell. Falls back to
    the full profile duration if no clear repeating period is found (e.g.
    NN/Cycle_x files, which are not simple loops).
    """
    total_s = float(t_src[-1])
    t_grid_1hz = np.arange(0.0, total_s, 1.0)
    i_grid_1hz = np.interp(t_grid_1hz, t_src, i_src)
    centered = i_grid_1hz - i_grid_1hz.mean()

    n = len(centered)
    search_max = int(min(max_search_s, n // 2))
    lag_min = int(min_period_s)
    if n < 2 * min_period_s or search_max <= lag_min:
        return total_s

    spectrum = np.fft.rfft(centered, n=2 * n)
    acf = np.fft.irfft(spectrum * np.conj(spectrum))[:n]
    acf /= acf[0]

    window = acf[lag_min:search_max]
    peak_lag = int(np.argmax(window)) + lag_min
    if acf[peak_lag] < min_acf:
        return total_s
    return float(peak_lag)


def build_setpoint_profile(mat_path):
    """
    Load a Panasonic drive-cycle .mat file and return (t_grid, load_currents):
      t_grid        : uniform time grid at control_period, starting at 0,
                       covering num_repeats loops of the detected single-cycle
                       duration (or the whole file if no loop is detected).
      load_currents : discharge-current magnitude (A), min-max normalized to
                       [0, max_load_current] against the replayed segment's
                       own peak. Regen/charge portions (source current >= 0)
                       are 0 since the load can only sink current.
    """
    df = load_mat_file(mat_path, temperature_c=25)
    t_src = df["Time [s]"].to_numpy()
    i_src = df["Current [A]"].to_numpy()

    period_s = detect_single_cycle_duration_s(t_src, i_src)
    replay_duration_s = min(period_s * num_repeats, float(t_src[-1]))
    print(f"Detected single-cycle duration ~{period_s:.0f}s. Replaying {num_repeats} repeat(s) "
          f"= {replay_duration_s:.0f}s ({replay_duration_s / 60:.1f} min).")

    t_grid = np.arange(0.0, replay_duration_s, control_period)
    i_grid = np.interp(t_grid, t_src, i_src)

    discharge_current = np.where(i_grid < 0.0, -i_grid, 0.0)
    peak = discharge_current.max()
    if peak <= 0.0:
        raise ValueError(f"No discharge current found in {mat_path}")

    load_currents = (discharge_current * (max_load_current / peak)).astype(np.float32)
    return t_grid, load_currents

######################################## DRIVE CYCLE REPLAY ########################################

def run_drive_cycle(load, bms_reader, t_grid, currents):
    global last_bms

    load.set_mode(Mode.CURRENT)
    load.set_current(0.0)
    peak_current = float(currents.max()) if len(currents) else 0.0
    protection_current = min(peak_current * current_protection_margin, max_load_current)
    load.set_current_protection(protection_current)
    load.check_errors()

    print(f"Starting drive cycle replay: {cycle_name}  peak_current={peak_current:.2f}A  "
          f"duration={t_grid[-1]:.1f}s  control_hz={CONTROL_HZ}")

    load.input_on(True)
    start_time = time.monotonic()

    try:
        for i, (t, target_current) in enumerate(zip(t_grid, currents)):
            if stop_flag.is_set():
                break

            sample = bms_reader.get_latest()
            if sample is not None:
                last_bms = sample

            soc     = last_bms.get("battery_level")
            voltage = last_bms.get("voltage")
            if soc is not None and soc <= min_soc_cutoff:
                print(f"\nSOC={soc:.1f}% <= min_soc_cutoff={min_soc_cutoff}%. Aborting run.")
                break
            if voltage is not None and voltage <= min_voltage_cutoff:
                print(f"\nVoltage={voltage:.2f}V <= min_voltage_cutoff={min_voltage_cutoff}V. Aborting run.")
                break

            load.set_current(float(target_current))
            meas = load.measure_all()

            hb_dict = {
                "load_target_current": float(target_current),
                "load_meas_voltage":   meas["voltage"],
                "load_meas_current":   meas["current"],
                "load_meas_power":     meas["power"],
                "cycle_time_s":        float(t),
            }
            append_row(hdf5_file, run_name, get_timestamp(), get_time_string(), hb_dict, last_bms)

            soc_str = f"{soc:.1f}%" if soc is not None else "?"
            print(f"\r  t={t:7.1f}s  I_set={target_current:5.2f}A  "
                  f"V_meas={meas['voltage']:5.2f}V  I_meas={meas['current']:5.2f}A  SOC={soc_str}  ",
                  end="")

            next_tick  = start_time + (i + 1) * control_period
            sleep_time = next_tick - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        load.set_current(0.0)
        load.input_on(False)
        print("\nDrive cycle replay stopped. Load input off.")

######################################## MAIN ########################################

mat_path = find_cycle_file(panasonic_root, cycle_name)
print(f"Drive cycle source: {mat_path}")
t_grid, currents = build_setpoint_profile(mat_path)

init_run_dynamic(hdf5_file, run_name, run_metadata, hb_init_sample, bms_init_sample)

load = UTL8211(port=load_com_port, baudrate=load_baud_rate)
print(f"Load connected: {load.idn()}")

bms_reader = BMSReader(device_name=bms_name)
bms_reader.start()
print(f"BMS Reader started for '{bms_name}'.")

while bms_reader.get_latest() is None:
    print("Waiting for BMS Bluetooth connection...")
    time.sleep(1)

try:
    run_drive_cycle(load, bms_reader, t_grid, currents)
except KeyboardInterrupt:
    print("\nInterrupted. Stopping load...")
    stop_flag.set()
finally:
    load.set_current(0.0)
    load.input_on(False)
    load.close()
    bms_reader.stop()
    print("Cleanup complete.")
