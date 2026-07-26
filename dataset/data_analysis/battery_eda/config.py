"""
config.py
=========

Single source of truth for everything analysis code needs to know about the
*meaning* of the dataset, as opposed to its mechanics. If the HDF5 schema
changes (a field gets renamed, a new sensor is added), this is almost always
the only file that needs to change.

Two ideas drive the layout of this module:

1. SIGNAL_MAP decouples "what a signal means" (e.g. 'voltage') from "what its
   column is called after loading" (e.g. 'bms_voltage'). Every other module
   looks a column up through SIGNAL_MAP instead of hardcoding the HDF5 field
   name, so the rest of the codebase does not care about the raw schema.

2. PHYSICAL_LIMITS / thresholds are pulled out as named constants so a reader
   can sanity-check them against the real hardware instead of hunting for
   magic numbers buried in analysis functions.
"""

# ---------------------------------------------------------------------------
# Pack specification (see CLAUDE.md / run script CONFIGS)
# ---------------------------------------------------------------------------
PACK_NOMINAL_VOLTAGE_V = 42.0
PACK_NOMINAL_CAPACITY_AH = 10.2
PACK_SERIES_CELL_COUNT = 10          # 10S pack -> bms/cell_voltages has 10 columns
PACK_TEMP_SENSOR_COUNT = 3           # bms/temp_values has 3 columns

# ---------------------------------------------------------------------------
# Canonical signal name -> flattened dataframe column name.
#
# Columns are flattened by data_loader as "<hdf5_group>_<dataset_name>"
# (e.g. group "bms", dataset "voltage" -> "bms_voltage"), and 2D datasets are
# expanded to one column per index, 1-based (e.g. "bms_cell_voltages_1..10").
# ---------------------------------------------------------------------------
SIGNAL_MAP = {
    "voltage": "bms_voltage",                       # pack terminal voltage [V]
    "current": "bms_current",                        # + charging, - discharging [A]
    "power": "bms_power",                             # [W], usually V * I from the BMS
    "soc": "bms_battery_level",                       # BMS-reported SOC [%]
    "temperature": "bms_temperature",                 # BMS main/average temperature [degC]
    "charging_flag": "bms_battery_charging",           # BMS charging state, 0/1
    "cycle_charge": "bms_cycle_charge",                # BMS coulomb counter for this cycle
    "cycle_capacity": "bms_cycle_capacity",            # BMS capacity counter for this cycle
    "cycles_counter": "bms_cycles",                    # BMS lifetime cycle counter
    "delta_voltage": "bms_delta_voltage",              # BMS-reported max-min cell spread [V]
    "cell_count": "bms_cell_count",
    "motor_speed_r": "hoverboard_hb_speedR_meas",
    "motor_speed_l": "hoverboard_hb_speedL_meas",
    "hb_voltage": "hoverboard_hb_measured_voltage",
    "hb_temp": "hoverboard_hb_board_temp",
    "hppc_phase": "hoverboard_hppc_phase",             # only present in HPPC runs
    "time": "elapsed_s",                               # seconds since the start of the run
    "datetime": "datetime",
    "run": "run_name",
}

# Prefixes for the columns produced by expanding 2D HDF5 datasets.
CELL_VOLTAGE_PREFIX = "bms_cell_voltages_"   # -> bms_cell_voltages_1 .. bms_cell_voltages_10
CELL_TEMP_PREFIX = "bms_temp_values_"        # -> bms_temp_values_1 .. bms_temp_values_3

# ---------------------------------------------------------------------------
# Physical plausibility bounds, used by data_cleaning.remove_impossible_values
# and anomaly_detection. These are deliberately generous (they should only
# catch sensor faults / logging bugs, not flag legitimate extremes).
# ---------------------------------------------------------------------------
PHYSICAL_LIMITS = {
    "bms_voltage": (0.0, 50.0),            # 10S pack: 10*2.0V .. 10*4.35V ballpark, padded
    "bms_current": (-40.0, 40.0),          # generous headroom over observed +/-5..10A
    "bms_temperature": (-20.0, 80.0),
    "bms_battery_level": (0.0, 100.0),
}
# Per-cell voltage bounds, applied to every bms_cell_voltages_N column.
CELL_VOLTAGE_LIMITS = (0.0, 4.35)
# Per-sensor temperature bounds, applied to every bms_temp_values_N column.
CELL_TEMP_LIMITS = (-20.0, 80.0)

# ---------------------------------------------------------------------------
# Cycle / segment detection
# ---------------------------------------------------------------------------
# |current| below this is treated as "rest" rather than charge/discharge.
CURRENT_REST_THRESHOLD_A = 0.05
# A charge/discharge/rest segment shorter than this many samples is folded
# into its neighbour instead of being reported as its own segment (removes
# single-sample noise from the state machine).
MIN_SEGMENT_SAMPLES = 3

# ---------------------------------------------------------------------------
# Resistance estimation (R = dV / dI at current steps)
# ---------------------------------------------------------------------------
RESISTANCE_MIN_DI_A = 0.5       # ignore steps smaller than this (too noisy to trust)
RESISTANCE_SETTLE_SAMPLES = 2   # samples after the step used for the "after" voltage
RESISTANCE_MAX_OHM = 5.0        # discard R estimates above this as sensor artifacts

# ---------------------------------------------------------------------------
# Anomaly detection thresholds
# ---------------------------------------------------------------------------
CURRENT_JUMP_THRESHOLD_A = 10.0        # |delta current| between consecutive samples
VOLTAGE_DISCONTINUITY_THRESHOLD_V = 1.0  # |delta voltage| between consecutive samples
DROPOUT_GAP_FACTOR = 5.0               # gap > factor * median dt => communication dropout
SPIKE_ROLLING_WINDOW = 11              # samples, odd, centered rolling median for spike test
SPIKE_ZSCORE_THRESHOLD = 6.0           # robust z-score threshold for a spike flag

# ---------------------------------------------------------------------------
# Smoothing / cleaning defaults
# ---------------------------------------------------------------------------
DEFAULT_SMOOTH_WINDOW = 5
DEFAULT_OUTLIER_IQR_FACTOR = 3.0

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
ROLLING_WINDOWS_S = [5, 30, 60]   # window sizes (in samples, ~seconds at 1 Hz logging)

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
FIGURE_DPI = 150
PLOT_STYLE = "seaborn-v0_8-whitegrid"   # matplotlib style; falls back silently if missing
MAX_PAIRPLOT_COLUMNS = 6                 # keep pair plots readable
# FFT produces up to 2 figures (current, voltage) per run; above this many
# runs it is skipped by default (pass --force-fft on the CLI to override).
MAX_RUNS_FOR_FFT = 8
