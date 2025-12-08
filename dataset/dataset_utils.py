import h5py
import os
import numpy as np
from datetime import datetime
import time

def init_run(hdf5_file: str, run_name: str, metadata: dict):
    """
    Initialize a new run in the HDF5 dataset, creating the file if it doesn't exist.

    Args:
        hdf5_file (str): Path to the HDF5 file.
        run_name (str): Name of the run, e.g., "run_001".
        metadata (dict): Key/value pairs to store as run metadata (can include description).
    """
    # Check if file exists
    file_exists = os.path.exists(hdf5_file)

    with h5py.File(hdf5_file, "a") as f:
        if run_name in f:
            raise ValueError(f"Run {run_name} already exists in {hdf5_file}")

        # Create a group for this run
        g_run = f.create_group(run_name)

        # Shared timestamp
        g_run.create_dataset("timestamp_ns", shape=(0,), maxshape=(None,), dtype='float64')

        # Hoverboard datasets
        g_hover = g_run.create_group("hoverboard")
        g_hover.create_dataset("speedR_meas", shape=(0,), maxshape=(None,), dtype='int32')
        g_hover.create_dataset("speedL_meas", shape=(0,), maxshape=(None,), dtype='int32')
        g_hover.create_dataset("batVoltage", shape=(0,), maxshape=(None,), dtype='float32')
        g_hover.create_dataset("boardTemp", shape=(0,), maxshape=(None,), dtype='float32')

        # BMS datasets
        g_bms = g_run.create_group("bms")
        g_bms.create_dataset("battery_charging", shape=(0,), maxshape=(None,), dtype='float32')
        g_bms.create_dataset("battery_level", shape=(0,), maxshape=(None,), dtype='float32')
        g_bms.create_dataset("voltage", shape=(0,), maxshape=(None,), dtype='float32')
        g_bms.create_dataset("current", shape=(0,), maxshape=(None,), dtype='float32')
        g_bms.create_dataset("temperatures", shape=(0, 3), maxshape=(None, 3), dtype='float32')
        g_bms.create_dataset("cell_voltages", shape=(0, 10), maxshape=(None, 10), dtype='float32')

        # Metadata group
        g_meta = g_run.create_group("metadata")
        for key, value in metadata.items():
            g_meta.attrs[key] = value

    if file_exists:
        print(f"Run {run_name} added to existing file {hdf5_file}")
    else:
        print(f"New HDF5 file {hdf5_file} created and run {run_name} initialized")

def init_run_dynamic(hdf5_file: str, run_name: str, metadata: dict, 
                     hoverboard_sample: dict, bms_sample: dict):
    """
    Initialize a new run in the HDF5 dataset based on sample feedback dicts.

    Args:
        hdf5_file (str): Path to HDF5 file
        run_name (str): Run name, e.g., 'run_001'
        metadata (dict): Metadata for the run
        hoverboard_sample (dict): Example hoverboard dict for column names and types
        bms_sample (dict): Example BMS dict for column names and types
    """
    file_exists = os.path.exists(hdf5_file)
    
    with h5py.File(hdf5_file, "a") as f:
        if run_name in f:
            raise ValueError(f"Run {run_name} already exists in {hdf5_file}")
        
        g_run = f.create_group(run_name)
        # Shared timestamp
        g_run.create_dataset("timestamp_ms", shape=(0,), maxshape=(None,), dtype='float64')
        g_run.create_dataset("time_string", shape=(0,), maxshape=(None,), dtype=h5py.string_dtype(encoding='utf-8'))
    
        
        # Hoverboard datasets
        g_hover = g_run.create_group("hoverboard")
        for key, val in hoverboard_sample.items():
            dtype = np.float32 if isinstance(val, float) else np.int32
            if isinstance(val, list):
                # Determine shape from list length
                g_hover.create_dataset(key, shape=(0, len(val)), maxshape=(None, len(val)), dtype=np.float32)
            else:
                g_hover.create_dataset(key, shape=(0,), maxshape=(None,), dtype=dtype)
        
        # BMS datasets
        g_bms = g_run.create_group("bms")
        for key, val in bms_sample.items():
            dtype = np.float32 if isinstance(val, float) else np.int32
            if isinstance(val, list):
                g_bms.create_dataset(key, shape=(0, len(val)), maxshape=(None, len(val)), dtype=np.float32)
            else:
                g_bms.create_dataset(key, shape=(0,), maxshape=(None,), dtype=dtype)
        
        # Metadata
        g_meta = g_run.create_group("metadata")
        for k, v in metadata.items():
            g_meta.attrs[k] = v
    
    if file_exists:
        print(f"Run {run_name} added to existing file {hdf5_file}")
    else:
        print(f"New HDF5 file {hdf5_file} created and run {run_name} initialized")

def append_row(hdf5_file: str, run_name: str, timestamp_ms: float, time_string: str,
               hoverboard_data: dict, bms_data: dict):
    """
    Append a new row of data to the HDF5 datasets for a specific run.

    Args:
        hdf5_file (str): Path to the HDF5 file.
        run_name (str): Name of the run (e.g., "run_001").
        timestamp_ms (float): Timestamp in milliseconds.
        time_string (str): Human-readable time string.
        hoverboard_data (dict): Dict with hoverboard measurements.
        bms_data (dict): Dict with BMS measurements.
    """
    with h5py.File(hdf5_file, "a") as f:
        if run_name not in f:
            raise ValueError(f"Run {run_name} does not exist in {hdf5_file}")

        g_run = f[run_name]

        # Append timestamps
        for ts_name, ts_val in [("timestamp_ms", timestamp_ms), ("time_string", time_string)]:
            ts_dataset = g_run[ts_name]
            ts_dataset.resize((ts_dataset.shape[0] + 1,))
            ts_dataset[-1] = ts_val

        # Append hoverboard data
        g_hover = g_run["hoverboard"]
        for key, value in hoverboard_data.items():
            if key not in g_hover:
                raise KeyError(f"Hoverboard dataset '{key}' not found in run '{run_name}'")
            ds = g_hover[key]


            # Handle 2D arrays
            if len(ds.shape) == 2:
                value = np.array(value).reshape(1, ds.shape[1])
                ds.resize((ds.shape[0] + 1, ds.shape[1]))
                ds[-1, :] = value
            else:
                ds.resize((ds.shape[0] + 1,))
                ds[-1] = value

        # Append BMS data
        g_bms = g_run["bms"]
        if bms_data is None:
            bms_data = {}
        for key, value in bms_data.items():
            if key not in g_bms:
                raise KeyError(f"BMS dataset '{key}' not found in run '{run_name}'")
            ds = g_bms[key]

            # Handle 2D arrays
            if len(ds.shape) == 2:
                value = np.array(value).reshape(1, ds.shape[1])
                ds.resize((ds.shape[0] + 1, ds.shape[1]))
                ds[-1, :] = value
            else:
                ds.resize((ds.shape[0] + 1,))
                ds[-1] = value

def get_timestamp():
    """
    Get the current timestamp in milliseconds.
    
    Returns:
        int: Current timestamp in milliseconds.
    """
    return int(time.time() * 1000)  # integer milliseconds

def get_time_string():
    """
    Get the current time as a string in HH:MM:SS.ssssss format.
    
    Returns:
        str: Current time string.
    """
    return datetime.now().strftime("%H:%M:%S.%f")[:-1]

def get_date_string():
    """
    Get the current date as a string in YYYY-MM-DD format.
    
    Returns:
        str: Current date string.
    """
    return datetime.now().strftime("%Y-%m-%d")


# ---------- Dummy Samples for testing  ----------

# hb_sample = {
#     "speedR_meas": 100,
#     "speedL_meas": 95,
#     "batVoltage": 36.5,
#     "boardTemp": 42.0
# }

# bms_sample = {
#     "battery_charging": True,
#     "battery_level": 85.0,
#     "voltage": 41.2,
#     "current": 1.3,
#     "cycle_charge": 2.5,
#     "total_charge": 5,
#     "temp_sensors": 3,
#     "temp_values": [33.5, 34.0, 25],
#     "power": 50.0,
#     "cycle_capacity": 10.0,
#     "cycles": 15,
#     "delta_voltage": 0.05,
#     "balance_current": 0.1,
#     "temperature": 34.0,
#     "cell_count": 10,
#     "cell_voltages": [3.7]*10
# }

# # ---------- Run metadata ----------
# hdf5_file = "test_dataset.h5"
# run_name = "run_002"
# run_metadata = {
#     "description": "Test run with dummy hoverboard and BMS data",
#     "date": datetime.now().strftime("%Y-%m-%d"),
#     "battery_pack": "Li-ion 36V 10Ah",
#     "battery_age": "new"
# }

# # ---------- Initialize run ----------
# init_run_dynamic(hdf5_file, run_name, run_metadata, hb_sample, bms_sample)

# # ---------- Append a single row ----------
# ts_ms = get_timestamp()
# ts_str = get_time_string()

# append_row(hdf5_file, run_name, ts_ms, ts_str, hb_sample, bms_sample)

# print(f"Test run '{run_name}' initialized and one row appended to {hdf5_file}.")