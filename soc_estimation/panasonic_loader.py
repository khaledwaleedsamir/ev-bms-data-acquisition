import os
import numpy as np
import pandas as pd
import scipy.io

C_NOMINAL_AH = 2.9

# All known single-cycle drive cycle identifiers in the Panasonic filenames
CYCLE_TYPES = ['Cycle_1', 'Cycle_2', 'Cycle_3', 'Cycle_4', 'US06', 'HWFTa', 'HWFTb', 'HWFET', 'UDDS', 'LA92', 'NN']

TRAIN_CYCLES = {'Cycle_1', 'Cycle_2', 'Cycle_3', 'US06', 'HWFTa', 'HWFTb', 'HWFET', 'UDDS', 'LA92'}
VAL_CYCLES   = {'Cycle_4'}
TEST_CYCLES  = {'NN'}

# 25degC data lives one subfolder deeper due to the original release structure
_TEMP_DRIVE_CYCLE_DIRS = {
    25:  os.path.join('Panasonic 18650PF Data', '25degC', 'Drive cycles'),
    10:  os.path.join('10degC', 'Drive Cycles'),
     0:  os.path.join('0degC',  'Drive Cycles'),
    -10: os.path.join('-10degC', 'Drive Cycles'),
    -20: os.path.join('-20degC', 'Drive Cycles'),
}


def _get_cycle_type(filename):
    """Return the single drive-cycle keyword found in a filename, or None if 0 or >1 match."""
    matches = [ct for ct in CYCLE_TYPES if ct in filename]
    return matches[0] if len(matches) == 1 else None


def load_mat_file(path, temperature_c):
    """
    Load one Panasonic .mat drive-cycle file.

    SOC is derived from the Ah counter (resets to 0 at start of each test):
        SOC = clip(1 + Ah / C_NOMINAL_AH, 0, 1)
    Ah is negative during discharge, so SOC decreases as the cell discharges.

    Returns a DataFrame with columns:
        Voltage [V], Current [A], Temperature [degC],
        SOC [-], Ah, Time [s], nominal_temp_degC, source_file, cycle_type
    """
    mat  = scipy.io.loadmat(path)
    meas = mat['meas']

    voltage = meas['Voltage'][0, 0].flatten().astype(np.float32)
    current = meas['Current'][0, 0].flatten().astype(np.float32)
    temp    = meas['Battery_Temp_degC'][0, 0].flatten().astype(np.float32)
    ah      = meas['Ah'][0, 0].flatten().astype(np.float32)
    time_s  = meas['Time'][0, 0].flatten().astype(np.float32)

    soc = np.clip(1.0 + ah / C_NOMINAL_AH, 0.0, 1.0).astype(np.float32)

    fname = os.path.basename(path)
    return pd.DataFrame({
        'Voltage [V]':        voltage,
        'Current [A]':        current,
        'Temperature [degC]': temp,
        'Ah':                 ah,
        'Time [s]':           time_s,
        'SOC [-]':            soc,
        'nominal_temp_degC':  temperature_c,
        'source_file':        fname,
        'cycle_type':         _get_cycle_type(fname) or 'unknown',
    })


def load_all_drive_cycles(panasonic_root):
    """
    Load all individual (non-combined) drive-cycle .mat files from every
    temperature sub-folder under panasonic_root.

    Combined files (e.g. US06_HWFET_UDDS_LA92) are automatically skipped
    because they match more than one CYCLE_TYPE keyword.

    Returns:
        list of DataFrames, one per file
    """
    all_dfs = []
    for temp_c, rel_dir in _TEMP_DRIVE_CYCLE_DIRS.items():
        folder = os.path.join(panasonic_root, rel_dir)
        if not os.path.isdir(folder):
            print(f"[WARNING] Directory not found, skipping: {folder}")
            continue
        for fname in sorted(os.listdir(folder)):
            if not fname.endswith('.mat'):
                continue
            cycle_type = _get_cycle_type(fname)
            if cycle_type is None:
                continue
            df = load_mat_file(os.path.join(folder, fname), temp_c)
            all_dfs.append(df)
            print(f"  [{temp_c:>4}°C]  {cycle_type:<8}  {len(df):>8,} samples  {fname}")
    return all_dfs


def split_by_cycle(dfs):
    """
    Split a list of DataFrames into train / val / test by cycle_type.

    Train : Cycle_1, Cycle_2, Cycle_3, US06, HWFET, UDDS, LA92
    Val   : Cycle_4
    Test  : NN  (Neural Network drive cycle — designed to stress SOC estimators)

    Returns:
        train_df, val_df, test_df  (all concatenated)
    """
    train, val, test = [], [], []
    for df in dfs:
        ct = df['cycle_type'].iloc[0]
        if ct in TRAIN_CYCLES:
            train.append(df)
        elif ct in VAL_CYCLES:
            val.append(df)
        elif ct in TEST_CYCLES:
            test.append(df)
    return (
        pd.concat(train, ignore_index=True),
        pd.concat(val,   ignore_index=True),
        pd.concat(test,  ignore_index=True),
    )
