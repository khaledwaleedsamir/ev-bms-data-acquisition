# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a data acquisition and SOC (State of Charge) estimation system for a lithium-ion battery pack (42V, 10.2Ah) coupled to a hoverboard motor load. Data is collected from two hardware sources simultaneously:
- **Daly BMS** via Bluetooth Low Energy (`aiobmsble` + `bleak`)
- **Modified hoverboard controller** via USART serial (TTL-to-USB)

The acquired data feeds into an offline MLP neural network pipeline for SOC estimation.

## Environment Setup

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

Key packages: `aiobmsble`, `bleak`, `h5py`, `pyserial`, `torch`, `PySide6`, `pyqtgraph`, `scikit-learn`, `pandas`, `matplotlib`.

All scripts must be run from the **repository root** so that imports like `from drivers.bms_reader import BMSReader` and `from dataset.dataset_utils import ...` resolve correctly.

## Running Data Collection Scripts

All run scripts live in `dataset/run_scripts/`. Edit the `CONFIGS` section at the top of each script before running — they hardcode COM ports, BMS device name, speed targets, and HDF5 paths.

```bash
# PC-based discharge run (BMS via BLE + hoverboard via serial)
python -m dataset.run_scripts.discharge_run

# PC-based charge run
python -m dataset.run_scripts.charge_run

# Real-time SOC prediction run with live PySide6/pyqtgraph plots
python -m dataset.run_scripts.prediction_run

# ESP32-based discharge run (ESP32 runs MLP on-device, PC logs CSV over serial)
python -m dataset.run_scripts.esp32_discharge_run [--port COM4] [--out output.csv]
```

Common config values to update per run:
- `hdf5_file`: target HDF5 file path
- `run_name`: unique string like `"run_015_80pct_speed_discharge"`
- `hb_com_port`: Windows COM port for hoverboard (e.g., `"COM3"`)
- `bms_name`: Bluetooth device name (e.g., `"EGIKE_STATION_1"`)
- `speed`: fraction of `FULL_SPEED = 580`
- `stop_soc`: SOC percentage at which the run auto-stops

## HDF5 Data Management

HDF5 files are stored in `dataset/all_data/h5_files/`. Structure per run:
```
run_name/
  timestamp_ms        (float64, ms since epoch)
  time_string         (utf-8 string)
  hoverboard/
    hb_speedR_meas, hb_speedL_meas  (int32)
    hb_measured_voltage, hb_board_temp  (float32)
  bms/
    battery_level, voltage, current, power, ...  (float32)
    temp_values  (float32, shape [N, 3])
    cell_voltages  (float32, shape [N, 10])
  attrs: run metadata
```

Utility scripts:
```bash
# Combine multiple HDF5 files
python dataset/all_data/h5_files/combine_h5.py file1.h5 file2.h5 --output combined.h5

# Delete a run from an HDF5 file
python dataset/all_data/h5_files/delete_runs.py

# Downsample an Excel file
python dataset/preprocessing_scripts/downsample.py input.xlsx output.xlsx --ratio 10
```

The `dataset/dataset_utils.py` module is the central API for HDF5 I/O:
- `init_run_dynamic()` — create a new run group based on sample dict shapes
- `append_row()` — append one timestamped sample to all datasets
- `delete_run()` — permanently remove a run group

## SOC Estimation (MLP)

The MLP pipeline is in `soc_estimation/`:

- `soc_estimation/mlp/mlp.py` — `MLP_SOC` (PyTorch `nn.Module`) and `ModelManager` (training, validation, inference with optional `StandardScaler` normalization). The model outputs are sigmoid-constrained to [0, 1]; multiply by 100 for percentage.
- `soc_estimation/dataset_manager.py` — `DatasetManager` (train/val split + scaling) and `H5DatasetHandler` (HDF5-to-DataFrame extraction).
- `soc_estimation/mlp/test_mlp.py` — loads saved weights/scalers, runs inference on an HDF5 run, computes MAE/RMSE/R², saves CSV results.

MLP input features used in production: `[voltage, current, mean(temp_values), cycle_charge]` (4 features) or with `cycle_capacity` added (5 features). Model weights and scalers are saved as `.pth` and `.pkl` (joblib) files respectively in `soc_estimation/mlp/outputs/` (gitignored).

Typical training/inference pattern:
```python
model = MLP_SOC(input_size=4, hidden_sizes=[32, 16], output_size=1)
manager = ModelManager(model, device='cpu')
manager.load_model_weights("outputs/mlp_model.pth")
manager.load_scalers("outputs/scalers.pkl")
pred_soc = manager.predict([voltage, current, temp_mean, cycle_charge])[0] * 100
```

## Architecture Overview

```
drivers/
  bms_reader.py          # BMSReader — async BLE polling in background thread
  hoverboard_controller.py  # HoverboardController — serial TX/RX/print in 3 threads

dataset/
  dataset_utils.py       # HDF5 read/write API (init_run_dynamic, append_row, delete_run)
  run_scripts/           # Experiment entry points (discharge, charge, prediction, ESP32)
  data_analysis/         # Offline analysis: correlation matrices, feature engineering, plots
  preprocessing_scripts/ # Downsampling utilities
  all_data/              # Raw data: h5_files/, csv/, excel_files/

soc_estimation/
  dataset_manager.py     # H5DatasetHandler, DatasetManager (split + scale)
  mlp/
    mlp.py               # MLP_SOC model + ModelManager (train/validate/predict)
    test_mlp.py          # Offline evaluation script
```

**Threading model**: `BMSReader` runs an asyncio event loop in a dedicated thread, protected by a `Lock`. `HoverboardController` spawns separate sender, receiver, and printer threads, each guarded by `Lock`s. The data logger in run scripts is a fourth thread reading from both drivers at 1 Hz.

**Dual data paths**: Runs collected by the PC pipeline go to HDF5. The ESP32 path (`esp32_discharge_run.py` + `esp_bms_logger.py`) logs directly to CSV from the ESP32's UART output, with the MLP running embedded on the ESP32.

## Gitignored Outputs

`.h5`, `.xlsx`, `.csv`, `soc_estimation/mlp/outputs/`, and `output_figures/` are all gitignored. Raw data files must be managed manually or via DVC (`dvc-gdrive` is listed as a dependency).
