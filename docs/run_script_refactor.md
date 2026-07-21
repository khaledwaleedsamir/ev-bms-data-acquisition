# Run Script Refactor Design

## Problems with the Current Setup

1. **Duplicated structure.** `discharge_run.py` and `charge_run.py` are ~85% identical — same init samples, same logger loop skeleton, same HDF5 calls.
2. **Module-level execution.** Hardware is initialized and threads are started at import time, not inside a `if __name__ == "__main__"` guard. Importing the module starts a BLE scan.
3. **Global mutable state.** `last_hb`, `last_bms`, and `stop_flag` are module-level globals mutated from inside `data_logger`. This makes it impossible to run two instances or unit-test the logger.
4. **No graceful shutdown.** There is no `KeyboardInterrupt` handler. Ctrl-C leaves the hoverboard spinning at `speed` and the serial port open.
5. **Commented-out charge stop condition.** The charge run's SOC stop is disabled — the script runs forever.
6. **Config is scattered.** Battery metadata, hardware ports, speed fraction, and HDF5 path all live in different sections of the same file with no schema or validation.
7. **Init samples are duplicated.** Both files copy-paste the same `hb_init_sample` and `bms_init_sample` dicts.

---

## Decisions

| # | Question | Decision |
|---|----------|----------|
| 1 | Config format | **TOML files** — `speed_fraction` and `full_speed` are stored as plain numbers; Python computes `speed = int(full_speed * speed_fraction)` when loading the config |
| 2 | Init samples | **Option C — derive dynamically** from the first live BMS/HB sample after connection |
| 3 | `prediction_run.py` / `esp32_discharge_run.py` | **Out of scope** |
| 4 | Logging verbosity | **Python `logging` module** with configurable levels |

---

## Proposed Design

### File layout

```
dataset/run_scripts/
  run.py                          ← single generic entry point
  run_configs/
    template_discharge.toml       ← copy-and-fill template for discharge runs
    template_charge.toml          ← copy-and-fill template for charge runs
    discharge_80pct_run015.toml   ← example actual run config
    full_charge_run003.toml
```

Run a discharge:
```bash
python -m dataset.run_scripts.run dataset/run_scripts/run_configs/discharge_80pct_run015.toml
```

The old `discharge_run.py` and `charge_run.py` are deleted once the new script is validated.

---

### Config file format (TOML)

TOML stores `speed_fraction` and `full_speed` as plain numbers. The Python loader computes
`speed = int(full_speed * speed_fraction)` when building the `RunConfig` — no expressions
needed in the file itself.

```toml
# run_configs/discharge_80pct_run015.toml

[run]
name        = "run_015_80pct_discharge"
mode        = "discharge"           # "discharge" | "charge"
description = "80% speed discharge with rollers resistance."
stop_soc    = 30.0                  # discharge: stop when SOC <= this value

[hardware]
hdf5_file    = "dataset/all_data/h5_files/hoverboard_bms.h5"
bms_name     = "EGIKE_STATION_1"
hb_com_port  = "COM3"
hb_baud_rate = 115200

[discharge]
full_speed       = 580              # maximum speed value for the hoverboard
speed_fraction   = 0.8             # fraction of full_speed to target
# computed in Python: speed = int(full_speed * speed_fraction) → 464

[logging]
log_hz = 1

[metadata]
battery_pack = "Lithium-Ion 42V 10.2Ah"
battery_age  = "new"
```

```toml
# run_configs/full_charge_run003.toml

[run]
name        = "run_full_charge_003"
mode        = "charge"
description = "Charging to 100% SOC after HPPC run."
stop_soc    = 100.0                 # charge: stop when SOC >= this value

[hardware]
hdf5_file    = "dataset/all_data/h5_files/hoverboard_bms_HPPC_data.h5"
bms_name     = "EGIKE_STATION_1"
hb_com_port  = ""                  # empty string = no hoverboard
hb_baud_rate = 115200

[logging]
log_hz = 1

[metadata]
battery_pack = "Lithium-Ion 42V 10.2Ah"
battery_age  = "new"
```

The `[discharge]` section is simply absent in charge configs. The loader treats a missing
section (or an empty `hb_com_port`) as "charge-only mode" and skips hoverboard
initialisation.

---

### `RunConfig` dataclass

Lives in `run.py`. Validated at construction time before any hardware is touched.

```
RunConfig
  name:           str
  mode:           Literal["discharge", "charge"]
  description:    str
  stop_soc:       float
  hdf5_file:      str
  bms_name:       str
  hb_com_port:    str | None      ← None means "charge mode, no hoverboard"
  hb_baud_rate:   int
  speed:          int | None      ← computed: int(full_speed * speed_fraction); None for charge
  log_hz:         int
  metadata:       dict[str, str]
```

`RunConfig` is never constructed directly by the user — only by `RunConfig.from_toml(path)`.
That class method reads the TOML, computes `speed = int(full_speed * speed_fraction)` from
the `[discharge]` section if present, and sets `speed = None` and `hb_com_port = None` for
charge configs.

`__post_init__` validation rules:
- If `mode == "discharge"`: `speed` must not be `None`, `hb_com_port` must not be `None` or empty.
- If `mode == "charge"`: `speed` and `hb_com_port` are forced to `None` regardless of what the file contains.
- `log_hz` must be >= 1.
- `stop_soc` must be in [0, 100].

---

### `ExperimentRunner` class

All mutable state that currently lives as module-level globals moves into this class. The public interface is a single method: `run()`.

```
ExperimentRunner
  __init__(config: RunConfig)
  run()               ← public entry point; calls _setup → _loop → _teardown in try/finally
  _setup()            ← start BMS + hoverboard, wait for first samples, init HDF5 run
  _loop()             ← timed logger loop, blocks until stop_flag is set
  _teardown()         ← ramp speed to 0, stop all threads, log run summary
  _log_one_sample()   ← read drivers, hold-last on None, append_row, check stop condition
```

**Instance state (no globals):**
```
self.stop_flag:   threading.Event
self.last_hb:     dict | None
self.last_bms:    dict | None
self.hoverboard:  HoverboardController | None
self.bms_reader:  BMSReader
self.samples_logged: int
self.start_time:  float
self.logger:      logging.Logger
```

**Stop condition** is set once at `_setup()` time as a callable:
```python
# discharge
self._should_stop = lambda soc: soc <= config.stop_soc
# charge
self._should_stop = lambda soc: soc >= config.stop_soc
```

**`_loop()` timing** uses `time.sleep` aligned to `sample_interval = 1.0 / config.log_hz`. The loop accounts for drift by measuring actual elapsed time and adjusting sleep accordingly, so the logging rate stays accurate over long runs.

---

### Dynamic init samples (Decision 2 — Option C)

Instead of hardcoded dummy dicts, `_setup()` waits for the first live sample from each driver and uses those dicts as the HDF5 schema template. This means:

1. Start BMS reader, wait until `bms_reader.get_latest()` is not `None`.
2. If discharge mode: start hoverboard threads, wait until `hoverboard.get_feedback()` is not `None`.
3. Call `init_run_dynamic(hdf5_file, run_name, metadata, hb_sample, bms_sample)` with the real first samples.

This eliminates the duplicated hardcoded init dicts entirely and ensures the HDF5 schema always matches what the hardware actually reports. If the BMS firmware version changes its fields, the schema adapts automatically.

---

### Logging setup (Decision 4)

A single logger named `"experiment"` is configured in `run.py` at startup. Two handlers:
- `StreamHandler` to stdout — level `INFO` by default, switchable to `DEBUG` via `--verbose` CLI flag.
- Optionally a `FileHandler` writing to `run_name.log` alongside the HDF5 file.

Log levels used:
- `DEBUG` — raw sample dicts every tick (replaces the current `print(hb_dict)` / `print(bms_dict)`)
- `INFO` — connection events, ramp complete, SOC milestones (every 10%), run start/stop
- `WARNING` — missed sample (driver returned `None`, using last-known value)
- `ERROR` — hardware failure, HDF5 write error

```
[2026-06-24 14:32:01] INFO     BMS connected (EGIKE_STATION_1)
[2026-06-24 14:32:03] INFO     Hoverboard connected on COM3
[2026-06-24 14:32:04] INFO     Run "run_015_80pct_discharge" started — stop at SOC <= 30.0%
[2026-06-24 14:32:04] INFO     Hoverboard ramped to speed 464
[2026-06-24 14:45:10] INFO     SOC milestone: 80%
[2026-06-24 15:02:44] INFO     SOC milestone: 70%
[2026-06-24 15:02:44] WARNING  HB feedback is None — using last-known values
...
[2026-06-24 17:11:03] INFO     Stop condition reached: SOC = 29.8% <= 30.0%
[2026-06-24 17:11:03] INFO     Run complete. Samples logged: 10739 | Elapsed: 2h 39m 00s
```

---

### Graceful shutdown sequence

Triggered by either the SOC stop condition or `KeyboardInterrupt`. Always runs inside `finally`:

1. Set `stop_flag` — `_loop()` exits after its current sample.
2. If discharge mode: `hoverboard.ramp_speed(0)` → `hoverboard.close()`.
3. `bms_reader.stop()`.
4. Log run summary: name, elapsed time, final SOC, total samples logged.

---

### Entry point (`run.py` `__main__` block)

```python
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generic BMS/hoverboard experiment runner")
    parser.add_argument("config", help="Path to a run config .toml file")
    parser.add_argument("--verbose", action="store_true", help="Set log level to DEBUG")
    args = parser.parse_args()

    config = RunConfig.from_toml(args.config)
    ExperimentRunner(config, verbose=args.verbose).run()
```

`tomllib` is in the Python standard library from 3.11 onward. For older versions, the
`tomli` backport (`pip install tomli`) is a drop-in replacement.
