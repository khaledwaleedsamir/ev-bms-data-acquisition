"""
Generic BMS/hoverboard experiment runner.

Usage:
    python -m dataset.scripts.run <config.toml> [--verbose]
"""
from __future__ import annotations

import argparse
import logging
import math
import threading
import time
import tomllib
from dataclasses import dataclass
from typing import Literal

from dataset.dataset_utils import append_row, get_timestamp, get_time_string, init_run_dynamic
from drivers.bms_reader import BMSReader
from drivers.hoverboard_controller import HoverboardController

_ZERO_HB_SAMPLE: dict = {
    "hb_speedR_meas": 0,
    "hb_speedL_meas": 0,
    "hb_measured_voltage": 0.0,
    "hb_board_temp": 0.0,
}

_SOC_MILESTONE_STEP = 10  # log an INFO message every N% SOC change


@dataclass
class RunConfig:
    name: str
    mode: Literal["discharge", "charge"]
    description: str
    stop_soc: float
    hdf5_file: str
    bms_name: str
    hb_com_port: str | None
    hb_baud_rate: int
    speed: int | None
    log_hz: int
    metadata: dict

    def __post_init__(self):
        if self.mode == "discharge":
            if not self.hb_com_port:
                raise ValueError("discharge mode requires a non-empty hb_com_port")
            if self.speed is None:
                raise ValueError("discharge mode requires a [discharge] section in the config file")
        else:
            self.speed = None
            self.hb_com_port = None

        if self.log_hz < 1:
            raise ValueError(f"log_hz must be >= 1, got {self.log_hz}")
        if not (0.0 <= self.stop_soc <= 100.0):
            raise ValueError(f"stop_soc must be in [0, 100], got {self.stop_soc}")

    @classmethod
    def from_toml(cls, path: str) -> RunConfig:
        with open(path, "rb") as f:
            data = tomllib.load(f)

        run = data["run"]
        hw = data["hardware"]
        log = data.get("logging", {})
        meta = data.get("metadata", {})

        speed = None
        hb_com_port = hw.get("hb_com_port") or None

        if run["mode"] == "discharge":
            dis = data["discharge"]
            speed = int(dis["full_speed"] * dis["speed_fraction"])

        return cls(
            name=run["name"],
            mode=run["mode"],
            description=run.get("description", ""),
            stop_soc=float(run["stop_soc"]),
            hdf5_file=hw["hdf5_file"],
            bms_name=hw["bms_name"],
            hb_com_port=hb_com_port,
            hb_baud_rate=int(hw.get("hb_baud_rate", 115200)),
            speed=speed,
            log_hz=int(log.get("log_hz", 1)),
            metadata=dict(meta),
        )


class ExperimentRunner:
    def __init__(self, config: RunConfig, verbose: bool = False):
        self.config = config
        self.stop_flag = threading.Event()
        self.last_hb: dict = _ZERO_HB_SAMPLE.copy()
        self.last_bms: dict | None = None
        self.hoverboard: HoverboardController | None = None
        self.bms_reader: BMSReader | None = None
        self.samples_logged: int = 0
        self.start_time: float = 0.0
        self._next_milestone: float = 0.0

        self.logger = logging.getLogger("experiment")
        level = logging.DEBUG if verbose else logging.INFO
        handler = logging.StreamHandler()
        handler.setLevel(level)
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        self.logger.setLevel(level)
        self.logger.addHandler(handler)

        if config.mode == "discharge":
            self._should_stop = lambda bms: bms.get("battery_level", 100.0) <= config.stop_soc
        elif config.mode == "charge" and config.stop_soc >= 100.0:
            # Full charge: SOC reaching 100% is the CC→CV transition, not the end.
            # Wait for the BMS to cut off the charger (current drops to 0).
            self._should_stop = lambda bms: bms.get("current", 1.0) == 0.0
        else:
            self._should_stop = lambda bms: bms.get("battery_level", 0.0) >= config.stop_soc

    def run(self):
        try:
            self._setup()
            self._loop()
        except KeyboardInterrupt:
            self.logger.info("KeyboardInterrupt received — shutting down.")
        finally:
            self._teardown()

    def _setup(self):
        cfg = self.config

        # Start BMS and wait for first sample
        self.bms_reader = BMSReader(device_name=cfg.bms_name)
        self.bms_reader.start()
        self.logger.info("Waiting for BMS connection (%s)...", cfg.bms_name)
        while self.bms_reader.get_latest() is None:
            time.sleep(1)
        self.logger.info("BMS connected (%s)", cfg.bms_name)
        first_bms = self.bms_reader.get_latest()
        self.last_bms = first_bms

        # Start hoverboard and wait for first feedback (discharge only)
        first_hb = _ZERO_HB_SAMPLE.copy()
        if cfg.mode == "discharge":
            self.hoverboard = HoverboardController(
                serial_port=cfg.hb_com_port,
                baud_rate=cfg.hb_baud_rate,
                print_feedback=False,
            )
            self.hoverboard.start_threads()
            self.logger.info("Waiting for hoverboard feedback on %s...", cfg.hb_com_port)
            while self.hoverboard.get_feedback() is None:
                time.sleep(0.5)
            self.logger.info("Hoverboard connected on %s", cfg.hb_com_port)
            first_hb = self.hoverboard.get_feedback()
            self.last_hb = first_hb

        # Initialize HDF5 run schema from first live samples
        run_metadata = {"description": cfg.description, "mode": cfg.mode, **cfg.metadata}
        init_run_dynamic(cfg.hdf5_file, cfg.name, run_metadata, first_hb, first_bms)
        self.logger.info(
            'Run "%s" started — stop at SOC %s %.1f%%',
            cfg.name,
            "<=" if cfg.mode == "discharge" else ">=",
            cfg.stop_soc,
        )

        # Ramp hoverboard after HDF5 init so no data is missed
        if cfg.mode == "discharge":
            self.hoverboard.ramp_speed(cfg.speed)
            self.logger.info("Hoverboard ramped to speed %d", cfg.speed)

        # Set first SOC milestone based on actual starting SOC
        soc0 = first_bms.get("battery_level", 0.0) if first_bms else 0.0
        if cfg.mode == "discharge":
            self._next_milestone = (int(soc0) // _SOC_MILESTONE_STEP) * _SOC_MILESTONE_STEP
            if self._next_milestone >= soc0:
                self._next_milestone -= _SOC_MILESTONE_STEP
        else:
            self._next_milestone = math.ceil(soc0 / _SOC_MILESTONE_STEP) * _SOC_MILESTONE_STEP
            if self._next_milestone <= soc0:
                self._next_milestone += _SOC_MILESTONE_STEP

        self.start_time = time.time()

    def _loop(self):
        sample_interval = 1.0 / self.config.log_hz
        while not self.stop_flag.is_set():
            self._log_one_sample()
            time.sleep(sample_interval)

    def _log_one_sample(self):
        timestamp = get_timestamp()
        time_string = get_time_string()

        # Hoverboard (hold last-known on None; zero dict for charge runs)
        if self.hoverboard is not None:
            hb_feedback = self.hoverboard.get_feedback()
            if hb_feedback is not None:
                self.last_hb = hb_feedback
            else:
                self.logger.warning("HB feedback is None — using last-known values")
        hb_dict = self.last_hb

        # BMS (hold last-known on None)
        bms_sample = self.bms_reader.get_latest()
        if bms_sample is not None:
            self.last_bms = bms_sample
        else:
            self.logger.warning("BMS sample is None — using last-known values")
        bms_dict = self.last_bms

        self.logger.debug("HB: %s", hb_dict)
        self.logger.debug("BMS: %s", bms_dict)

        append_row(self.config.hdf5_file, self.config.name, timestamp, time_string, hb_dict, bms_dict)
        self.samples_logged += 1

        # Full status block printed every sample tick
        bms = bms_dict or {}
        hb = hb_dict or {}
        temp_vals = bms.get("temp_values") or []
        temps_str = " / ".join(f"{t:.1f}" for t in temp_vals) if temp_vals else "—"
        cell_volts = bms.get("cell_voltages") or []
        cells_str = "  ".join(f"{v:.2f}" for v in cell_volts) if cell_volts else "—"
        lines = [
            f"──── [{time_string}] Sample #{self.samples_logged} {'─' * 30}",
            f"BMS │ SOC: {bms.get('battery_level', 0.0):5.1f}%  V: {bms.get('voltage', 0.0):5.2f}V  I: {bms.get('current', 0.0):6.2f}A  P: {bms.get('power', 0.0):7.1f}W  Charging: {'Yes' if bms.get('battery_charging') else 'No'}",
            f"    │ Temps: {temps_str} °C  ΔV: {bms.get('delta_voltage', 0.0):.3f}V  Cycles: {bms.get('cycles', 0)}",
            f"    │ CycleCharge: {bms.get('cycle_charge', 0.0):.2f}Ah  CycleCap: {bms.get('cycle_capacity', 0.0):.2f}Ah",
            f"    │ Cells: {cells_str} V",
        ]
        if self.config.mode == "discharge":
            lines.append(
                f" HB │ Speed R/L: {hb.get('hb_speedR_meas', 0)}/{hb.get('hb_speedL_meas', 0)}"
                f"  V: {hb.get('hb_measured_voltage', 0.0):.1f}V  Temp: {hb.get('hb_board_temp', 0.0):.1f}°C"
            )
        lines.append("─" * 67)
        self.logger.info("\n".join(lines))

        # SOC milestone logging
        soc = bms_dict.get("battery_level", 0.0) if bms_dict else 0.0
        if self.config.mode == "discharge":
            if soc <= self._next_milestone:
                self.logger.info("SOC milestone: %.0f%%", self._next_milestone)
                self._next_milestone -= _SOC_MILESTONE_STEP
        else:
            if soc >= self._next_milestone:
                self.logger.info("SOC milestone: %.0f%%", self._next_milestone)
                self._next_milestone += _SOC_MILESTONE_STEP

        # Stop condition check
        if bms_dict and self._should_stop(bms_dict):
            if self.config.mode == "charge" and self.config.stop_soc >= 100.0:
                self.logger.info(
                    "Stop condition reached: BMS current = %.2fA (charger cutoff)",
                    bms_dict.get("current", 0.0),
                )
            else:
                self.logger.info(
                    "Stop condition reached: SOC = %.1f%% %s %.1f%%",
                    soc,
                    "<=" if self.config.mode == "discharge" else ">=",
                    self.config.stop_soc,
                )
            self.stop_flag.set()

    def _teardown(self):
        self.stop_flag.set()

        if self.hoverboard is not None:
            self.logger.info("Ramping hoverboard to 0...")
            self.hoverboard.ramp_speed(0)
            self.hoverboard.close()

        if self.bms_reader is not None:
            self.bms_reader.stop()

        elapsed = time.time() - self.start_time if self.start_time else 0.0
        h, rem = divmod(int(elapsed), 3600)
        m, s = divmod(rem, 60)
        self.logger.info(
            "Run complete. Samples logged: %d | Elapsed: %dh %02dm %02ds",
            self.samples_logged,
            h, m, s,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generic BMS/hoverboard experiment runner")
    parser.add_argument("config", help="Path to a run config .toml file")
    parser.add_argument("--verbose", action="store_true", help="Set log level to DEBUG")
    args = parser.parse_args()

    cfg = RunConfig.from_toml(args.config)
    ExperimentRunner(cfg, verbose=args.verbose).run()
