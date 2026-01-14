import asyncio
import logging
from typing import Final
from threading import Thread, Lock

from bleak import BleakScanner
from bleak.backends.device import BLEDevice
from bleak.exc import BleakError

from aiobmsble import BMSSample
from aiobmsble.bms.daly_bms import BMS

class BMSReader:
    def __init__(self, device_name: str):
        self.device_name = device_name
        self.latest_sample: BMSSample | None = None
        self._lock = Lock()
        self._stop_flag = False
        self._thread: Thread | None = None

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("BMSReader")

    async def _async_loop(self):
        """Async loop to discover BLE device and read continuously."""
        device: BLEDevice | None = await BleakScanner.find_device_by_name(self.device_name)
        if device is None:
            self.logger.error("Device '%s' not found.", self.device_name)
            return

        try:
            async with BMS(ble_device=device) as bms:
                self.logger.info("Connected to BMS: %s", device.address)
                while not self._stop_flag:
                    data: BMSSample = await bms.async_update()
                    with self._lock:
                        self.latest_sample = data
                    await asyncio.sleep(0.5) # Polling interval
        except BleakError as ex:
            self.logger.error("Failed to update BMS: %s", type(ex).__name__)
    
    def start(self):
        self._stop_flag = False
        self._thread = Thread(target=self.run_loop)
        self._thread.start()

    def run_loop(self):
        asyncio.run(self._async_loop())

    def stop(self):
        self._stop_flag = True
        if self._thread:
            self._thread.join()
    
    def get_latest(self) -> BMSSample | None:
        """Return the latest BMS sample with only the fields needed for logging."""
        with self._lock:
            if self.latest_sample is None:
                return None
            return {
                "battery_charging": self.latest_sample.get("battery_charging"),
                "battery_level": self.latest_sample.get("battery_level"),
                "voltage": self.latest_sample.get("voltage"),
                "current": self.latest_sample.get("current"),
                "cycle_charge": self.latest_sample.get("cycle_charge"),
                "temp_sensors": self.latest_sample.get("temp_sensors"),
                "temp_values": self.latest_sample.get("temp_values"),
                "power": self.latest_sample.get("power"),
                "cycle_capacity": self.latest_sample.get("cycle_capacity"),
                "cycles": self.latest_sample.get("cycles"),
                "delta_voltage": self.latest_sample.get("delta_voltage"),
                "temperature": self.latest_sample.get("temperature"),
                "cell_count": self.latest_sample.get("cell_count"),
                "cell_voltages": self.latest_sample.get("cell_voltages"),

            }

