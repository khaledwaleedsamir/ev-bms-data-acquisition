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
        self._thread = Thread(target=self.run_loop, daemon=True)
        self._thread.start()

    def run_loop(self):
        asyncio.run(self._async_loop())

    def stop(self):
        self._stop_flag = True
        if self._thread:
            self._thread.join()
    
    def get_latest(self) -> BMSSample | None:
        with self._lock:
            return self.latest_sample

