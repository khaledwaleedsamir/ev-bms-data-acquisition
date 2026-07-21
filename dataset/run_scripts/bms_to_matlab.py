"""
Streams live BMS data to MATLAB over a TCP connection (newline-delimited JSON).

Usage
-----
1. Edit the CONFIGS section below.
2. Run from the repo root:
       python -m dataset.run_scripts.bms_to_matlab
3. In MATLAB, run  bms_receiver.m  — it will connect to this server.

Each JSON line sent to MATLAB has the full BMS dict fields plus a Unix
timestamp so MATLAB can align samples to wall-clock time.
"""

import socket
import json
import time
import logging

from drivers.bms_reader import BMSReader

######################################## CONFIGS ########################################

BMS_NAME = "EGIKE_STATION_1"   # BLE device name
TCP_HOST = "0.0.0.0"           # listen on all interfaces (127.0.0.1 for localhost only)
TCP_PORT = 5005                 # MATLAB connects to this port
LOG_HZ   = 1                    # samples per second to send

######################################## END OF CONFIGS ########################################


class _BMSEncoder(json.JSONEncoder):
    """Handles numpy arrays/scalars and iterables that json can't serialise natively."""
    def default(self, obj):
        if hasattr(obj, "tolist"):       # numpy ndarray / scalar
            return obj.tolist()
        if isinstance(obj, (tuple,)):    # tuple → list
            return list(obj)
        return super().default(obj)


def _sanitize(d: dict) -> dict:
    """
    Replace None values with 0 / zero-arrays so MATLAB jsondecode
    always gets a consistent numeric struct with no missing fields.
    """
    out = {}
    for k, v in d.items():
        if v is None:
            out[k] = 0
        elif isinstance(v, (list, tuple)):
            out[k] = [0 if x is None else float(x) if isinstance(x, (int, float)) else x for x in v]
        elif isinstance(v, bool):
            out[k] = int(v)             # True/False → 1/0 for Simulink compatibility
        else:
            out[k] = v
    return out


def _run_server(bms_reader: BMSReader, log: logging.Logger) -> None:
    interval = 1.0 / LOG_HZ

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((TCP_HOST, TCP_PORT))
        srv.listen(1)
        log.info("TCP server listening on %s:%d — waiting for MATLAB…", TCP_HOST, TCP_PORT)

        while True:
            conn, addr = srv.accept()
            log.info("MATLAB connected from %s", addr)
            last_bms: dict | None = None

            try:
                with conn:
                    while True:
                        t0 = time.monotonic()

                        sample = bms_reader.get_latest()
                        if sample is not None:
                            last_bms = sample

                        if last_bms is not None:
                            payload = _sanitize(last_bms)
                            payload["timestamp"] = time.time()          # Unix seconds
                            line = json.dumps(payload, cls=_BMSEncoder) + "\n"
                            conn.sendall(line.encode("utf-8"))

                        elapsed = time.monotonic() - t0
                        time.sleep(max(0.0, interval - elapsed))

            except (BrokenPipeError, ConnectionResetError, OSError) as exc:
                log.warning("MATLAB disconnected (%s). Waiting for reconnect…", exc)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    log = logging.getLogger("bms_to_matlab")

    bms_reader = BMSReader(device_name=BMS_NAME)
    bms_reader.start()
    log.info("BMS Reader started for device '%s'.", BMS_NAME)

    log.info("Waiting for first BMS sample over BLE…")
    while bms_reader.get_latest() is None:
        time.sleep(0.5)
    log.info("BMS connected. Launching TCP server on port %d.", TCP_PORT)

    try:
        _run_server(bms_reader, log)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt — stopping.")
    finally:
        bms_reader.stop()
        log.info("BMS Reader stopped.")


if __name__ == "__main__":
    main()
