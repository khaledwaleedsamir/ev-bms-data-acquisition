"""
esp_bms_logger.py
─────────────
Reads CSV lines sent by the ESP32 over USB-Serial and logs them to a CSV file.

ESP32 CSV line format:
  CSV,<timestamp_ms>,<voltage>,<current>,<temperature>,
      <cycle_charge>,<cycle_capacity>,<bms_soc>,<pred_soc>,<inference_us>

Usage:
  pip install pyserial
  python bms_logger.py                       # uses defaults below
  python bms_logger.py --port COM5           # Windows
  python bms_logger.py --port /dev/ttyUSB0   # Linux
  python bms_logger.py --port /dev/cu.usbserial-0001 --baud 115200 --out my_log.csv
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    sys.exit(
        "pyserial not found. Install it with:  pip install pyserial"
    )

# ── Configuration defaults ────────────────────────────────────────────────────
DEFAULT_BAUD    = 115200
DEFAULT_OUT     = "charge_run-002_esp.csv"
CSV_MARKER      = "CSV"          # lines that start with this are data rows
RECONNECT_DELAY = 5              # seconds to wait before retrying after disconnect

CSV_HEADER = [
    "datetime_utc",
    "esp_timestamp_ms",
    "voltage_V",
    "current_A",
    "temperature_degC",
    "cycle_charge_Ah",
    "cycle_capacity_Wh",
    "bms_soc_pct",
    "pred_soc_pct",
    "inference_us",
    "esp_temp_degC",
]


def auto_detect_port() -> str | None:
    """Return the first USB-serial port that looks like an ESP32."""
    esp_keywords = ("cp210", "ch340", "ch341", "ftdi", "esp", "usb serial", "uart")
    ports = serial.tools.list_ports.comports()
    for p in ports:
        desc = (p.description + " " + (p.manufacturer or "")).lower()
        if any(kw in desc for kw in esp_keywords):
            return p.device
    # fallback: return the first available port
    return ports[0].device if ports else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ESP32 BMS serial logger")
    parser.add_argument(
        "--port", "-p",
        default=None,
        help="Serial port (e.g. COM5 or /dev/ttyUSB0). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--baud", "-b",
        type=int,
        default=DEFAULT_BAUD,
        help=f"Baud rate (default: {DEFAULT_BAUD})",
    )
    parser.add_argument(
        "--out", "-o",
        default=DEFAULT_OUT,
        help=f"Output CSV file path (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--no-print",
        action="store_true",
        help="Suppress printing each row to the terminal",
    )
    return parser.parse_args()


def open_csv(path: str) -> tuple[csv.DictWriter, object]:
    """Open (or append to) the CSV file; write header if the file is new."""
    file_exists = Path(path).exists() and Path(path).stat().st_size > 0
    fh = open(path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=CSV_HEADER)
    if not file_exists:
        writer.writeheader()
        fh.flush()
        print(f"[logger] Created {path}")
    else:
        print(f"[logger] Appending to existing {path}")
    return writer, fh


def parse_csv_line(line: str) -> dict | None:
    """
    Parse a CSV line from the ESP32.
    Returns a dict ready to write, or None if the line is not a data row.
    """
    parts = line.strip().split(",")
    # Expected: CSV, ts, V, I, T, CC, Cap, bms_soc, pred_soc, inf_us  → 10 fields
    if len(parts) != 11 or parts[0] != CSV_MARKER:
        return None
    try:
        return {
            "datetime_utc":       datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "esp_timestamp_ms":   int(parts[1]),
            "voltage_V":          float(parts[2]),
            "current_A":          float(parts[3]),
            "temperature_degC":   float(parts[4]),
            "cycle_charge_Ah":    float(parts[5]),
            "cycle_capacity_Wh":  float(parts[6]),
            "bms_soc_pct":        float(parts[7]),
            "pred_soc_pct":       float(parts[8]),
            "inference_us":       int(parts[9]),
            "esp_temp_degC":      float(parts[10]),
        }
    except ValueError:
        return None


def run(port: str, baud: int, out: str, silent: bool) -> None:
    writer, fh = open_csv(out)
    row_count = 0

    print(f"[logger] Connecting to {port} @ {baud} baud …")
    while True:
        try:
            with serial.Serial(port, baud, timeout=2) as ser:
                print(f"[logger] Connected. Waiting for data (Ctrl+C to stop) …\n")
                while True:
                    raw = ser.readline()
                    if not raw:
                        continue                    # timeout, keep polling
                    try:
                        line = raw.decode("utf-8", errors="replace")
                    except Exception:
                        continue

                    row = parse_csv_line(line)
                    if row is None:
                        # Print non-CSV debug lines from the ESP32 as-is
                        if not silent:
                            print(f"[ESP32] {line.rstrip()}")
                        continue

                    writer.writerow(row)
                    fh.flush()
                    row_count += 1

                    if not silent:
                        print(
                            f"[{row['datetime_utc']}] "
                            f"V={row['voltage_V']:.1f}V  "
                            f"I={row['current_A']:.2f}A  "
                            f"T={row['temperature_degC']:.1f}°C  "
                            f"CChg={row['cycle_charge_Ah']:.1f}Ah  "
                            f"CCap={row['cycle_capacity_Wh']:.1f}Wh  "
                            f"BMS={row['bms_soc_pct']:.1f}%  "
                            f"MLP={row['pred_soc_pct']:.2f}%  "
                            f"t={row['inference_us']}µs  "
                            f"T={row['esp_temp_degC']:.1f}°C  "
                            f"(row #{row_count})"
                        )

        except serial.SerialException as exc:
            print(f"\n[logger] Serial error: {exc}")
            print(f"[logger] Retrying in {RECONNECT_DELAY} s …")
            try:
                fh.flush()
            except Exception:
                pass
            time.sleep(RECONNECT_DELAY)

        except KeyboardInterrupt:
            print(f"\n[logger] Stopped. {row_count} rows written to {out}")
            fh.close()
            break


def main() -> None:
    args = parse_args()

    port = args.port
    if port is None:
        port = auto_detect_port()
        if port is None:
            sys.exit("[logger] No serial port found. Plug in the ESP32 or use --port.")
        print(f"[logger] Auto-detected port: {port}")

    run(port=port, baud=args.baud, out=args.out, silent=args.no_print)


if __name__ == "__main__":
    main()
