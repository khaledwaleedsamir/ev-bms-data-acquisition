import argparse
import csv
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
DEFAULT_OUT     = "charge_cell_voltages.csv"
CSV_MARKER      = "CSV"
RECONNECT_DELAY = 5

CSV_HEADER = [
    "datetime_utc",
    "esp_timestamp_ms",
    "temperature_degC",
    "pack_voltage_V",
    "cell_10_V",
    "cell_9_V",
    "cell_8_V",
    "cell_7_V",
    "cell_6_V",
    "cell_5_V",
    "cell_4_V",
    "cell_3_V",
    "cell_2_V",
    "cell_1_V",
    "loop_time_ms",
]


def auto_detect_port() -> str | None:
    """Return the first USB-serial port that looks like an ESP32."""
    esp_keywords = ("cp210", "ch340", "ch341", "ftdi", "esp", "usb serial", "uart")
    ports = serial.tools.list_ports.comports()
    for p in ports:
        desc = (p.description + " " + (p.manufacturer or "")).lower()
        if any(kw in desc for kw in esp_keywords):
            return p.device
    return ports[0].device if ports else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ESP32 BMS serial logger (charge)")
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
    Parse a CSV line from the ESP32 cell-voltage monitor.
    Expected format (15 fields):
      CSV,<millis>,<tempC>,<pack_V>,<cell10_V>,...,<cell1_V>,<loop_ms>
    """
    parts = line.strip().split(",")
    if len(parts) != 15 or parts[0] != CSV_MARKER:
        return None
    try:
        return {
            "datetime_utc":     datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "esp_timestamp_ms": int(parts[1]),
            "temperature_degC": float(parts[2]),
            "pack_voltage_V":   float(parts[3]),
            "cell_10_V":        float(parts[4]),
            "cell_9_V":         float(parts[5]),
            "cell_8_V":         float(parts[6]),
            "cell_7_V":         float(parts[7]),
            "cell_6_V":         float(parts[8]),
            "cell_5_V":         float(parts[9]),
            "cell_4_V":         float(parts[10]),
            "cell_3_V":         float(parts[11]),
            "cell_2_V":         float(parts[12]),
            "cell_1_V":         float(parts[13]),
            "loop_time_ms":     int(parts[14]),
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
                        continue
                    try:
                        line = raw.decode("utf-8", errors="replace")
                    except Exception:
                        continue

                    row = parse_csv_line(line)
                    if row is None:
                        if not silent:
                            print(f"[ESP32] {line.rstrip()}")
                        continue

                    writer.writerow(row)
                    fh.flush()
                    row_count += 1

                    if not silent:
                        cells = [row[f"cell_{i}_V"] for i in range(1, 11)]
                        print(
                            f"[{row['datetime_utc']}] "
                            f"Pack={row['pack_voltage_V']:.3f}V  "
                            f"T={row['temperature_degC']:.1f}°C  "
                            f"min={min(cells):.3f}V  max={max(cells):.3f}V  "
                            f"loop={row['loop_time_ms']}ms  "
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
