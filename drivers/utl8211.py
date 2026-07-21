"""
UTL8211+ DC Electronic Load — RS-232 (SCPI) control library.

Reference: UTL8200+ Series DC Electronic Loads - Programming Manual (V1.1).

Serial defaults for the UTL8200+ series
----------------------------------------
    Baud    : 9600 (also supports 19200/38400/57600/115200 -- must match the
              value set on the instrument under System Config -> Baud Rate)
    Framing : 8 data bits, No parity, 1 stop bit (8N1)
    Wiring  : DB9 female on the load, uses pins 2(TXD), 3(RXD), 5(GND)
    Terminator: '\\n' (0x0A) on every command and response

Quick start
-----------
    from utl8211 import UTL8211, Mode

    with UTL8211() as load:              # auto-detects the COM port
        print(load.idn())
        load.set_mode(Mode.CURRENT)      # CC mode
        load.set_current_protection(6)   # OCP a bit above your target
        load.set_current(5.0)            # 5 A setpoint
        load.input_on()
        print(load.measure_all())
        load.input_on(False)

Driving a time-varying profile (e.g. a drive cycle) is just a Python loop
that calls set_current()/set_power()/etc. on a timer -- see set_setpoint()
below for a mode-agnostic way to do that from a table of (mode, value) pairs.

Every write() call is fire-and-forget; the instrument does not ACK plain
commands. If you need to confirm a command was accepted, follow it with
check_errors(), which reads back SYSTem:ERRor? and raises on anything other
than "no error".
"""

import re
import time
import serial
import serial.tools.list_ports


class Mode:
    """SCPI parameter values for [SOURce:]FUNCtion / MODE. Pass these (or
    their long forms, e.g. "CURRent") to UTL8211.set_mode()."""
    CURRENT = "CURR"      # constant current (CC)
    VOLTAGE = "VOLT"      # constant voltage (CV)
    POWER = "POW"         # constant power (CP)
    RESISTANCE = "RES"    # constant resistance (CR)
    DYNAMIC = "DYN"       # dynamic (on-instrument two-level switching)
    BATTERY = "BAT"       # battery discharge test
    LIST = "LIST"         # on-instrument step-list sequencing


class UTL8211Error(RuntimeError):
    """Raised when the instrument reports a SCPI error via SYSTem:ERRor?."""


class UTL8211:
    """Driver for a single UTL8211+ (or other UTL8200+ series) DC electronic
    load connected over RS-232."""

    def __init__(self, port=None, baudrate=9600, timeout=1.0, address=None):
        """
        port     : COM port name (e.g. 'COM8'). If None, auto-detects the
                   first likely USB-serial adapter found.
        baudrate : must match the instrument's System Config -> Baud Rate.
        timeout  : read/write timeout in seconds for the serial port.
        address  : optional RS485 device address (1..255) used to prefix
                   every command as 'ADDR <address>::<cmd>'. Leave None for
                   a single directly-connected (RS232) load.
        """
        if port is None:
            port = self.autodetect_port()
            if port is None:
                raise IOError(
                    "No serial port found. Plug in the USB-to-RS232 adapter and "
                    "install its driver, then check Device Manager -> Ports (COM & LPT)."
                )
        self.address = address
        self.ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=timeout,
            write_timeout=timeout,
        )
        # give the adapter a moment to settle after opening
        time.sleep(0.1)
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

    # ---- connection lifecycle ------------------------------------------------
    def close(self):
        """Close the serial port. Safe to call multiple times."""
        try:
            self.ser.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @staticmethod
    def autodetect_port():
        """Return the device name of the first COM port that looks like a
        USB-serial adapter (CH340/CP210x/FTDI/Prolific/etc.), or the first
        port found if none match, or None if no ports are present."""
        ports = list(serial.tools.list_ports.comports())
        if not ports:
            return None
        keywords = ("serial", "uart", "ch340", "cp210", "ftdi", "prolific", "pl2303")
        for p in ports:
            desc = (p.description or "").lower()
            if any(k in desc for k in keywords):
                return p.device
        return ports[0].device

    @staticmethod
    def list_ports():
        """Return the raw list of serial.tools.list_ports ListPortInfo objects
        currently visible to the OS, for diagnostics/port pickers."""
        return list(serial.tools.list_ports.comports())

    # ---- low level SCPI I/O ---------------------------------------------------
    def _frame(self, cmd):
        if self.address is not None:
            cmd = f"ADDR {self.address}::{cmd}"
        return cmd

    def write(self, cmd):
        """Send a raw SCPI command that expects no reply (e.g. 'CURR 5').
        A '\\n' terminator is appended automatically -- do not include one."""
        line = self._frame(cmd) + "\n"
        self.ser.write(line.encode("ascii"))
        self.ser.flush()

    def query(self, cmd):
        """Send a raw SCPI query (should end in '?') and return the
        stripped text response. Returns '' on timeout/no reply."""
        self.ser.reset_input_buffer()
        self.write(cmd)
        resp = self.ser.readline()
        return resp.decode("ascii", errors="replace").strip()

    def query_float(self, cmd):
        """Like query(), but strips a trailing unit suffix (e.g. '1.23A')
        and parses the result as a float."""
        return self._to_float(self.query(cmd))

    @staticmethod
    def _to_float(text):
        """Parse a leading numeric token out of a response like '0.000V' or
        '1.23ohm', ignoring the unit suffix. Returns nan if unparsable."""
        match = re.match(r"^\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", text)
        return float(match.group(1)) if match else float("nan")

    def check_errors(self):
        """Query SYSTem:ERRor? and raise UTL8211Error if the instrument
        reported anything other than 'no error'. Call this after a command
        (or a batch of commands) when you need to be sure they were valid,
        e.g. before starting an unattended driving-cycle run."""
        msg = self.query("SYSTem:ERRor?")
        if msg and "no error" not in msg.lower():
            raise UTL8211Error(msg)
        return msg

    # ---- common / system commands ---------------------------------------------
    def idn(self):
        """*IDN? -> 'Manufacturer,Model,SerialNumber,FirmwareVersion'."""
        return self.query("*IDN?")

    def reset(self):
        """*RST -- reset the instrument to its power-on default state."""
        self.write("*RST")

    def scpi_version(self):
        return self.query("SYSTem:VERSion?")

    def beep(self, on=True):
        """Enable/disable the instrument's beeper (SYSTem:BEEPer:STATe)."""
        self.write(f"SYST:BEEP:STAT {1 if on else 0}")

    # ---- input on/off & short circuit ------------------------------------------
    def input_on(self, on=True):
        """Turn the load input on/off, i.e. start/stop actually sinking
        current/power according to the current mode and setpoint."""
        self.write(f"INP {1 if on else 0}")

    def input_state(self):
        """Query whether the input is currently on. Returns '1' or '0'."""
        return self.query("INP?")

    def short(self, on=True):
        """Enable/disable the short-circuit test function (INPut:SHORt)."""
        self.write(f"INP:SHOR {1 if on else 0}")

    def short_state(self):
        return self.query("INP:SHOR?")

    # ---- mode selection ---------------------------------------------------------
    def set_mode(self, mode):
        """Select the load's operating mode.

        mode: one of Mode.CURRENT/VOLTAGE/POWER/RESISTANCE/DYNAMIC/BATTERY/LIST
              (or the equivalent SCPI long/short form as a string, e.g.
              'CURRent', 'CURR', 'CV', 'RESistance', ...).
        """
        self.write(f"MODE {mode}")

    def get_mode(self):
        """Query the current operating mode, e.g. 'CURR', 'VOLT', 'POW', 'RES'."""
        return self.query("MODE?")

    # ---- CC / CV / CR / CP setpoints ---------------------------------------------
    # These only take effect in the matching mode (set_mode() first).
    def set_current(self, amps):
        """Set the CC-mode current setpoint, in amps."""
        self.write(f"CURR {amps}")

    def get_current(self):
        """Query the CC-mode current setpoint, in amps."""
        return self.query_float("CURR?")

    def set_voltage(self, volts):
        """Set the CV-mode voltage setpoint, in volts."""
        self.write(f"VOLT {volts}")

    def get_voltage(self):
        """Query the CV-mode voltage setpoint, in volts."""
        return self.query_float("VOLT?")

    def set_resistance(self, ohms):
        """Set the CR-mode resistance setpoint, in ohms."""
        self.write(f"RES {ohms}")

    def get_resistance(self):
        """Query the CR-mode resistance setpoint, in ohms."""
        return self.query_float("RES?")

    def set_power(self, watts):
        """Set the CP-mode power setpoint, in watts."""
        self.write(f"POW {watts}")

    def get_power(self):
        """Query the CP-mode power setpoint, in watts."""
        return self.query_float("POW?")

    def set_setpoint(self, mode, value):
        """Mode-agnostic setpoint helper -- convenient for driving-cycle
        tables of (mode, value, duration) tuples where the mode may change
        step to step.

        mode  : Mode.CURRENT/VOLTAGE/POWER/RESISTANCE (CC/CV/CP/CR only --
                dynamic/battery/list modes have their own dedicated setters).
        value : the setpoint in that mode's native unit (A/V/W/ohm).
        """
        setters = {
            Mode.CURRENT: self.set_current,
            Mode.VOLTAGE: self.set_voltage,
            Mode.POWER: self.set_power,
            Mode.RESISTANCE: self.set_resistance,
        }
        key = mode[:4].upper()
        if key not in setters:
            raise ValueError(f"set_setpoint() only supports CC/CV/CP/CR, got mode={mode!r}")
        setters[key](value)

    # ---- protection limits -------------------------------------------------------
    def set_current_protection(self, amps):
        """Set the over-current protection (OCP) trip level, in amps."""
        self.write(f"CURR:PROT {amps}")

    def get_current_protection(self):
        return self.query_float("CURR:PROT?")

    def set_power_protection(self, watts):
        """Set the over-power protection (OPP) trip level, in watts."""
        self.write(f"POW:PROT {watts}")

    def get_power_protection(self):
        return self.query_float("POW:PROT?")

    # ---- slew rates -----------------------------------------------------------
    def set_current_slew(self, amps_per_us, rise=True, fall=True):
        """Set the current rise/fall slew rate, in A/us. By default sets
        both; pass rise=False or fall=False to set only one direction."""
        if rise and fall:
            self.write(f"CURR:SLEW {amps_per_us}")
        elif rise:
            self.write(f"CURR:SLEW:RISE {amps_per_us}")
        elif fall:
            self.write(f"CURR:SLEW:FALL {amps_per_us}")

    def set_voltage_slew(self, volts_per_ms):
        """Set the voltage rise/fall slew rate, in V/ms."""
        self.write(f"VOLT:SLEW {volts_per_ms}")

    # ---- start/stop load thresholds (Von/Voff) -------------------------------------
    def set_von(self, volts):
        """Voltage at which the load starts drawing (VOLTage:ON)."""
        self.write(f"VOLT:ON {volts}")

    def set_voff(self, volts):
        """Voltage at which the load stops drawing (VOLTage:OFF)."""
        self.write(f"VOLT:OFF {volts}")

    # ---- battery discharge test mode -----------------------------------------------
    def set_battery_mode(self, mode):
        """Select which quantity is held constant while discharging.
        mode: 'CURRent'|'RESistance'|'POWer' (or CURR/RES/POW)."""
        self.write(f"BAT:MODE {mode}")

    def get_battery_mode(self):
        return self.query("BAT:MODE?")

    def set_battery_current(self, amps):
        """Discharge current for CC battery mode. Range 0.01~20 A."""
        self.write(f"BAT:CURR {amps}")

    def set_battery_power(self, watts):
        """Discharge power for CP battery mode. Range 0.1~400 W."""
        self.write(f"BAT:POW {watts}")

    def set_battery_resistance(self, ohms):
        """Discharge resistance for CR battery mode. Range 0.05~7500 ohm."""
        self.write(f"BAT:RES {ohms}")

    def set_battery_cutoff_voltage(self, volts):
        """Voltage (Unload/Voff) at which the load auto-stops the discharge
        test. Set this to the correct low-voltage limit for your battery
        chemistry before starting a real discharge."""
        self.write(f"BAT:Unloade {volts}")

    def get_battery_capacity(self):
        """Accumulated discharge capacity: Ah in CC/CR battery mode, Wh in
        CP battery mode. Resets when a new discharge test starts."""
        return self.query_float("BAT:CAPA?")

    # ---- measurements -----------------------------------------------------------
    def measure_voltage(self):
        """Measured input voltage, in volts."""
        return self.query_float("MEAS:VOLT?")

    def measure_current(self):
        """Measured input current, in amps."""
        return self.query_float("MEAS:CURR?")

    def measure_power(self):
        """Measured input power, in watts."""
        return self.query_float("MEAS:POW?")

    def measure_resistance(self):
        """Measured equivalent resistance, in ohms."""
        return self.query_float("MEAS:RES?")

    def measure_all(self):
        """Read voltage/current/power/resistance in a single query
        (MEASure:REAL?). Returns a dict with float keys 'voltage',
        'current', 'power', 'resistance' plus the 'raw' response string.
        This is a single round trip, so prefer it over calling the
        individual measure_*() methods when you need several values at once
        (e.g. once per control-loop tick in a driving-cycle script)."""
        raw = self.query("MEAS:REAL?")
        parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
        keys = ("voltage", "current", "power", "resistance")
        out = {k: self._to_float(v) for k, v in zip(keys, parts)}
        out["raw"] = raw
        return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Test/command the UTL8211+ DC load over RS-232.")
    ap.add_argument("--port", default=None, help="COM port (default: auto-detect)")
    ap.add_argument("--baud", type=int, default=9600, help="baud rate (must match instrument)")
    ap.add_argument("--list", action="store_true", help="list serial ports and exit")
    args = ap.parse_args()

    if args.list:
        ports = UTL8211.list_ports()
        if not ports:
            print("No serial ports found. Is the USB-RS232 adapter plugged in / driver installed?")
        for p in ports:
            print(f"{p.device:8} | {p.description} | {p.hwid}")
        raise SystemExit(0)

    print("Opening load...")
    load = UTL8211(port=args.port, baudrate=args.baud)
    print(f"Connected on {load.ser.port} @ {load.ser.baudrate} 8N1")

    idn = load.idn()
    print(f"*IDN? -> {idn!r}")
    if not idn:
        print("No response. Check: baud rate matches instrument, TXD/RXD not swapped, "
              "load is powered and not keyboard-locked.")
    else:
        print("Communication OK.")
        print("Measurements:", load.measure_all())

    load.close()
