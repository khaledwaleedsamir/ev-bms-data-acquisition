import serial
import struct
import time
import sys
import threading

"""Hoverboard serial controller.
"""

class HoverboardController:
    def __init__(self, serial_port="COM5", baud_rate=115200, start_frame=0xABCD, print_feedback=True):
        # Start frame and feedback size from the hoverboard firmware
        self.start_frame = start_frame
        self.feedback_size = 18
        self.startBytes = bytes.fromhex('ABCD')[::-1] # lower byte first
        self.incomingBytesPrev = bytes()
        
        # Internal state tracking 
        self.current_speed = 0
        self.current_steer = 0
        self.state_lock = threading.Lock()

        # Send timing
        self.send_interval = 0.1 # 100 ms

        # feedback data
        self.latest_feedback = None
        self.latest_feedback_lock = threading.Lock()
        self.print_feedback = print_feedback

        # Threading
        self.stop_threads_flag = False
        self.tx_thread = None
        self.rx_thread = None
        self.print_thread = None
        
        # Initialize serial connection
        try:
            self.ser_port = serial.Serial(serial_port, baud_rate, timeout=0.1)
            print("PC Hoverboard Serial Communications Started")
        except serial.SerialException as e:
            print(f"Error opening serial port: {e}")
            sys.exit(1)
    
    def start_threads(self):
        """ Start sender, receiver and print threads. """
        self.stop_threads_flag = False
        if self.tx_thread is None or not self.tx_thread.is_alive():
            self.tx_thread = threading.Thread(target=self.sender_loop)
            self.tx_thread.start()

        if self.rx_thread is None or not self.rx_thread.is_alive():
            self.rx_thread = threading.Thread(target=self.receiver_loop)
            self.rx_thread.start()

        if self.print_feedback:
            if self.print_thread is None or not self.print_thread.is_alive():
                self.print_thread = threading.Thread(target=self.print_loop)
                self.print_thread.start()
    
    def send_cmd(self, speed, steer):
        start = self.start_frame
        # XOR-based checksum (matches hoverboard firmware)
        checksum = start ^ steer ^ speed

        # Pack data into bytes (little-endian)
        packet = struct.pack("<HhhH", start, steer, speed, checksum)

        # Send packet over serial
        self.ser_port.write(packet)
        self.ser_port.flush()

    def ramp_speed(self, target_speed, step=20):
        while True:
            with self.state_lock:
                if self.current_speed == target_speed:
                    break
                if self.current_speed < target_speed:
                    self.current_speed = min(self.current_speed + step, target_speed)
                else:
                    self.current_speed = max(self.current_speed - step, target_speed)
            time.sleep(self.send_interval)  # Sleep OUTSIDE the lock
        
    def read_feedback(self):
        # Read incomingByte and Construct the start frame
        incomingByte = self.ser_port.read()
        bufStartFrame = self.incomingBytesPrev+incomingByte
        # Control bufStartFrame is startBytes
        if bufStartFrame != self.startBytes:
            self.incomingBytesPrev=incomingByte
            return
        else:
            feedback = {"cmd1":0, "cmd2":0, "speedR_meas":0, "speedL_meas":0, "batVoltage":0, "boardTemp":0, "cmdLed":0}
            checksumBytesCalculate=bufStartFrame
    
            for key,value in feedback.items():
                
                # Read 2 Next Bytes
                elementBytes=self.ser_port.read(2)
                
                # Convert 2 Bytes to Integer in feedback dictionnary
                feedback[key]=int.from_bytes(elementBytes, byteorder='little',signed=True)
                
                # Calculate checksumBytes
                checksumBytesCalculate = bytes(a^b for (a, b) in zip(checksumBytesCalculate, elementBytes))
                
            # Control checksumBytes Read is checksumBytes Calculate
            checksumBytesRead=self.ser_port.read(2)
            if checksumBytesCalculate == checksumBytesRead:
                #print("checksumBytes True")
                return feedback
            else:
                print("False checksum! Ignoring data.")
            return
    
    def sender_loop(self):
        while not self.stop_threads_flag:
            speed, steer = self.get_speed_steer()
            self.send_cmd(speed, steer)
            time.sleep(self.send_interval)   # 100 ms interval (firmware default)

    def receiver_loop(self):
        while not self.stop_threads_flag:
            fb = self.read_feedback()
            if fb:
                with self.latest_feedback_lock:
                    self.latest_feedback = fb

    def get_feedback(self):
        with self.latest_feedback_lock:
            return self.latest_feedback
    
    def print_loop(self):
        while not self.stop_threads_flag:
            fb = self.get_feedback()
            if fb:
                print(
                    f"cmd1(steer)={fb['cmd1']}  cmd2(speed)={fb['cmd2']}  "
                    f"speedR_meas={fb['speedR_meas']}  speedL_meas={fb['speedL_meas']}  "
                    f"Vbat={fb['batVoltage']/100}  Temp={fb['boardTemp']/10}  LED={fb['cmdLed']}"
                )
            time.sleep(1)  # Print every 1 s

    def stop_threads(self):
        self.stop_threads_flag = True
        for t in (self.tx_thread, self.rx_thread, self.print_thread):
            if t is not None and t.is_alive():
                t.join(timeout=2.0)

    def set_speed(self, speed):
        with self.state_lock:
            self.current_speed = speed

    def set_steer(self, steer):
        with self.state_lock:
            self.current_steer = steer

    def set_speed_steer(self, speed, steer):
        with self.state_lock:
            self.current_speed = speed
            self.current_steer = steer

    def get_speed(self):
        with self.state_lock:
            return self.current_speed

    def get_steer(self):
        with self.state_lock:
            return self.current_steer

    def get_speed_steer(self):
        with self.state_lock:
            return self.current_speed, self.current_steer

    def close(self):
        """Stop threads and clean up serial resources."""
        print("Closing Hoverboard Controller...")
        self.stop_threads()
        if self.ser_port and self.ser_port.is_open:
            try:
                self.ser_port.flush()
                self.ser_port.close()
            except Exception:
                pass
        self.tx_thread = None
        self.rx_thread = None
        self.print_thread = None