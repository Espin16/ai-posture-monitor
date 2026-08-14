import serial
import time


class ArduinoLink:

    def __init__(self, port: str = "COM3", baud_rate: int = 9600, timeout: float = 0.1):
        self._serial = serial.Serial(port=port,
                                    baudrate = baud_rate,
                                    timeout = timeout)
        time.sleep(2)

    def send(self, signal: str):
        self._serial.write(f"{signal}\n".encode())

    # Only for debugging
    def send_and_wait(self, signal: str, response_delay: float = 1.0) -> str:
        self._serial.write(f"{signal}\n".encode())
        time.sleep(response_delay)
        response = self._serial.readline().decode('utf-8').strip()
        return response
    
    def close(self):
        self._serial.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()