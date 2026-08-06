import serial
import time

arduino = serial.Serial(port='COM3', baudrate=9600, timeout=.1)  
time.sleep(2)  # Wait for the serial connection to initialize

# Insert code

def send_data(s: str):
    arduino.write(f"{s}\n".encode()) # Sending data to Arduino
    time.sleep(1) 
    data = arduino.readline() # Read the response from Arduino
    data = data.decode('utf-8').strip()  # Decode and strip whitespace
    return data

while True:
    state = input("Enter state (G/A/S): ")
    value = send_data(state)
    print(value)