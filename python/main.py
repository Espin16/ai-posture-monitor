import serial
import time

ser = serial.Serial('COM3', 9600)  
time.sleep(2)  # Wait for the serial connection to initialize

# Insert code