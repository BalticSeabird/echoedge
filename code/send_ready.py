import pandas as pd
import serial
import time
import sys
import os


def send_values_to_datalogger(message, ser_path):

    # message = str.encode(message) 
    message = f'{message}\r\n'.encode()
    ser = serial.Serial(ser_path, 9600,timeout=(5),parity=serial.PARITY_NONE)           

    ser.write(message) # Send message to datalogger
    time.sleep(1)
    ser.close()

    return message


if __name__ == '__main__':

    ser_path = sys.argv[1]
    send_values_to_datalogger('ready', ser_path)
