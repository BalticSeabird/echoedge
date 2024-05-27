import sys
import os

from serial_communication import send_vals_to_datalogger

if __name__ == '__main__':

    ser_path = sys.argv[1]
    send_vals_to_datalogger('ready', ser_path)
