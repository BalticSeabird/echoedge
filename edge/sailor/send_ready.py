import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.serial_communication import send_vals_to_datalogger

if __name__ == '__main__':

    ser_path = sys.argv[1]
    send_vals_to_datalogger('ready', ser_path)
