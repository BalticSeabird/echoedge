import sys
import time
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.serial_communication.params_update import get_data_from_sailor, send_values_to_datalogger, read_ranges, val_incoming_changes, read_incoming_data, update_yaml_file


if __name__ == '__main__':

    # read incoming params
    params_path = sys.argv[1]
    params_ranges_path = sys.argv[2]
    
    # incoming_string = sys.argv[3]
    ser_path = sys.argv[4]

    # send message_transfer_start to irridium
    send_values_to_datalogger('message_transfer_start', ser_path)
    time.sleep(3)

    # read data from serial input (sailor)
    incoming_string = get_data_from_sailor(ser_path)

    # transform incoming changes to changes in data
    print(incoming_string)
    incoming_changes = read_incoming_data(incoming_string)
    ranges_dict = read_ranges(params_ranges_path)
    ranges_dict = val_incoming_changes(incoming_changes, ranges_dict)
    print(ranges_dict)

    # update params based on incoming changes
    if len(ranges_dict) > 0:
        update_yaml_file(ranges_dict, params_path)
        print('Values are updated.')
    else:
        print('No values to update')
