import sys
import yaml
import ruamel.yaml
import serial
import time


def get_data_from_sailor(ser_path):

    ser = serial.Serial(ser_path, 9600,timeout=(5),parity=serial.PARITY_NONE)    
    incoming_string = ser.readline()

    return incoming_string


def send_values_to_datalogger(message, ser_path):

    # message = str.encode(message) 
    message = f'{message}\r\n'.encode()
    ser = serial.Serial(ser_path, 9600,timeout=(5),parity=serial.PARITY_NONE)           

    ser.write(message) # Send message to datalogger
    time.sleep(1)
    ser.close()

    return message


def send_data(data, ser_path):
    
    message = ''
    for key, val in data.items():
        val = round(val, 2)
        message += f'{val} '

    print(message)
    send_values_to_datalogger(message, ser_path)



def read_incoming_data(input_string):
    dict_to_update = {}

    input_strings = input_string.split('#')
    for param in input_strings[1:]:
        params = param.split(' ')

        if '.' in params[0]:
  
            nested_dict = dict_to_update
            cur_dict = nested_dict
            keys = params[0].split('.')
  
            for i, key in enumerate(keys):

                if i == len(keys) - 1:
                    cur_dict[key] = params[1]
                else:
                    cur_dict = cur_dict.setdefault(key, {})

            cur_dict[key] = params[1]

        else:
            dict_to_update[params[0]] = params[1]
    
    return dict_to_update


def read_ranges(ranges_path):
    with open(ranges_path) as f:
        ranges_dict = yaml.load(f, Loader=yaml.FullLoader)
    return ranges_dict


def val_incoming_changes(incoming_changes, ranges_dict): 
    new_dict = {}

    for key, incoming_val in incoming_changes.items():
        if key in ranges_dict:
            ranges_val = ranges_dict[key]
            
            if isinstance(ranges_val, list) and len(ranges_val) == 2:

                if int(ranges_dict[key][0]) <= int(incoming_val) <= int(ranges_dict[key][1]):
                    print('Incoming value is within the range.')
                    new_dict[key] = incoming_val
                else:
                    print('Incoming value is outside range!')
            
            elif isinstance(ranges_val, dict):
                new_dict[key] = val_incoming_changes(incoming_changes[key], ranges_val)

    return new_dict


def update_nested_dicts(base_dict, changes_dict):
    for key, val in changes_dict.items():
        if isinstance(val, dict):
            base_dict[key] = update_nested_dicts(base_dict.get(key, {}), val)
        else:
            base_dict[key] = float(val)
    return base_dict


def update_yaml_file(changes_dict, path):
    config, ind, bsi = ruamel.yaml.util.load_yaml_guess_indent(open(params_path))
    config = update_nested_dicts(config, changes_dict)

    yaml = ruamel.yaml.YAML()
    yaml.indent(mapping=ind, sequence=ind, offset=bsi) 
    with open(path, 'w') as fp:
        yaml.dump(config, fp)
