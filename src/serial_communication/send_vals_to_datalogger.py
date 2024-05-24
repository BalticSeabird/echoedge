import serial
import time
import pandas as pd


def send_values_to_datalogger(message, ser_path):

    # message = str.encode(message) 
    message = f'{message}\r\n'.encode()
    ser = serial.Serial(ser_path, 9600,timeout=(5),parity=serial.PARITY_NONE)           

    ser.write(message) # Send message to datalogger
    time.sleep(1)
    ser.close()

    return message


def read_txt_file(path_to_txt):

    """
    Function to return list with filenames from path to txt-file with newly processed files. 
    """

    txt_file = open(path_to_txt, 'r')
    new_files = [line.replace('\n', '') for line in txt_file.readlines()]

    return new_files


def send_data(data, ser_path):
    
    message = ''
    for key, val in data.items():
        val = round(val, 2)
        message += f'{val} '

    print(message)
    send_values_to_datalogger(message, ser_path)
    

def calc_mean_and_send_data(new_files, save_path):

    """
    Function to calc mean for each parameter and file.
    """
    
    cols = ['bottom_hardness','bottom_roughness', 'wave_depth', 'depth', 'nasc0', 'fish_depth0', 'nasc1', 'fish_depth1', 'nasc2', 'fish_depth2', 'nasc3', 'fish_depth3']
    cols_dict = {}

    for col in cols:
        cols_dict[col] = []

    for file in new_files:
        df_temp = pd.read_csv(f'{save_path}/{file}')
        
        for col in cols:
            cols_dict[col].extend(df_temp[col].tolist())

    df = pd.DataFrame(cols_dict, columns=cols)

    means = df.median(axis=0)
    means = means.to_dict()
    send_data(means, file[:-4])
