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


def read_txt_file(path_to_txt):

    """
    Function to return list with filenames from path to txt-file with newly processed files. 
    """

    txt_file = open(path_to_txt, 'r')
    new_files = [line.replace('\n', '') for line in txt_file.readlines()]

    return new_files


def send_data(data, filename):
    
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

    if len(new_files) == 2:
        # calc mean for two files together
        if new_files[0].endswith('-0.csv') and new_files[1].endswith('-1.csv'):
            df1 = pd.read_csv(f'{save_path}/{new_files[0]}')
            df2 = pd.read_csv(f'{save_path}/{new_files[1]}')
            df = pd.concat([df1, df2], axis=0)
            df = df.drop(columns=['Unnamed: 0', 'time'])
            means = df.mean(axis=0)
            means = means.to_dict()
            send_data(means, new_files[0][:-6])
            

    else:
        # send files one and one
        for file in new_files:
            df = pd.read_csv(f'{save_path}/{file}')
            df = df.drop(columns=['Unnamed: 0', 'time'])
            means = df.mean(axis=0)
            means = means.to_dict()
            send_data(means, file[:-4])




def calc_mean_and_send_data2(new_files, save_path):

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
            


if __name__ == '__main__':

    save_path = sys.argv[1]
    txt_path = sys.argv[2]
    ser_path = sys.argv[3]
    
    files = read_txt_file(txt_path)
    send_values_to_datalogger('message_transfer_start', ser_path)

    if files:
        send_values_to_datalogger('values_transfer_start', ser_path)
        calc_mean_and_send_data2(files, save_path)
        print('Message successfully sent to datalogger.')
        open(txt_path, "w").close()
        
    else:
        print('No new results to send to datalogger.')
        send_values_to_datalogger('values_transfer_start', ser_path)
        send_values_to_datalogger('-1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0', ser_path)
    
    time.sleep(10)
    send_values_to_datalogger('shutdown', ser_path)
