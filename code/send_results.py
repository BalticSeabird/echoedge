#!/home/joakim/Dokument/git/echodata/venv/bin/python

import pandas as pd
import serial
import time
import os


def send_values_to_datalogger(message):

    message = str.encode(message) 
    ser = serial.Serial('/dev/ttyUSB0', 9600,timeout=(5),parity=serial.PARITY_NONE)           

    ser.write(b'values_transfer_start\r\n')
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
    message = f'#filename={filename}'
    for key, val in data.items():
        message += f'#{key}={val}'

    print(message)
    send_values_to_datalogger(message)




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
            


if __name__ == '__main__':
    save_path = '/media/joakim/BSP-CORSAIR/edge/output'
    txt_path = 'new_processed_files.txt'

    files = read_txt_file(txt_path)
    print(files)
    if files:
        calc_mean_and_send_data(files, save_path)
        print('Message successfully sent to datalogger.')
    else:
        print('No new results to send to datalogger.')
