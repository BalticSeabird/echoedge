import pandas as pd
import os


def read_txt_file(path_to_txt):

    """
    Function to return list with filenames from path to txt-file with newly processed files. 
    """

    txt_file = open(path_to_txt, 'r')
    new_files = [line for line in txt_file.readlines()]

    return new_files


def send_data(data, filename):
    cols = data.columns

    message = f'#{filename}'

    for col in cols:




def calc_sum(new_files, save_path):

    """
    Function to calc mean for each parameter 
    """

    if len(new_files) == 2:
        if new_files[0][:-6] == new_files[1][:-6]:
            # add files and calc mean
            df1 = pd.read_csv(f'{save_path}/{new_files[0]}')
            df2 = pd.read_csv(f'{save_path}/{new_files[1]}')
            df = df1.append(df2)
            
            df = df.mean(axis=0)
            send_data(df, )

    else:
        # send files one and one
        for file in new_files:
            df = pd.read_csv(f'{save_path}/{file}')
            df = df.mean(axis=0)





if __name__ == '__main__':

    save_path = '/media/joakim/BSP-CORSAIR/edge/output'