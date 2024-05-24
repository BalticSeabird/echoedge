import time
import sys


from src.serial_communication.send_vals_to_datalogger import send_values_to_datalogger, read_txt_file, calc_mean_and_send_data


if __name__ == '__main__':

    save_path = sys.argv[1]
    txt_path = sys.argv[2]
    ser_path = sys.argv[3]
    
    files = read_txt_file(txt_path)
    send_values_to_datalogger('message_transfer_start', ser_path)

    if files:
        send_values_to_datalogger('values_transfer_start', ser_path)
        calc_mean_and_send_data(files, save_path)
        print('Message successfully sent to datalogger.')
        open(txt_path, "w").close()
        
    else:
        print('No new results to send to datalogger.')
        send_values_to_datalogger('values_transfer_start', ser_path)
        send_values_to_datalogger('-1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0', ser_path)
    
    time.sleep(10)
    send_values_to_datalogger('shutdown', ser_path)
