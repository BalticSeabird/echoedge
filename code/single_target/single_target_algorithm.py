import echopype as ep
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2

from scipy.signal import argrelextrema, savgol_filter


# Funktion för att ladda in och processera data från en raw-fil
def process_data(data_path, cal_params, env_params):
    raw_echodata = ep.open_raw(data_path, sonar_model="EK80")

    ds_Sv_raw = ep.calibrate.compute_Sv(
        raw_echodata,
        env_params=env_params,
        cal_params=cal_params,
        waveform_mode='BB',
        encode_mode="complex",
    )
    selected_data = ds_Sv_raw.Sv.to_numpy()[0]  # ping 35 includes a lot of interesting stuff
    return selected_data, raw_echodata, ds_Sv_raw


# Funktion för att hitta index för punkter som är 6 dB under kandidaten, en före och en efter
def find_6dB_points(signal, candidate_indices, delta_dB):
    indices_6dB_before = []
    indices_6dB_after = []
    for idx in candidate_indices:
        target_down = signal[idx] - delta_dB

        # Hitta index för punkten 6 dB nedåt före kandidaten
        idx_before = np.where(signal[:idx] <= target_down)[0]
        if len(idx_before) > 0:
            indices_6dB_before.append(idx_before[-1])
        else:
            indices_6dB_before.append(None)

        # Hitta index för punkten 6 dB nedåt efter kandidaten
        idx_after = np.where(signal[idx:] <= target_down)[0]
        if len(idx_after) > 0:
            indices_6dB_after.append(idx + idx_after[0])
        else:
            indices_6dB_after.append(None)

    return indices_6dB_before, indices_6dB_after


def create_echogram(data, filepath='echogram'):
    np_data = np.nan_to_num(data, copy=True)
    np_data[np_data<-90]= -90
    np_data = (np_data - np_data.min())/(np_data.max() - np_data.min())
    np_data = np_data*256
    
    flip_np_data = np.rot90(np_data, 3) # flip the image to display it correctly

    cv2.imwrite(f'{filepath}_greyscale.png', flip_np_data) # greyscale image

    image = cv2.imread(f'{filepath}_greyscale.png', 0)
    colormap = plt.get_cmap('viridis') # 
    heatmap = (colormap(image) * 2**16).astype(np.uint16)[:,:,:3]
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)

    cv2.imwrite(f'{filepath}.png', heatmap) 


if __name__ == "__main__":

    envs = {
        'temperature': 16,
        'salinity': 9,
        'pressure': 10.1325
    }

    cals = {
        'gain_correction': 28.49,
        'equivalent_beam_angle': -21
    }

    path = '../../Raw_data/SLUAquaSailor2020V2-Phase0-D20200627-T031144-0.raw'
    # path = '../../Calibration 20210517/SLUAquaSailor2020CALIBRATION-Phase0-D20210517-T090725-0.raw'


    data, echodata, sv_data = process_data(path, cals, envs)

    splitbeam_angle = ep.consolidate.add_splitbeam_angle(echodata=echodata, waveform_mode='BB', encode_mode='complex', source_Sv=sv_data, to_disk=False)

    create_echogram(data)

    # Styrka, pingnummer, djup i pingen
    # Itererar över varje ping i det inladdade ekogrammet

    ping_num, ping_depth, ping_strength = [], [], []

    for id, ping in enumerate(data):

        ping = ping[500:16000]

        # Savitzky-Golay-filter
        window_length = 20
        polyorder = 2
        smoothed_data = savgol_filter(ping, window_length, polyorder)

        # Mean med Savgol filter
        window_size = 6
        window = np.ones(window_size) / window_size
        mean = np.convolve(smoothed_data, window, 'same')

        local_max = argrelextrema(mean, np.greater)  # Hitta lokala maxpunkter på mean av sav-gol 
        plt.plot(ping, label='Original Data')
        plt.plot(mean, label='Mean of Savitzky-Golay Smoothed Data')

        candidates = []
        
        # Plottar alla kandidater
        for v in local_max[0]:
            if ping[v] > -60:
                candidates.append(v)
                plt.vlines(x=v, ymin=-60, ymax=0, colors='green')

        # Använd funktionen för att hitta index för punkter 6 dB under före och efter
        indices_6dB_before, indices_6dB_after = find_6dB_points(mean, candidates, 6)

        # Kontrollera och stryk överlappande kandidater
        valid_candidates = []
        for i, idx in enumerate(candidates):
            if indices_6dB_before[i] is not None and indices_6dB_after[i] is not None:
                overlap = False
                for j, other_idx in enumerate(candidates):
                    if i != j:
                        if (indices_6dB_before[i] < other_idx < indices_6dB_after[i]):
                            overlap = True
                            break
                if not overlap:
                    valid_candidates.append(idx)
                    print(idx)
        print(valid_candidates)
        ping_num.extend([id for x in range(len(valid_candidates))])
        ping_depth.extend([vc/(170) for vc in valid_candidates]) # / med 170 för djup i meter
        ping_strength.extend([(ping[s]) for s in valid_candidates])


        
        # Visa resultat och plotta linjer för 6 dB nedåt punkter före och efter för giltiga kandidater
        for i, idx in enumerate(valid_candidates):
            if indices_6dB_before[candidates.index(idx)] is not None:
                plt.vlines(x=indices_6dB_before[candidates.index(idx)], ymin=-60, ymax=0, colors='blue', linestyles='dotted')
            if indices_6dB_after[candidates.index(idx)] is not None:
                plt.vlines(x=indices_6dB_after[candidates.index(idx)], ymin=-60, ymax=0, colors='red', linestyles='dotted')

        plt.legend()
        plt.show()


    # Itererar över alla valid_candidatews och försöker hitta vart i beamen de har träffats av ekot
    # På så sätt kan vi kompensera för detta och försöka följa och verifiera 
    # att en fisk som träffats i en ping följer ett rimligt mönster i nästa ping
    
    angle_athwartship = splitbeam_angle.angle_athwartship.data[0]
    angle_alongship = splitbeam_angle.angle_alongship.data[0]
    
    angle_alongships, angle_athwartships, combined = [], [], []
    for num, depth, strength in zip(ping_num, ping_depth, ping_strength):
        print(num, depth, strength)
        angle_alongships.append(angle_alongship[int(num)][int(depth)])
        angle_athwartships.append(angle_athwartship[int(num)][int(depth)])
        combined.append((abs(angle_alongship[int(num)][int(depth)]) + abs(angle_athwartship[int(num)][int(depth)])))

    valid_candidates_dict = {
        'ping_num': ping_num,
        'ping_depgh': ping_depth,
        'ping_strength': ping_strength,
        'angle_alongship': angle_alongships,
        'angle_athwartship': angle_athwartships,
        'combined': combined
    }

    plt.close()
    plt.scatter(valid_candidates_dict['angle_athwartship'], valid_candidates_dict['ping_strength'])
    plt.show()

    df = pd.DataFrame(valid_candidates_dict)
    df.to_excel('valid_candidates.xlsx')

    # Gör en scatter_plot med ping_num i x-led, ping_depth i y-led och ping_strength som storlek på varje scatter
    plt.scatter(ping_num, ping_depth, s=[(ps+100) for ps in ping_strength])
    plt.ylim(0, 100)
    plt.xlim(0, 255)
    plt.gca().invert_yaxis()
    plt.show()
    plt.close()

    # Histogram med storleksfördelningen 
    plt.hist(ping_strength, bins=50)  # Använder 10 intervall (bins) för histogrammet
    plt.xlabel('Storlek')
    plt.ylabel('Frekvens')
    plt.title('Histogram över storleksfördelningen')
    plt.show()

    # Scatter plot med djup på x-axeln och styrka på y-axeln
    plt.scatter(ping_depth, ping_strength)
    plt.show()

    
        


