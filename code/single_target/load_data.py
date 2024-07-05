import echopype as ep
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from scipy.signal import argrelextrema


def swap_chan(ds: xr.Dataset) -> xr.Dataset:
    """Function to replace the channel dimension and coordinate with the
    frequency_nominal variable containing actual frequency values.
    Note that this step is possible only because there are
    no duplicated frequencies present.
    """
    return (
        ds.set_coords("frequency_nominal")
        .swap_dims({"channel": "frequency_nominal"})
        .reset_coords("channel")
    )

env_params = {
    'temperature': 5,
    'salinity': 9,
    'pressure': 10.1325
}

cal_params = {
    'gain_correction': 28.49,
    'equialent_beam_angle': -21
}


data_path = '../../Raw_data/SLUAquaSailor2020V2-Phase0-D20200627-T031144-0.raw'

raw_echodata = ep.open_raw(data_path, sonar_model="EK80")

ds_Sv_raw = ep.calibrate.compute_Sv(
    raw_echodata,
    env_params = env_params,
    cal_params = cal_params,
    waveform_mode='BB',
    encode_mode="complex",
)

ts = ep.calibrate.compute_TS(
    raw_echodata,
    env_params = env_params,
    cal_params = cal_params,
    waveform_mode='BB',
    encode_mode="complex",
)

ts_np = ts.TS.to_numpy()[0]

random_ping = ts_np[35] # 35 för första bra exempel, 185 för stim
random_ping = random_ping[350:-600]

window_size = 12  # Storleken på fönstret för det rullande medelvärdet

# Skapa ett fönster för medelvärdet
window = np.ones(window_size) / window_size

# Skapa ett rullande medelvärde utifrån fönstret
mean = np.convolve(random_ping, window, 'same')

local_max = argrelextrema(mean, np.greater)
print(local_max)

ax2 = [i for i in range(len(random_ping))]

plt.plot(random_ping)
plt.plot(mean)
plt.plot(ds_Sv_raw.Sv.to_numpy()[0][35])
for v in local_max[0]:
    print(random_ping[v])
    if random_ping[v] > -50:
        plt.hlines(y=v, xmin=-20, xmax=0)
plt.show()


