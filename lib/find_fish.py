import numpy as np


def find_fish_median(echodata, waves, ground):
    """
    Function to calc the cumulative sum of fish for each ping.
    Args:
        data (numpy.ndarray): The sonar data (dB)
        waves: list of indices where the waves are ending in each ping
        ground: list of indices where ground starts for each ping
    Returns:
        sum (numpy.ndarray): a sum for each ping (NASC).
    """

    for i, ping in enumerate(echodata):

        wave_limit = waves[i]

        #checking if the value is nan
        if ground[i] == ground[i]:
            ground_limit = ground[i]
            ping[(ground_limit-5):] = np.nan # Also masking dead zone
        ping[:(wave_limit+3)] = np.nan # lab with different + n values here

    # calc NASC (Nautical Area Scattering Coefficient - m2nmi-2)1
    nasc = 4 * np.pi * (1852**2) * (10**(echodata/10)) * 0.1

    # nan to zero
    where_are_nans = np.isnan(nasc)
    nasc[where_are_nans] = 0
    return nasc



def medianfun(nasc, start, stop):
    """
    Function to calculate the median of cumulative sums for each list in the input list.
    It uses nasc outputted from the find_fish_median2 function
    """
    nascx, fish_depth = [], []
    nasc_copy = nasc.copy()
    for ping in nasc_copy:
        ping[0:(start*10)] = 0 #*10 to transform into pixels
        ping[(stop*10):1000] = 0
        cumsum = np.cumsum(ping)
        totnasc = sum(ping)
        medval = totnasc/2
        fishdepth = np.argmax(cumsum>medval)/10
        nascx.append(totnasc)
        fish_depth.append(fishdepth)
    return nascx, fish_depth