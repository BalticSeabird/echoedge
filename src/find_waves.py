import numpy as np

def get_beam_dead_zone(echodata): 

    echodata[np.isnan(echodata)] = 0
    row_sums = np.mean(echodata, axis=1).tolist()
    for i, row in enumerate(row_sums):
        if row == row:
            if row < (-50):
                return i


def find_wave_smoothness(waves_list):
    wave_difs = [abs(j-i) for i, j in zip(waves_list[:-1], waves_list[1:])]
    wave_smoothness = sum(wave_difs) / len(waves_list)
    return wave_smoothness


def find_layer(echodata, beam_dead_zone, in_a_row_thresh, layer_quantile, layer_strength_thresh, layer_size_thresh):

    echodata[np.isnan(echodata)] = 0
    echodata = echodata[beam_dead_zone:]
    in_a_row = 0

    for n, row in enumerate(echodata):
        row = row[~np.isnan(row)]
        avg_val = np.quantile(row, layer_quantile)

        if avg_val < layer_strength_thresh:
            in_a_row += 1

        if in_a_row == in_a_row_thresh:
            break

    if n > layer_size_thresh:
        try: 
            layer = n + beam_dead_zone
            return layer
        
        except:
            return False
    else:
        return False


def find_waves(echodata, wave_thresh, in_a_row_waves, beam_dead_zone):

    echodata[np.isnan(echodata)] = 0

    line = []

    for i, ping in enumerate(echodata.T):

        in_a_row = 0
        found_limit = False

        # if depth[i] == depth[i]:
        #     ping_depth = int(depth[i])
        #     ping = ping[:ping_depth]

        for i, value in enumerate(ping):
            if value < wave_thresh:
                in_a_row += 1
            else:
                in_a_row = 0 
            if in_a_row == in_a_row_waves:
                found_limit = True 
                line.append(i-in_a_row)
                break
        if not found_limit:
            line.append(beam_dead_zone)


    for ping in range(echodata.shape[1]):
        echodata[:(line[ping]), ping] = 0

    wave_avg = sum(line) / len(line)
    wave_smoothness = find_wave_smoothness(line)
    
    return echodata, line, wave_avg, wave_smoothness
