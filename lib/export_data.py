import pandas as pd

def shorten_list(original_list, target_length):
    if target_length >= len(original_list):
        return original_list

    step_size = (len(original_list) - 1) / (target_length - 1)
    shortened_list = [original_list[int(round(i * step_size))] for i in range(target_length)]
    return shortened_list


def save_data(data, filename, save_path, txt_path=False):

    df = pd.DataFrame(data, index = range(0, len(data["depth"])))
    df.to_csv(f'{save_path}/{filename}')

    if txt_path:
        with open(txt_path, 'a') as txt_doc:
            txt_doc.write(f'{filename}\n')