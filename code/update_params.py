#!/home/joakim/Dokument/git/echodata/venv/bin/python

import os
import sys
import ruamel.yaml


params_path = 'params.yaml'

def read_incoming_data(input_string):
    dict_to_update = {}

    input_strings = input_string.split('#')
    for param in input_strings[1:]:
        params = param.split('=')

        if '.' in params[0]:
  
            nested_dict = dict_to_update
            cur_dict = nested_dict
            keys = params[0].split('.')
  
            for i, key in enumerate(keys):

                if i == len(keys) - 1:
                    cur_dict[key] = params[1]
                else:
                    cur_dict = cur_dict.setdefault(key, {})

            cur_dict[key] = params[1]

        else:
            dict_to_update[params[0]] = params[1]
    
    return dict_to_update


def is_nested_dict(d):
    if not isinstance(d, dict):
        return False

    for value in d.values():
        if isinstance(value, dict):
            return True 
    return False


def get_all_keys(d):
    for key, value in d.items():
        yield key
        if isinstance(value, dict):
            yield from get_all_keys(value)


def update_nested_dicts(base_dict, changes_dict):
    for key, val in changes_dict.items():
        if isinstance(val, dict):
            base_dict[key] = update_nested_dicts(base_dict.get(key, {}), val)
        else:
            base_dict[key] = float(val)
    return base_dict


def update_yaml_file(changes_dict, path):
    config, ind, bsi = ruamel.yaml.util.load_yaml_guess_indent(open(params_path))
    
    config = update_nested_dicts(config, changes_dict)

    yaml = ruamel.yaml.YAML()
    yaml.indent(mapping=ind, sequence=ind, offset=bsi) 
    with open(path, 'w') as fp:
        yaml.dump(config, fp)



if __name__ == '__main__':

    incoming_string = '#bin_size=0.2#wave_thresh=-78#env_params.temperature=21'
    incoming_changes = read_incoming_data(incoming_string)
    update_yaml_file(incoming_changes, params_path)
