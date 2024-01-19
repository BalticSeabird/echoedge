#!/bin/bash

# define paths
params_path="params.yaml"
params_ranges_path="ranges.yaml"
csv_path="/media/joakim/BSP-CORSAIR/edge/output"
raw_path="/media/joakim/BSP-CORSAIR/edge/input"
new_files_path="new_processed_files.txt"
completed_files_path="completed_files.txt"
serial_path="/dev/ttyUSB0"
params_to_update="#env_params.temperature=25"

# go to dir
cd code

# run python scripts
/home/joakim/Dokument/git/echoedge/venv/bin/python3.11 update_params.py "$params_path" "$params_ranges_path" "$params_to_update"
/home/joakim/Dokument/git/echoedge/venv/bin/python3.11 main.py "$raw_path" "$completed_files_path" "$new_files_path" "$csv_path"
/home/joakim/Dokument/git/echoedge/venv/bin/python3.11 send_results.py "$csv_path" "$new_files_path" "$serial_path"