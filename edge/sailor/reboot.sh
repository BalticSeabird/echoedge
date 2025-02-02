#!/bin/bash

# define paths
params_path="/home/joakim/Dokument/git/echoedge/edge/sailor/params.yaml"
params_ranges_path="/home/joakim/Dokument/git/echoedge/edge/sailor/ranges.yaml"
csv_path="/media/Sailor_flash/csv"
img_path="/media/Sailor_flash/img"
raw_path="/media/Sailor_flash"
new_files_path="/home/joakim/Dokument/git/echoedge/edge/sailor/new_processed_files.txt"
completed_files_path="/home/joakim/Dokument/git/echoedge/edge/sailor/completed_files.txt"
serial_path="/dev/ttyUSB0"
params_to_update="#env_params.temperature=25"

# run python scripts
/home/joakim/Dokument/git/echoedge/venv/bin/python3.11 /home/joakim/Dokument/git/echoedge/code/send_ready.py "$serial_path"
/home/joakim/Dokument/git/echoedge/venv/bin/python3.11 /home/joakim/Dokument/git/echoedge/code/update_params.py "$params_path" "$params_ranges_path" "$params_to_update" "$serial_path"
/home/joakim/Dokument/git/echoedge/venv/bin/python3.11 /home/joakim/Dokument/git/echoedge/code/main.py "$raw_path" "$completed_files_path" "$new_files_path" "$csv_path" "$params_path" "$img_path"
/home/joakim/Dokument/git/echoedge/venv/bin/python3.11 /home/joakim/Dokument/git/echoedge/code/send_results.py "$csv_path" "$new_files_path" "$serial_path"
