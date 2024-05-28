# define paths
params_path="/home/jonas/Documents/vscode/echodata/echoedge/code/params.yaml"
params_ranges_path="/home/jonas/Documents/vscode/echodata/echoedge/code/ranges.yaml"
csv_path="//home/jonas/Documents/vscode/echodata/echoedge/data"
img_path="/home/jonas/Documents/vscode/echodata/echoedge/data"
raw_path="/media/Sailor_flash/raw"
new_files_path="/home/joakim/Dokument/git/echoedge/code/new_processed_files.txt"
completed_files_path="/home/joakim/Dokument/git/echoedge/code/completed_files.txt"
serial_path="/dev/ttyUSB0"
params_to_update="#env_params.temperature=25"

# run python scripts
/home/joakim/Dokument/git/echoedge/venv/bin/python3.11 /home/joakim/Dokument/git/echoedge/code/send_ready.py "$serial_path"
/home/joakim/Dokument/git/echoedge/venv/bin/python3.11 /home/joakim/Dokument/git/echoedge/code/update_params.py "$params_path" "$params_ranges_path" "$params_to_update"
/home/joakim/Dokument/git/echoedge/venv/bin/python3.11 /home/joakim/Dokument/git/echoedge/code/main.py "$raw_path" "$completed_files_path" "$new_files_path" "$csv_path" "$params_path" "$img_path"
/home/joakim/Dokument/git/echoedge/venv/bin/python3.11 /home/joakim/Dokument/git/echoedge/code/send_results.py "$csv_path" "$new_files_path" "$serial_path"
