#!/bin/bash

# MOUNT 
# sudo umount -l /home/joakim/Dokument/shared/drive
# sudo mount -t cifs -o credentials=/home/joakim/.smbcredentials //scifi01.svea.slu.se/temp/ /home/joakim/Dokument/shared/drive 

# DEFINE BASE PATHS
BASE_DIR_REPO="edge/svea"
BASE_DIR_DATA="/Volumes/temp/AQUA/2024/SPRAS 2024"

# RUN ECHOSOUNDER ANALYSIS WITH PYTHON
PARAMS_PATH="$BASE_DIR_REPO/params.yaml"
CSV_PATH="$BASE_DIR_REPO/out/csv"
IMG_PATH="$BASE_DIR_REPO/out/img"
NEW_FILES_PATH="$BASE_DIR_REPO/new_processed_files.txt"
COMPLETED_FILES_PATH="$BASE_DIR_REPO/completed_files.txt"

venv/bin/python3.11 $BASE_DIR_REPO/main.py "$BASE_DIR_DATA" "$COMPLETED_FILES_PATH" "$NEW_FILES_PATH" "$CSV_PATH" "$PARAMS_PATH" "$IMG_PATH"

# UPLOAD PROCESSED FILES TO GOOGLE CLOUD STORAGE

# define variables
BUCKET_NAME="svea"
SERVICE_ACCOUNT_KEY="$BASE_DIR_REPO/seabirdaidatabase-0a68840d87ff.json"
TEMP_FILE=$(mktemp)

# SQL variables
source $BASE_DIR_REPO/credentials.sh

# Add Google Cloud SDK to PATH
export PATH=$PATH:/home/joakim/Dokument/git/echoedge/google-cloud-sdk/bin

# Authenticate with Google Cloud service account key
gcloud auth activate-service-account --key-file="$SERVICE_ACCOUNT_KEY"

# Read each line from the file
while IFS= read -r line; do

    CSV_FILE="$CSV_PATH/$line"
    echo "CSV_FILE: $CSV_FILE"
    # Kontrollera om CSV-filen existerar
    if [[ -f $CSV_FILE ]]; then
        echo "Bearbetar fil: $CSV_FILE"
        
        # Läs CSV-filen rad för rad, hoppa över header-raden
        tail -n +2 "$CSV_FILE" | awk -F, '{OFS=","; $1=""; sub(/^,/, ""); print}' | while IFS=, read -r time lat lon depth wave_depth nasc0 fish_depth0 nasc1 fish_depth1 nasc2 fish_depth2 nasc3 fish_depth3 transmit_type file upload_time; do
            
            # SQL-fråga för att infoga data i tabellen
            SQL_QUERY="INSERT INTO svea (time, lat, lon, depth, wave_depth, nasc0, fish_depth0, nasc1, fish_depth1, nasc2, fish_depth2, nasc3, fish_depth3, transmit_type, file, upload_time) VALUES ('$time', '$lat', '$lon', '$depth', '$wave_depth', '$nasc0', '$fish_depth0', '$nasc1', '$fish_depth1', '$nasc2', '$fish_depth2', '$nasc3', '$fish_depth3', '$transmit_type', '$file', '$upload_time');"

            # Använd mysql-kommandot för att köra SQL-frågan
            mariadb --host=$HOST --user=$USER --password=$PASSWORD --database=$DATABASE_NAME --execute="$SQL_QUERY"
            
            # Kontrollera om kommandot lyckades
            if [[ $? -ne 0 ]]; then
                echo "Fel vid infogning av data från $CSV_FILE"
            fi
        done
    else
        echo "Filen $CSV_FILE finns inte."
    fi

    # Generate the two filenames based on the line
    FILE1="$IMG_PATH/${line/.csv/_complete.png}"
    FILE2="$IMG_PATH/${line/.csv/.png}"

    # Check if both files exist
    if [ -f "$FILE1" ] && [ -f "$FILE2" ]; then

        # Upload the first file to Google Cloud Storage bucket
        echo "Uploading $FILE1 to gs://$BUCKET_NAME/"

        if gsutil cp "$FILE1" "gs://$BUCKET_NAME/"; then
            echo "Upload of $FILE1 succeeded."
        else
            echo "Upload of $FILE1 failed."
            # If the upload failed, write the line to the temporary file
            echo "$line" >> "$TEMP_FILE"
            continue
        fi

        # Upload the second file to Google Cloud Storage bucket
        echo "Uploading $FILE2 to gs://$BUCKET_NAME/"

        if gsutil cp "$FILE2" "gs://$BUCKET_NAME/"; then
            echo "Upload of $FILE2 succeeded."
        else
            echo "Upload of $FILE2 failed."
            # If the upload failed, write the line to the temporary file
            echo "$line" >> "$TEMP_FILE"
        fi

    else
        echo "Files $FILE1 or $FILE2 not found."
        # If either file is not found, write the line to the temporary file
        echo "$line" >> "$TEMP_FILE"
    fi
done < "$NEW_FILES_PATH"

# Replace the original file with the temporary file
mv "$TEMP_FILE" "$NEW_FILES_PATH"

# Delete the temporary file if it still exists
[ -f "$TEMP_FILE" ] && rm "$TEMP_FILE"
