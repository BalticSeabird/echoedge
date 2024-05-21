#!/bin/bash

FILEPATH='test2.png'
BUCKET_NAME='svea'

SERVICE_ACCOUNT_KEY="seabirdaidatabase-0a68840d87ff.json"

# Add Google Cloud SDK to PATH
export PATH=$PATH:/Users/joakimeriksson/Documents/GitHub/echoedge/svea/google-cloud-sdk/bin

# Ensure gcloud is available
if ! command -v gcloud &> /dev/null; then
    echo "gcloud command not found. Please ensure Google Cloud SDK is installed and gcloud is in your PATH."
    exit 1
fi

# Authenticate with Google Cloud service account key
gcloud auth activate-service-account --key-file="$SERVICE_ACCOUNT_KEY"

echo "Uploading $FILEPATH to gs://$BUCKET_NAME/"

if gcloud alpha storage cp "$FILEPATH" "gs://$BUCKET_NAME/"; then
    echo "Upload of $FILEPATH succeeded."
else
    echo "Upload of $FILEPATH failed."
fi
