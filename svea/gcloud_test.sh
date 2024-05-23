#!/bin/bash

# Add Google Cloud SDK to PATH
export PATH=$PATH:/home/joakim/Dokument/git/echoedge/google-cloud-sdk/bin

# Print PATH to debug
echo "PATH is: $PATH"

# Check if gcloud exists in the specified path
if [ ! -f "/home/joakim/Dokument/git/echoedge/google-cloud-sdk/bin/gcloud" ]; then
    echo "gcloud not found at /home/joakim/Dokument/git/echoedge/google-cloud-sdk/bin/gcloud"
    exit 1
fi

# Ensure gcloud is available
if ! command -v gcloud &> /dev/null; then
    echo "gcloud command not found. Please ensure Google Cloud SDK is installed and gcloud is in your PATH."
    exit 1
fi

# Print gcloud version
gcloud --version
