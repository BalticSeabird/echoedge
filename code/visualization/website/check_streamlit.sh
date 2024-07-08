#!/bin/bash

# Kontrollera om Streamlit körs
if ! pgrep -f "streamlit run" > /dev/null
then
    echo "Streamlit is not running. Starting Streamlit..."
    /home/joakim_e/git/echoedge/code/visualization/website/run_streamlit.sh
fi