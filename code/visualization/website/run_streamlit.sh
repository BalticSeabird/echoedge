#!/bin/bash

# Fullständig sökväg till din virtuella miljö
VENV_PATH="/home/joakim_e/git/echoedge/venv"

# Aktivera den virtuella miljön
source "$VENV_PATH/bin/activate"

# Fullständig sökväg till Streamlit-applikationen
STREAMLIT_APP="/home/joakim_e/git/echoedge/code/visualization/website/streamlit_website.py"

# Kontrollera om filen existerar
if [ ! -f "$STREAMLIT_APP" ]; then
    echo "Streamlit-fil existerar inte: $STREAMLIT_APP"
    exit 1
fi

# Lägg till den virtuella miljöns bin-katalog till PATH
export PATH="$VENV_PATH/bin:$PATH"

# Kör Streamlit-applikationen
streamlit run "$STREAMLIT_APP"

# Avaktivera den virtuella miljön
deactivate