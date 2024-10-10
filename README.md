# Echoedge
Repo with code and instructions on how to run echodata processing and analysis with Python. 

![Sailor](https://www.slu.se/globalassets/ew/org/inst/aqua/externwebb/om-oss/forskningsinfrastruktur/aquasailor-jhentati-300.jpg?width=480&height=480&mode=crop&upscale=false&format=webp)


## Getting started

##### Check your Python version
Start by checking your current environment, our configurations are shown below. It should be possible to run with other python-versions and other operating systems. Continue by cloning this repo, installing necessary packages and creating a cronjob. Please note that the latest version of Echopype is only compatible with Python>=3.9.

##### Start by cloning the git repo
```
git clone https://github.com/balticseabird/echoedge
cd echoedge
```

##### Create an environment and install packages
```
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
pip3 install -e . # install library for this repo
```


## Structure
```
echoedge/
├── edge/
│   ├── sailor/
│   └── svea/
├── postprocessing/
│   ├── SPRAS_2024/
│   └── subfolder-per-survey/
├── src/
│    ├── serial_communication/
│    ├── processing.py
│    ├── find_fish.py
│    └── visualization.py
├── .gitignore
├── requirements.txt
└── README.md
```


## Ackowledgements
* The processing of the raw-files from the echosounder is based on the [Echopype](https://echopype.readthedocs.io/en/stable/) library. 
