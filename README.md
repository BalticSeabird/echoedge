# Echoedge
Repo with code and instructions on how to run echodata analysis on the edge. 


## Usage and installation instructions
These instructions are based on Raspberry Pi 5 (4GB RAM) as the edge device. It is possible that the setup could differ with other edge devices. 

#### Installation and setup
Clone this git repo and create a virtual environment with all necessary packages.

```Shell
git clone https://github.com/aidotsejoakim/echoedge
cd echoedge
python3 -m venv venv
pip3 install -r requirements.txt
```

### Worklflow

### Create txt-file from a dir with files

```Python
import os 

for file in os.listdir('/home/joakim/Dokument/sailor_data'):
    print(file)

    with open('completed_files.txt', 'a') as txt_doc:

        txt_doc.write(f'{file}\n')
```

### Help
To find path to mounted USB-device
```Shell
lsblk
```

## Hardware

### Edge devices
Since energy consumption, boot time and processing speed are the three of the major parameters when running onboard, we decided to go for the Raspberry Pi 5 with 4GB RAM. Before making the final decision, we went through a quite extensive testing process to find the most efficient edge device in our usecase.

| **Model**  | **RAM (GB)** | **Estimated power consumption (W)** | **Processing time (seconds, 100 MB raw-file)** |
|---------------|---------|---------|---------|
| Rasoberry Pi 5, Raspbian 12 (bullseye) OS | 4 | 4-5 | 12 | 0.655 | 0.828 |
| Coral Dev Board | 4 | 2-3 | N/A |
| Raspberry Pi 4, Raspbian 11 (bullseye) OS | 8 | 3-5 | 27 |
| Nvidia AGX Xavier, JetPack 5.1 R35.3.1 | 32 | 10-30 | 8 |
| VM (8 cores), Ubuntu 22.04.3 LTS| 32 | N/A | 5 |

### Echosounder 


## Ackownledgements
* The processing of the raw-files from the echosounder are based on the [Echopype](https://echopype.readthedocs.io/en/stable/) library. 