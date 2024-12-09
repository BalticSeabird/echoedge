


import echopype as ep
path = '../../../../../../../../mnt/BSP_NAS2/Acoustics/VOTO_Sailbuoy/HudsonBay_2024/Raw_files/MissionPlanHudsonBay2024V1-Phase0-D20240722-T013110-0.raw'
raw_echodata = ep.open_raw(path, sonar_model="EK80", use_swap=True)

