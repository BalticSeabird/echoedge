


import echopype as ep
#path = '../../../../../../mnt/BSP_NAS2/Acoustics/Sailor_Karlso/Raw_data/2023/SLUAquaSailor2020V3-Phase0-D20230417-T121133-0.raw'
path = "/data/SLUAquaSailor2020V2-Phase0-D20200627-T001145-0.raw"
raw_echodata = ep.open_raw(path, sonar_model="EK80", use_swap=True)

