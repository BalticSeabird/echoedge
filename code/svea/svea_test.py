import echopype as ep
from echopype import open_raw
import xarray as xr
filepath = '/mnt/BSP_NAS2/Sailor/Raw_data/2021/SLUAquaSailor2020V2-Phase0-D20210429-T140100-0.raw'
#Här får man specifiera sonar type 
echodata = ep.open_raw(filepath, sonar_model="EK80")
#ep.to_netcdf(save_path='/home/jonas/Documents/vscode/echodata/echoedge/data/svea')
df = xr.open_dataset(echodata)
