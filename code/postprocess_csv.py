
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Read all csv-files from Sailor
localpath = "/Volumes/JHS-SSD2/Sailor2024_csv/"
allfiles = list(Path(localpath).glob("*.csv"))

data = pd.DataFrame()
for file in allfiles: 
    if len(file.name) == 47:
        data = pd.concat([data, pd.read_csv(file)])

data["time2"] = pd.to_datetime(data["time"])
data.sort_values(by = ["time2"], inplace = True)

# Make -1
data.loc[data["nasc0"] > 10000, "nasc0"] = np.nan

fig, ax = plt.subplots(3, 1)
ax[0].plot(data["time2"], data["nasc0"])
ax[1].plot(data["time2"], -data["depth"])
ax[2].plot(data["time2"], data["bottom_hardness"])
plt.show()







