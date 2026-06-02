
#Seafloor detection

"""This notebook demonstrates how to:

Load and preprocess echosounder data using echopype
Detect the seafloor using two methods with detect_seafloor (basic thresholding and the Blackwell method)
Export and visualise the detected bottom line with echoregions
Apply the seafloor mask to the Sv data using apply_mask
Compute and visualise the mean volume backscattering strength (MVBS) from the masked data
First, we download the acoustic data. """

import echopype as ep
import matplotlib.pyplot as plt
from echopype.mask import detect_seafloor
from echopype.mask import apply_mask
import xarray as xr
import pandas as pd
import echoregions as er
from datetime import datetime, timezone
import dask
import numpy as np
import re
import atexit


# Dictionary of frequencies
frequency_dict = {
    "WBT 741110-15 ES18_ES": 18000,
    "WBT 741862-15 ES38-7_ES": 38000,
    "WBT_741862-15 ES38-7_ES": 38000,
    "WBT 741865-15 ES120-7C_ES": 120000,
    "WBT 741869-15 ES70-7C_ES": 700000,
    "WBT 743340-15 ES200-7C_ES": 200000,
    "WBT 741871-15 ES333-7C_ES": 333000,
}

sel_channel = "WBT_741862-15_ES38-7_ES"


raw_path = "/../../../../../../Volumes/JHS-SSD2/RUTSPRAS_2026/raw//D20260214-T172041.raw"
ed = ep.open_raw(
    raw_path,
    sonar_model="EK80",
    storage_options={"anon": True},
)


def _cleanup_echodata() -> None:
    try:
        ed.cleanup_swap_files()
    except Exception:
        pass


atexit.register(_cleanup_echodata)


# Use the correct waveform_mode and encode_mode combination
ds_Sv = ep.calibrate.compute_Sv(ed, waveform_mode="CW", encode_mode="complex")

# depth_offset=9.8 because transducer was on a lowered centerboard at 9.8 m deep
ds_Sv = ep.consolidate.add_depth(ds_Sv, ed, depth_offset=7.6)

channel_names = ds_Sv['channel'].values


def _normalize_channel_label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", value).lower()


available_channels = [str(c) for c in channel_names]
if sel_channel not in available_channels:
    normalized_target = _normalize_channel_label(sel_channel)
    normalized_matches = [
        c for c in available_channels if _normalize_channel_label(c) == normalized_target
    ]

    if normalized_matches:
        sel_channel = normalized_matches[0]
    else:
        es38_matches = [c for c in available_channels if "ES38" in c]
        if es38_matches:
            sel_channel = es38_matches[0]
        else:
            raise ValueError(
                f"Requested channel '{sel_channel}' not found. Available channels: {available_channels}"
            )

print(f"Using channel: {sel_channel}")


ds_Sv.isel(range_sample=slice(0, 200))["Sv"].plot(
    x="ping_time", 
    row="channel", 
    col_wrap=3,
    vmin=-40, vmax=-20,
    cmap="RdYlBu_r", 
    yincrease=False
)




# Call detect_seafloor dispatcher
basic_depth = detect_seafloor(
    ds_Sv,
    method="basic",
    params={
        "var_name": "Sv",
        "channel": sel_channel,
        "threshold": (-40, -20),
        "offset_m": 0.5,
        "bin_skip_from_surface": 200, # due to surface saturation
    },
)

# Check output
assert isinstance(basic_depth, xr.DataArray)
assert set(basic_depth.dims) == {"ping_time"}


# Add split-beam angles
ds_Sv = ep.consolidate.add_splitbeam_angle(
    ds_Sv,
    ed,
    waveform_mode="CW",
    encode_mode="complex",
    to_disk=False,
)

required_vars = ["Sv", "angle_alongship", "angle_athwartship", "depth"]
missing = [var for var in required_vars if var not in ds_Sv]

if not missing:
    print("All required variables are present for Blackwell detection.")
else:
    print(f"Missing required variables: {missing}.")



blackwell_depth = detect_seafloor(
    ds=ds_Sv,
    method="blackwell",
    params={
        "channel": sel_channel,
        "var_name": "Sv",
        "threshold": [-40, 2.4, 1.0],
        "offset": 0.5,
        "r0": 10,
        "r1": 1000,
        "wtheta": 28,
        "wphi": 52,
    }
)


# Get the ping_time values from each DataArray
pt_basic = basic_depth.ping_time
pt_blackwell = blackwell_depth.ping_time

# Find ping times in basic_depth but not in blackwell_depth
missing_in_blackwell = pt_basic[~pt_basic.isin(pt_blackwell)]
print(f"Ping times in basic_depth but missing in blackwell_depth: {missing_in_blackwell.size}")
print(missing_in_blackwell.values)

# Find ping times in blackwell_depth but not in basic_depth
missing_in_basic = pt_blackwell[~pt_blackwell.isin(pt_basic)]
print(f"Ping times in blackwell_depth but missing in basic_depth: {missing_in_basic.size}")
print(missing_in_basic.values)
print("bd.ping_time:", basic_depth)#.ping_time)
print("\n\n")
print("bw.ping_time:", blackwell_depth)#.ping_time)

# Align both bottom detections on ping_time using xarray
bd, bw = xr.align(basic_depth, blackwell_depth, join="inner")
assert bd.ping_time.equals(bw.ping_time), "Ping times are not aligned."

# Compute difference
diff = bd - bw
common_time = bd.ping_time  # aligned time axis, pick any


# Extract Sv, time, and depth (first ping)
Sv_da = ds_Sv["Sv"].sel(channel=sel_channel)
depth = ds_Sv["depth"].sel(channel=sel_channel).isel(ping_time=0)

# Build DataArray with (ping_time, depth) and add channel dim
Sv_plot = xr.DataArray(
    data=Sv_da.values,
    dims=["ping_time", "depth"],
    coords={"ping_time": Sv_da["ping_time"], "depth": depth.data},
    name="Sv"
).expand_dims(channel=[sel_channel])

# Wrap into Dataset and add frequency_nominal
ds_single = xr.Dataset({
    "Sv": Sv_plot,
    "frequency_nominal": xr.DataArray([frequency_dict[sel_channel]], dims=["channel"], coords={"channel": [sel_channel]})
})

# Create minimal DataFrame with required columns
df = pd.DataFrame({
    "time": blackwell_depth["ping_time"].values,
    "depth": blackwell_depth.values,
})

# Save to CSV (required at the moment)
# For now commented, because already exist in ./example_data
df.to_csv("./temp/seafloor_detection/bottom_depth_minimal.csv", index=False)

lines = er.read_lines_csv("./example_data/seafloor_detection/bottom_depth_minimal.csv")

plt.figure(figsize=(20, 6))

# Plot Sv echogram for one channel
ds_single["Sv"].isel(channel=0).T.plot.pcolormesh(
    y="depth",
    cmap="RdYlBu_r",
    yincrease=False,
    vmin=-80,
    vmax=-40,
    alpha=0.2
)

# Plot seafloor line from echoregions Lines object
plt.plot(
    lines.data['time'], lines.data['depth'], 
    color='black', label='Bottom', linewidth=2
)

plt.title("Echogram and Detected Seafloor")
plt.tight_layout()
plt.show()

# Use the built in mask function
bottom_mask_da, bottom_points = lines.seafloor_mask(
    ds_single.Sv, # Pass a DataArray where depth is a coordinate
    operation="above_below",
    method="slinear",
    limit_area=None,
    limit_direction="both"
)

print("Unique Values in Bottom Mask:", np.unique(bottom_mask_da.data))
print("Bottom Mask Ping Time Dimension Length:", len(bottom_mask_da["ping_time"]))
print("Bottom Mask Depth Dimension Length:", len(bottom_mask_da["depth"]))
print("Echogram Ping Time Dimension Length:", len(ds_single.Sv["ping_time"]))
print("Echogram Depth Dimension Length:", len(ds_single.Sv["depth"]))

plt.figure(figsize = (20, 6))
bottom_mask_da.plot(y="depth", yincrease=False)

# Invert 1/0 or True/False mask to match echopype apply_mask expectations
inverted_mask = ~bottom_mask_da.astype(bool)

# Get only channel values where the mask is 1
mask_exists_Sv = xr.where(
    inverted_mask == 1,
    ds_single["Sv"].isel(channel=0),
    np.nan,
)

# Plot the masked Sv
plt.figure(figsize = (20, 6))
mask_exists_Sv.plot(y="depth", cmap="RdYlBu_r", yincrease=False, vmin=-80, vmax=-40)



print(ds_single["Sv"].dims)
print(ds_single["Sv"].shape)
print(inverted_mask.dims)
print(inverted_mask.shape)

# Transpose to match ('ping_time', 'depth')
bottom_mask_fixed = inverted_mask.transpose("ping_time", "depth")

print(ds_single["Sv"].dims)
print(ds_single["Sv"].shape)
print(bottom_mask_fixed.dims)
print(bottom_mask_fixed.shape)

ds_with_mask = apply_mask(ds_single, bottom_mask_fixed, var_name="Sv", fill_value=np.nan)
('channel', 'ping_time', 'depth')
(1, 213, 36198)
('depth', 'ping_time')
(36198, 213)
('channel', 'ping_time', 'depth')
(1, 213, 36198)
('ping_time', 'depth')
(213, 36198)


# Compute MVBS
ds_MVBS_mask = ep.commongrid.compute_MVBS(
    ds_with_mask,
    range_var="depth",
    range_bin="1m",
    ping_time_bin="5s",
    range_var_max="1000m",
)

ds_MVBS_mask["Sv"].plot(
    x="ping_time", cmap='RdYlBu_r', yincrease=False,
    vmin=-80, vmax=-30
)


df2 = pd.DataFrame({
    "mvbs": ds_MVBS_mask["Sv"].values,
})

# Save to CSV (required at the moment)
# For now commented, because already exist in ./example_data
df2.to_csv("./temp/seafloor_detection/mvbs_minimal.csv", index=False)


print(f"echopype: {ep.__version__}, dask: {dask.__version__}")
print(f"\n{datetime.now(timezone.utc)}")
