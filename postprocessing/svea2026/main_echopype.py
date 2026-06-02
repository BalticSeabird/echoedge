
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
import xarray as xr
import pandas as pd
import echoregions as er
from datetime import datetime, timezone
import dask
import numpy as np
import re
from pathlib import Path


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
REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_CSV = REPO_ROOT / "temp" / "seafloor_detection" / "bottom_depth_minimal.csv"
def _normalize_channel_label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", value).lower()


def _resolve_channel(ds: xr.Dataset, requested: str) -> str:
    available_channels = [str(c) for c in ds["channel"].values]
    if requested in available_channels:
        return requested

    normalized_target = _normalize_channel_label(requested)
    normalized_matches = [
        channel for channel in available_channels if _normalize_channel_label(channel) == normalized_target
    ]
    if normalized_matches:
        return normalized_matches[0]

    es38_matches = [channel for channel in available_channels if "ES38" in channel]
    if es38_matches:
        return es38_matches[0]

    raise ValueError(f"Requested channel '{requested}' not found. Available channels: {available_channels}")


def _cleanup_echodata(ed_obj) -> None:
    try:
        ed_obj.cleanup_swap_files()
    except Exception:
        pass


def main() -> int:
    ed = None
    try:
        ed = ep.open_raw(
            raw_path,
            sonar_model="EK80",
            storage_options={"anon": True},
        )

        # Use the correct waveform_mode and encode_mode combination
        ds_Sv = ep.calibrate.compute_Sv(ed, waveform_mode="CW", encode_mode="complex")

        # depth_offset=7.6 because transducer was on a lowered centerboard at 7.6 m deep
        ds_Sv = ep.consolidate.add_depth(ds_Sv, ed, depth_offset=7.6)

        sel = _resolve_channel(ds_Sv, sel_channel)
        print(f"Using channel: {sel}")

        ds_Sv["Sv"].plot(
            x="ping_time",
            row="channel",
            col_wrap=3,
            vmin=-115,
            vmax=-25,
            cmap="RdYlBu_r",
            yincrease=False,
        )

        ds_Sv["Sv"].plot(
            x="ping_time",
            row="channel",
            col_wrap=3,
            vmin=-40,
            vmax=-20,
            cmap="RdYlBu_r",
            yincrease=False,
        )

        ds_Sv.isel(range_sample=slice(0, 200))["Sv"].plot(
            x="ping_time",
            row="channel",
            col_wrap=3,
            vmin=-40,
            vmax=-20,
            cmap="RdYlBu_r",
            yincrease=False,
        )

        # Call detect_seafloor dispatcher
        basic_depth = detect_seafloor(
            ds_Sv,
            method="basic",
            params={
                "var_name": "Sv",
                "channel": sel,
                "threshold": (-40, -20),
                "offset_m": 0.5,
                "bin_skip_from_surface": 200,  # due to surface saturation
            },
        )

        assert isinstance(basic_depth, xr.DataArray)
        assert set(basic_depth.dims) == {"ping_time"}

        print(type(basic_depth))

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(basic_depth["ping_time"].values, basic_depth.values, fillstyle="full", markersize=1)
        ax.set_title("Seafloor depth over ping time")
        ax.set_xlabel("Ping time")
        ax.set_ylabel("Depth (m)")
        ax.invert_yaxis()
        plt.show()

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

        ds_Sv["angle_athwartship"].sel(channel=sel).plot(
            x="ping_time",
            y="range_sample",
            cmap="RdBu",
            yincrease=False,
            robust=True,
            cbar_kwargs={"label": "Athwart angle (deg)"},
        )
        plt.title(f"Channel {sel} – Athwart angle")
        plt.show()

        ds_Sv["angle_alongship"].sel(channel=sel).plot(
            x="ping_time",
            y="range_sample",
            cmap="RdBu",
            yincrease=False,
            robust=True,
            cbar_kwargs={"label": "Alongship angle (deg)"},
        )
        plt.title(f"Channel {sel} – Alongship angle")
        plt.show()

        blackwell_depth = detect_seafloor(
            ds=ds_Sv,
            method="blackwell",
            params={
                "channel": sel,
                "var_name": "Sv",
                "threshold": [-40, 2.4, 1.0],
                "offset": 0.5,
                "r0": 10,
                "r1": 1000,
                "wtheta": 28,
                "wphi": 52,
            },
        )

        assert isinstance(blackwell_depth, xr.DataArray)
        assert set(blackwell_depth.dims) == {"ping_time"}

        pt_basic = basic_depth.ping_time
        pt_blackwell = blackwell_depth.ping_time

        missing_in_blackwell = pt_basic[~pt_basic.isin(pt_blackwell)]
        print(f"Ping times in basic_depth but missing in blackwell_depth: {missing_in_blackwell.size}")
        print(missing_in_blackwell.values)

        missing_in_basic = pt_blackwell[~pt_blackwell.isin(pt_basic)]
        print(f"Ping times in blackwell_depth but missing in basic_depth: {missing_in_basic.size}")
        print(missing_in_basic.values)
        print("bd.ping_time:", basic_depth)
        print("\n\n")
        print("bw.ping_time:", blackwell_depth)

        bd, bw = xr.align(basic_depth, blackwell_depth, join="inner")
        assert bd.ping_time.equals(bw.ping_time), "Ping times are not aligned."

        diff = bd - bw
        common_time = bd.ping_time

        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 5), sharex=True)
        axs[0].plot(common_time, bd, label="Basic", color="navy")
        axs[0].plot(common_time, bw, label="Blackwell", color="firebrick", linestyle="--")
        axs[0].invert_yaxis()
        axs[0].set_title("Seafloor depth over ping time")
        axs[0].set_xlabel("Ping time")
        axs[0].set_ylabel("Depth (m)")
        axs[0].legend()
        axs[0].grid(True)

        axs[1].plot(common_time, diff, color="darkgreen")
        axs[1].axhline(0, color="gray", linestyle="--", linewidth=1)
        axs[1].set_title("Difference: Basic – Blackwell")
        axs[1].set_xlabel("Ping time")
        axs[1].set_ylabel("Depth difference (m)")
        axs[1].grid(True)

        plt.tight_layout()
        plt.show()

        print("Last 10 bottom_depth values (Basic):")
        print(bd.values[-10:])
        print("\nLast 10 blackwell_bottom values (Blackwell):")
        print(bw.values[-10:])

        Sv_da = ds_Sv["Sv"].sel(channel=sel)
        depth = ds_Sv["depth"].sel(channel=sel).isel(ping_time=0)

        Sv_plot = xr.DataArray(
            data=Sv_da.values,
            dims=["ping_time", "depth"],
            coords={"ping_time": Sv_da["ping_time"], "depth": depth.data},
            name="Sv",
        ).expand_dims(channel=[sel])

        ds_single = xr.Dataset(
            {
                "Sv": Sv_plot,
                "frequency_nominal": xr.DataArray([frequency_dict[sel]], dims=["channel"], coords={"channel": [sel]}),
            }
        )

        df = pd.DataFrame(
            {
                "time": blackwell_depth["ping_time"].values,
                "depth": blackwell_depth.values,
            }
        )

        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTPUT_CSV, index=False)

        lines = er.read_lines_csv(str(OUTPUT_CSV))
        plt.figure(figsize=(20, 6))
        sv_for_plot = ds_single["Sv"].isel(channel=0).sortby("depth")
        sv_for_plot.T.plot.pcolormesh(
            y="depth",
            cmap="RdYlBu_r",
            yincrease=False,
            vmin=-80,
            vmax=-40,
            alpha=0.2,
        )
        plt.plot(lines.data["time"], lines.data["depth"], color="black", label="Bottom", linewidth=2)
        plt.title("Echogram and Detected Seafloor")
        plt.tight_layout()
        plt.show()

        bottom_mask_da, bottom_points = lines.seafloor_mask(
            ds_single.Sv,
            operation="above_below",
            method="slinear",
            limit_area=None,
            limit_direction="both",
        )

        print("Unique Values in Bottom Mask:", np.unique(bottom_mask_da.data))
        print("Bottom Mask Ping Time Dimension Length:", len(bottom_mask_da["ping_time"]))
        print("Bottom Mask Depth Dimension Length:", len(bottom_mask_da["depth"]))
        print("Echogram Ping Time Dimension Length:", len(ds_single.Sv["ping_time"]))
        print("Echogram Depth Dimension Length:", len(ds_single.Sv["depth"]))

        plt.figure(figsize=(20, 6))
        bottom_mask_da.plot(y="depth", yincrease=False)

        inverted_mask = ~bottom_mask_da.astype(bool)

        mask_exists_Sv = xr.where(
            inverted_mask == 1,
            ds_single["Sv"].isel(channel=0),
            np.nan,
        )

        plt.figure(figsize=(20, 6))
        mask_exists_Sv.plot(y="depth", cmap="RdYlBu_r", yincrease=False, vmin=-80, vmax=-40)

        from echopype.mask import apply_mask

        print(ds_single["Sv"].dims)
        print(ds_single["Sv"].shape)
        print(inverted_mask.dims)
        print(inverted_mask.shape)

        bottom_mask_fixed = inverted_mask.transpose("ping_time", "depth")

        print(ds_single["Sv"].dims)
        print(ds_single["Sv"].shape)
        print(bottom_mask_fixed.dims)
        print(bottom_mask_fixed.shape)

        ds_with_mask = apply_mask(ds_single, bottom_mask_fixed, var_name="Sv", fill_value=np.nan)

        ds_MVBS_mask = ep.commongrid.compute_MVBS(
            ds_with_mask,
            range_var="depth",
            range_bin="1m",
            ping_time_bin="5s",
            range_var_max="1000m",
        )

        ds_MVBS_mask["Sv"].plot(
            x="ping_time",
            cmap="RdYlBu_r",
            yincrease=False,
            vmin=-80,
            vmax=-30,
        )

        print(f"echopype: {ep.__version__}, dask: {dask.__version__}")
        print(f"\n{datetime.now(timezone.utc)}")

        return 0

    finally:
        if ed is not None:
            _cleanup_echodata(ed)


if __name__ == "__main__":
    raise SystemExit(main())
