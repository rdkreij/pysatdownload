import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path


def list_nc_files(parts_dir: str):
    """
    List .nc files in a directory, sorted by timestamp from filename.
    """
    files = list(Path(parts_dir).glob("*.nc"))
    files_sorted = sorted(
        files,
        key=lambda f: pd.to_datetime(f.name[:14], format="%Y%m%d%H%M%S"),
    )
    return files_sorted


def merge_files_to_daily(files, output_dir: str):
    """
    Merge individual part files into daily netCDF files.
    """
    output_dir = Path(output_dir)
    times = pd.to_datetime([f.name[:14] for f in files], format="%Y%m%d%H%M%S")
    dates_unique = np.unique(times.date)

    output_dir.mkdir(exist_ok=True)
    for date in dates_unique:
        date_str = pd.Timestamp(date).strftime("%Y%m%d")
        files_date = [f for f in files if f.name.startswith(date_str)]
        print(f"   ↳ Processing {date} [{len(files_date)} files]...")

        try:
            with xr.open_mfdataset(files_date, combine="by_coords") as ds:
                ds.load().to_netcdf(output_dir / f"{date_str}.nc")
        except Exception as e:
            raise RuntimeError(f"Failed to process {date}: {e}")

    print(f"   ✓ Finished creating {len(dates_unique)} daily files.")


def merge_daily_to_all(daily_dir: str, output_file: str, ds_coords):
    """
    Merge all daily netCDF files into one dataset.
    """
    daily_dir = Path(daily_dir)
    output_file = Path(output_file)
    daily_files = sorted(daily_dir.glob("*.nc"))
    print(f"   ↳ Merging {len(daily_files)} daily files into {output_file.name}...")
    with xr.open_mfdataset(daily_files, combine="by_coords") as ds_all:
        ds_all.load()
        if ds_coords is not None:
            ds_all = ds_all.assign_coords(ds_coords.coords)
        ds_all.to_netcdf(output_file)
    print(f"   ✓ Saved merged file: {output_file}")


def process_dir(base_dir: str):
    """
    Process one base directory: merge parts → daily → all.
    """
    print(f"\n=== Processing base directory: {base_dir} ===")
    base = Path(base_dir)
    parts_dir = base / "parts"
    daily_dir = base / "merge_day"
    output_all = base / "merge_all.nc"

    if not parts_dir.exists():
        print(f"   ⚠ Skipping: {parts_dir} not found.")
        return

    nc_files = list_nc_files(str(parts_dir))
    print(f"   ✓ Found {len(nc_files)} part files in {parts_dir.name}")

    if not nc_files:
        print("   ⚠ No part files found, skipping.")
        return
    
    if "L2P" in base_dir:
        print("  ↳ Processing L2P dataset")
        ds_coords = xr.open_dataset(base / "coords.nc")
        print(f"   ↳ Coordinates dataset loaded from {base / 'coords.nc'}")

    merge_files_to_daily(nc_files, str(daily_dir))
    merge_daily_to_all(str(daily_dir), str(output_all), ds_coords)


if __name__ == "__main__":
    # List all base dirs as plain strings
    dir_base_list = [
        "data/H09-AHI-L2P-ACSPO-v2.90_CROP_lon_111_116_lat_-24.5_-19.5_time_2025-03-01T00_00_00_2025-05-01T00_00_00",
        "data/H09-AHI-L2P-ACSPO-v2.90_CROP_lon_111_116_lat_-24.5_-19.5_time_2024-03-01T00_00_00_2024-05-01T00_00_00",
        "data/H09-AHI-L2P-ACSPO-v2.90_CROP_lon_111_116_lat_-24.5_-19.5_time_2023-03-01T00_00_00_2023-05-01T00_00_00",
        "data/AHI_H08-STAR-L2P-v2.70_CROP_lon_111_116_lat_-24.5_-19.5_time_2022-03-01T00_00_00_2022-05-01T00_00_00",
    ]

    for base_dir in dir_base_list:
        process_dir(base_dir)
