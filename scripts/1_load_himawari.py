import os
import time
from pathlib import Path

import earthaccess
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Proj


def calculate_latlon_xr(ds):
    """Calculate latitude and longitude from a geostationary dataset."""
    # ONLY NEEDED FOR L2P DATASET

    x = ds['ni'].values
    y = ds['nj'].values

    geo_var = ds['geostationary']

    lon_0 = float(geo_var.longitude_of_projection_origin)
    h     = float(geo_var.perspective_point_height)
    sweep = str(geo_var.sweep_angle_axis)  

    p = Proj(proj='geos', h=h, lon_0=lon_0, sweep=sweep, datum='WGS84')

    X, Y = np.meshgrid(x*h, y*h)
    lon, lat = p(X, Y, inverse=True)

    # fill space pixels with NANs
    lon[np.abs(lon) > 360.0] = np.nan
    lat[np.abs(lat) > 90.0]  = np.nan

    return X, Y, lat, lon, x, y

def create_coords_dataset_xr(ds):
    """Create an xarray dataset with coordinates for a geostationary dataset."""
    # ONLY NEEDED FOR L2P DATASET
    
    X, Y, lat, lon, x, y = calculate_latlon_xr(ds)

    # create xarray dataset
    ds_coords = xr.Dataset(
        coords={
            'ni': ('ni', x),
            'nj': ('nj', y),
            'lat': (('nj','ni'), lat),
            'lon': (('nj','ni'), lon),
            'X': (('nj','ni'), X),
            'Y': (('nj','ni'), Y),
        }
    )

    # assign attributes
    ds_coords['lat'].attrs.update(long_name='latitude', units='degrees_north')
    ds_coords['lon'].attrs.update(long_name='longitude', units='degrees_east')
    ds_coords['X'].attrs.update(long_name='geostationary X', units='m')
    ds_coords['Y'].attrs.update(long_name='geostationary Y', units='m')

    return ds_coords


def get_ilims_jlims(lon, lat, lonlims, latlims):
    """Get the i and j limits for cropping based on longitude and latitude limits."""
    # OPTIONAL FOR L2P DATASET, NOT NEEDED FOR L3C DATASET    
    
    def find_nearest(x_grid, y_grid, x_point, y_point):
        """Find the (j, i) indices in x_grid, y_grid closest to the point (x_point, y_point), ignoring NaNs."""
        # Create a mask for valid points
        valid_mask = ~np.isnan(x_grid) & ~np.isnan(y_grid)
        
        # Compute distances only where valid
        
        distances = np.full_like(x_grid, np.inf, dtype=float)
        distances[valid_mask] = np.hypot(x_grid[valid_mask] - x_point,
                                        y_grid[valid_mask] - y_point)
        
        return np.unravel_index(np.argmin(distances), distances.shape)


    # # Find grid indices of the four corner points
    j1, i1 = find_nearest(lon, lat, lonlims[0], latlims[0])
    j2, i2 = find_nearest(lon, lat, lonlims[1], latlims[0])
    j3, i3 = find_nearest(lon, lat, lonlims[0], latlims[1])
    j4, i4 = find_nearest(lon, lat, lonlims[1], latlims[1])
    
    print(f"Indices: ({i1}, {j1}), ({i2}, {j2}), ({i3}, {j3}), ({i4}, {j4})")
    
    ilims = (min(i1, i2, i3, i4), max(i1, i2, i3, i4) + 1)
    jlims = (min(j1, j2, j3, j4), max(j1, j2, j3, j4) + 1)

    return ilims, jlims

def retry(func, retries=20, wait_seconds=60, stop_if_error=True, *args, **kwargs):
    """Retry a function call with a specified number of retries and wait time."""
    for attempt in range(1, retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"[Retry] Attempt {attempt}/{retries} failed: {e}")
            time.sleep(wait_seconds)
            
    if stop_if_error:
        raise RuntimeError(f"Function '{func.__name__}' failed after {retries} retries.")
    else: 
        print(f"[Retry] Function '{func.__name__}' failed after {retries} retries, but continuing execution.")

def main():
    timelims = ("2025-03-01T00:00:00","2025-05-01T00:00:00")
    tstep = 3600
    lonlims = (111, 116)
    latlims = (-24.5, -19.5)    

    version = 9 # Himawari 8 or 9
    level = "L2P"

    # set ilims and jlims instead of lonlims and latlims (optional and only for L2P dataset)
    ilims = None #(1314,1527) # (imin, imax) or None
    jlims = None #(3794,3995) # (jmin, jmax) or None
    
    dtlims = (np.datetime64(timelims[0]), np.datetime64(timelims[1]))
    dtrange = np.arange(dtlims[0], dtlims[1], np.timedelta64(tstep, 's'))


    if version == 8:
        short_name = f"AHI_H08-STAR-{level}-v2.70"
        long_name = f"STAR-{level}_GHRSST-SSTsubskin-AHI_H08-ACSPO_V2.70-v02.0-fv01.0"
    elif version == 9:
        short_name = f"H09-AHI-{level}-ACSPO-v2.90"
        long_name = f"STAR-{level}_GHRSST-SSTsubskin-AHI_H09-ACSPO_V2.90-v02.0-fv01.0"
    else:
        raise ValueError("Unsupported version.")

    print(f"Lon limits:  {lonlims}")
    print(f"Lat limits:  {latlims}")
    print(f"Time limits: {timelims}")
    print(f"Version:     {version}")
    print(f"Level:       {level}")

    # Authenticate with EarthAccess
    # For more information on how to authenticate, see:
    # https://earthaccess.readthedocs.io/en/stable/howto/authenticate/
    auth = earthaccess.login()
    print(f"EarthAccess authenticated: {auth.authenticated}")

    data_name = (
        f"{short_name}_CROP_lon_{lonlims[0]}_{lonlims[1]}_"
        f"lat_{latlims[0]}_{latlims[1]}_"
        f"time_{timelims[0].replace(':', '_')}_{timelims[1].replace(':', '_')}"
    )

    print(f"Data name: {data_name}")
    os.makedirs(f"data/{data_name}", exist_ok=True)
    os.makedirs(f"data/{data_name}/temp", exist_ok=True)
    os.makedirs(f"data/{data_name}/parts", exist_ok=True)

    temp_dir = f"data/{data_name}/temp"
    parts_dir = f"data/{data_name}/parts"

    retries=4
    wait_seconds=5

    flag = False
    for dt in dtrange:
        dt_pd = pd.Timestamp(dt)
        time_str = dt_pd.strftime("%Y%m%d%H%M%S")
        file_name = f"{time_str}-{long_name}.nc"
        link = f"https://archive.podaac.earthdata.nasa.gov/podaac-ops-cumulus-protected/{short_name}/{file_name}"
        file_path = f"{temp_dir}/{file_name}"
        
        
        attempts = 0
        for attempts in range(retries):
            try:
                earthaccess.download(link, temp_dir)
                        
                ds = xr.open_dataset(file_path)


                if level == "L2P":
                    if not flag:
                        print("Creating coordinates dataset")
                        ds_coords = create_coords_dataset_xr(ds)
                        lon = ds_coords['lon'].values
                        lat = ds_coords['lat'].values
                        if ilims is None and jlims is None:
                            print("Calculating ilims and jlims from lonlims and latlims")
                            ilims, jlims = get_ilims_jlims(lon, lat, lonlims, latlims)
                            
                        print(f"ilims: {ilims}, jlims: {jlims}")
                        print(f"Saving coordinates dataset to data/{data_name}/coords.nc")
                        
                        flag = True
                        ds_coords = ds_coords.isel(ni=slice(*ilims), nj=slice(*jlims))
                        ds_coords.to_netcdf(f"data/{data_name}/coords.nc")
                    ds_cropped = ds.isel(ni=slice(*ilims), nj=slice(*jlims))
                elif level == "L3C":
                    # Latitude decreases southward, so slice from max to min
                    lon_slice = slice(min(lonlims), max(lonlims))
                    lat_slice = slice(max(latlims), min(latlims))
                    ds_cropped = ds.sel(lon=lon_slice, lat=lat_slice)
                else:
                    raise ValueError(f"Unsupported level: {level}")

                # Ensure the dataset has the expected variables
                ds_cropped = ds_cropped[["quality_level", "sea_surface_temperature"]].copy()
                ds_cropped.attrs = {}

                output_path = f"{parts_dir}/{time_str}.nc"

                print(f"Saving cropped dataset to {output_path}")
                ds_cropped.to_netcdf(output_path)

                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Removed {file_path}")
                else:
                    print(f"File not found: {file_path}")
                    
                break
            except Exception as e:
                print(f"[Retry] Attempt {attempts + 1}/{retries} failed: {e}")
                time.sleep(wait_seconds)

if __name__ == "__main__":
    main()