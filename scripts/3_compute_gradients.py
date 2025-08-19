import numpy as np
import xarray as xr
from geographiclib.geodesic import Geodesic


from pyproj import Geod

def find_center_coordinates(lon: np.ndarray, lat: np.ndarray):
    """
    Find the center coordinates (lat, lon) of a given lat/lon grid.
    """
    return np.mean([np.min(lat), np.max(lat)]), np.mean([np.min(lon), np.max(lon)])


def convert_geographic_to_cartesian(
    lon, lat, lon_center, lat_center
):
    """
    Convert geographic coordinates to Cartesian distances (meters) relative to a center.
    Uses GeographicLib for geodesic calculations.
    """
    n = lon.size
    x = np.empty(n)
    y = np.empty(n)

    for i in range(n):
        x[i] = np.sign(lon[i] - lon_center) * Geodesic.WGS84.Inverse(
            lat[i], lon[i], lat[i], lon_center
        )["s12"]
        y[i] = np.sign(lat[i] - lat_center) * Geodesic.WGS84.Inverse(
            lat[i], lon[i], lat_center, lon[i]
        )["s12"]
    return x, y


def finite_difference_1d(s: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Compute finite difference derivatives dx/ds along a 1D axis.
    Handles NaNs as boundaries.
    """
    inan = np.isnan(x)
    dxds = np.full_like(x, np.nan, dtype=float)

    if np.any(inan):
        # Handle NaNs by segment boundaries
        inanR = np.hstack([inan[1:], True])
        inanL = np.hstack([True, inan[:-1]])

        sp = np.roll(s, -1)
        xp = np.roll(x, -1)
        sm = np.roll(s, 1)
        xm = np.roll(x, 1)

        iforward = ~inan & ~inanR & inanL
        icentral = ~inanL & ~inanR
        ibackward = ~inan & inanR & ~inanL

        dxds[iforward] = (xp[iforward] - x[iforward]) / (sp[iforward] - s[iforward])
        dxds[icentral] = (xp[icentral] - xm[icentral]) / (sp[icentral] - sm[icentral])
        dxds[ibackward] = (x[ibackward] - xm[ibackward]) / (
            s[ibackward] - sm[ibackward]
        )
    else:
        dxds[0] = (x[1] - x[0]) / (s[1] - s[0])  # forward
        dxds[-1] = (x[-1] - x[-2]) / (s[-1] - s[-2])  # backward
        dxds[1:-1] = (x[2:] - x[:-2]) / (s[2:] - s[:-2])  # central

    return dxds


def finite_difference_2d(s1_grid: np.ndarray, s2_grid: np.ndarray, x_grid: np.ndarray):
    """
    Compute finite differences along both axes for a 2D field.
    """
    N2, N1 = x_grid.shape
    dxds1 = np.stack(
        [finite_difference_1d(s1_grid[i, :], x_grid[i, :]) for i in range(N2)]
    )
    dxds2 = np.stack(
        [finite_difference_1d(s2_grid[:, i], x_grid[:, i]) for i in range(N1)]
    ).T
    return dxds1, dxds2

def rotate_along_grid(lon,lat,grad_1,grad_2, reversed_y=False):

    angle_x = np.ones_like(lon, dtype=float)
    angle_y = np.ones_like(lon, dtype=float)
    ny, nx = lon.shape  

    for j in range(ny):
        for i in range(nx):
            
            if i <nx-1:
                angle_x[j,i] = compute_heading_east_zero_np(lat[j,i],lon[j,i], lat[j,i+1], lon[j,i+1])
            else:
                angle_x[j,i] = compute_heading_east_zero_np(lat[j,i-1],lon[j,i-1], lat[j,i], lon[j,i])
            if j < ny-1:
                angle_y[j,i] = compute_heading_east_zero_np(lat[j,i],lon[j,i], lat[j+1,i], lon[j+1,i])
            else:
                angle_y[j,i] = compute_heading_east_zero_np(lat[j-1,i],lon[j-1,i], lat[j,i], lon[j,i])
                
    if reversed_y:
        grad_x = grad_1 * np.cos(angle_x) - grad_2 * np.cos(angle_y)
        grad_y = grad_1 * np.sin(angle_x) - grad_2 * np.sin(angle_y)
    else:
        grad_x = grad_1 * np.cos(angle_x) + grad_2 * np.cos(angle_y)
        grad_y = grad_1 * np.sin(angle_x) + grad_2 * np.sin(angle_y)
    return grad_x, grad_y

def compute_heading_east_zero_np(lat1, lon1, lat2, lon2):
    geod = Geod(ellps='WGS84')
    azimuth_deg, _, _ = geod.inv(lon1, lat1, lon2, lat2)
    
    # Convert degrees to radians using numpy
    azimuth_rad = np.deg2rad(azimuth_deg)
    
    # Convert so East=0 and anticlockwise (CCW)
    heading_rad = (np.pi/2 - azimuth_rad) % (2 * np.pi)
    return heading_rad

def process_dataset(dir_base: str):
    """
    Process an SST dataset: compute finite differences and save to a new file.
    """
    file_merge_all = f"{dir_base}/merge_all.nc"
    output_file = f"{dir_base}/merge_proc.nc"

    print(f"\n=== Processing dataset: {dir_base} ===")
    print(f"Opening {file_merge_all}")

    # check level
    if "L2P" in dir_base:
        level = "L2P"
    elif "L3C" in dir_base:
        level = "L3C"
    else:
        raise ValueError(f"Unknown dataset level in {dir_base}")
        
    with xr.open_dataset(file_merge_all) as ds:
        print("Dataset loaded")

        if level == "L3C":
            lat = ds["lat"].values
            lon = ds["lon"].values
            print("Creating coordinate grids")
            lon_grid, lat_grid = np.meshgrid(lon, lat)

            lat_center, lon_center = find_center_coordinates(lon, lat)

            x_vec, y_vec = convert_geographic_to_cartesian(
                lon_grid.ravel(), lat_grid.ravel(), lon_center, lat_center
            )
            x_grid = x_vec.reshape(lat_grid.shape)
            y_grid = y_vec.reshape(lat_grid.shape)

            dimx, dimy = "lon", "lat"
        elif level == "L2P":
            x_grid = ds["X"].values
            y_grid = ds["Y"].values
            
            lon_grid = ds["lon"].values
            lat_grid = ds["lat"].values
            
            dimx, dimy = "ni","nj"
        else:
            raise ValueError(f"Unknown level: {level}")

        temp_dx_list = []
        temp_dy_list = []

        n_time = ds.sizes["time"]
        print(f"Computing finite differences for {n_time} time steps")
        for t in range(n_time):
            if (t+1) % 10 == 0:
                print(f"  Processing time step {t + 1}/{n_time}",end="\r")
            temp = ds["sea_surface_temperature"].isel(time=t).values
            temp_dx, temp_dy = finite_difference_2d(x_grid, y_grid, temp)
            
            if level == "L2P":
                temp_dx, temp_dy = rotate_along_grid(lon_grid, lat_grid, temp_dx, temp_dy, reversed_y=True)
                
            temp_dx_list.append(temp_dx)
            temp_dy_list.append(temp_dy)

        print("\nTime loop complete. Attaching results to dataset")
        ds_out = ds.copy()
        ds_out["sea_surface_temperature_dx"] = (
            ("time", dimy, dimx),
            np.array(temp_dx_list),
        )
        ds_out["sea_surface_temperature_dy"] = (
            ("time", dimy, dimx),
            np.array(temp_dy_list),
        )

        print(f"Saving output to {output_file}")
        ds_out.to_netcdf(output_file)


if __name__ == "__main__":
    dir_base_list = [
        # "data/AHI_H08-STAR-L3C-v2.70_CROP_lon_111_116_lat_-24.5_-19.5_time_2022-03-01T00_00_00_2022-05-01T00_00_00",
        # "data/H09-AHI-L3C-ACSPO-v2.90_CROP_lon_111_116_lat_-24.5_-19.5_time_2024-03-01T00_00_00_2024-05-01T00_00_00",
        # "data/H09-AHI-L3C-ACSPO-v2.90_CROP_lon_111_116_lat_-24.5_-19.5_time_2023-03-01T00_00_00_2023-05-01T00_00_00",
        # "data/H09-AHI-L3C-ACSPO-v2.90_CROP_lon_111_116_lat_-24.5_-19.5_time_2025-03-01T00_00_00_2025-05-01T00_00_00",
        "data/H09-AHI-L2P-ACSPO-v2.90_CROP_lon_111_116_lat_-24.5_-19.5_time_2025-03-01T00_00_00_2025-05-01T00_00_00",
        "data/H09-AHI-L2P-ACSPO-v2.90_CROP_lon_111_116_lat_-24.5_-19.5_time_2024-03-01T00_00_00_2024-05-01T00_00_00",
        "data/H09-AHI-L2P-ACSPO-v2.90_CROP_lon_111_116_lat_-24.5_-19.5_time_2023-03-01T00_00_00_2023-05-01T00_00_00",
        "data/AHI_H08-STAR-L2P-v2.70_CROP_lon_111_116_lat_-24.5_-19.5_time_2022-03-01T00_00_00_2022-05-01T00_00_00",
    ]

    for dir_base in dir_base_list:
        process_dataset(dir_base)
