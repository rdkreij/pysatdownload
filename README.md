**Download and crop Himawari-8/9 sea surface temperature (SST) data from NASA PO.DAAC**

This repository provides a Python script for retrieving Himawari-8/9 SST data (L2P or L3C), cropping it to a defined region and time range, and saving the results in NetCDF format. 

The `scripts` contain the required code (running files in order `1_`, `2_`, etc.). This will populate the `data` folder, which can be further inspected with the notebooks in `notebooks`.

This project uses EarthAccess for login. For more information on how to authenticate, see: https://earthaccess.readthedocs.io/en/stable/howto/authenticate/