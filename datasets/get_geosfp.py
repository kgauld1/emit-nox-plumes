import os
import logging
import numpy as np
import xarray as xr
from scipy.optimize import curve_fit


def zero_pad_number(n):
    nstr = str(n)
    if len(nstr) == 1:
        nstr = "0" + nstr
    return nstr


def download_geosfp(date_time, cache="./../dat/_geosfp/"):
    """Copyright GHGSat, Inc. 2022"""

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    # Find nearest GEOS-FP time
    geos_time = f"{zero_pad_number(date_time.hour)}30"

    # Download the data
    TREE = "https://portal.nccs.nasa.gov/datashare/gmao/geos-fp/das"
    NAME = "GEOS.fp.asm.tavg1_2d_slv_Nx."
    TIME = [geos_time]
    y = date_time.year
    m = zero_pad_number(date_time.month)
    d = zero_pad_number(date_time.day)
    tree = f"{TREE}/Y{y}/M{m}/D{d}"
    name = f"{NAME}{y}{m}{d}_"
    for time in TIME:
        path = f"{tree}/{name}{time}.V01.nc4"
        if not os.path.isfile(f"{cache}{name}{time}.V01.nc4"):
            logging.info(f"downloading {path}")
            os.system(f"wget -P {cache} {path}")
        else:
            logging.info(f"{path} already downloaded")


def get_geosfp_wind(lat, lon, date_time, cache="./../dat/_geosfp/"):
    """
    Get GEOS-FP wind data from single-level (non-3d meteorology) files
    Infer 500-m wind from data for lower levels
    """
    # Download geos data if necessary
    mstr = zero_pad_number(date_time.month)
    dstr = zero_pad_number(date_time.day)
    hstr = zero_pad_number(date_time.hour)
    date_str = f"{date_time.year}{mstr}{dstr}"
    time_str = f"{hstr}30"
    geos_pth = f"{cache}/GEOS.fp.asm.tavg1_2d_slv_Nx.{date_str}_{time_str}.V01.nc4"
    if not os.path.exists(geos_pth):
        download_geosfp(date_time, cache=cache)

    # Read the data
    geos_data = xr.load_dataset(geos_pth)

    new_time_str = f"{date_time.year}-{mstr}-{dstr}T{hstr}:30:00"

    # Get 2-m wind
    # NOTE need np.mean() in u10_nearest, v10_nearest, etc.
    # because sometimes xr selects multiple nearest values
    U2 = geos_data.U2M
    V2 = geos_data.V2M
    u2_nearest = float(
        np.mean(U2.sel(time=new_time_str, lon=lon, lat=lat, method="nearest").values)
    )
    v2_nearest = float(
        np.mean(V2.sel(time=new_time_str, lon=lon, lat=lat, method="nearest").values)
    )
    U2_nearest = np.sqrt(u2_nearest**2 + v2_nearest**2)
    DIR2_nearest = 180 / np.pi * np.arctan2(-u2_nearest, -v2_nearest)
    if DIR2_nearest < 0:
        DIR2_nearest += 360

    # Get 10-m wind
    U10 = geos_data.U10M
    V10 = geos_data.V10M
    u10_nearest = float(
        np.mean(U10.sel(time=new_time_str, lon=lon, lat=lat, method="nearest").values)
    )
    v10_nearest = float(
        np.mean(V10.sel(time=new_time_str, lon=lon, lat=lat, method="nearest").values)
    )
    U10_nearest = np.sqrt(u10_nearest**2 + v10_nearest**2)
    DIR10_nearest = 180 / np.pi * np.arctan2(-u10_nearest, -v10_nearest)
    if DIR10_nearest < 0:
        DIR10_nearest += 360

    # Get 50-m wind
    U50 = geos_data.U50M
    V50 = geos_data.V50M
    u50_nearest = float(
        np.mean(U50.sel(time=new_time_str, lon=lon, lat=lat, method="nearest").values)
    )
    v50_nearest = float(
        np.mean(V50.sel(time=new_time_str, lon=lon, lat=lat, method="nearest").values)
    )
    U50_nearest = np.sqrt(u50_nearest**2 + v50_nearest**2)
    DIR50_nearest = 180 / np.pi * np.arctan2(-u50_nearest, -v50_nearest)
    if DIR50_nearest < 0:
        DIR50_nearest += 360

    # Get surface pressure and temperature
    PS = geos_data.PS
    PS_nearest = float(
        np.mean(PS.sel(time=new_time_str, lon=lon, lat=lat, method="nearest").values)
    )

    # Use a janky version of the log-law for wind to get U200 -- TODO
    # Assumes stable conditions, which is wrong -- need stability correction
    # z0 = 0.1  # assumed roughness height
    # d = 0.1  # assumed zero-plane displacement
    # U200_from_10 = U10_nearest * np.log((200 - d) / z0) / np.log((10 - d) / z0)
    # U200_from_50 = U50_nearest * np.log((200 - d) / z0) / np.log((50 - d) / z0)
    # U200 = np.mean([U200_from_10, U200_from_50])
    y = [U10_nearest, U50_nearest]
    z = [10, 50]

    # Zero-plane displacement
    d = 0.001  # Reasonable value for desert: https://en.wikipedia.org/wiki/Log_wind_profile

    def log_law(z, z0):
        return U2_nearest * np.log((z - d) / z0) / np.log((2 - d) / z0)

    def power_law(z, alpha):
        return U2_nearest * (z / 2) ** alpha

    # --- compute alpha from 10 m and 50 m only ---
    alpha_1050 = np.log(U50_nearest / U10_nearest) / np.log(50.0 / 10.0)
    
    def power_law_10m(z):
        return U10_nearest * (z / 10.0) ** alpha_1050


    # log law: https://en.wikipedia.org/wiki/Log_wind_profile
    try:
        p0 = 1
        popt, _ = curve_fit(log_law, z, y, p0=p0, maxfev=1000)
        z0_fit = popt[0]
        U500_log = log_law(500, z0_fit)
    except:
        print("curve_fit failed. Can't compute U500 with log law.")
        z0_fit = np.nan
        U500_log = np.nan
    # power law: https://en.wikipedia.org/wiki/Wind_profile_power_law
    try:
        p0 = 1 / 7
        popt, _ = curve_fit(power_law, z, y, p0=p0, maxfev=1000)
        alpha = popt[0]
        U500_power = power_law(500, alpha)
    except:
        print("curve_fit failed. Can't compute U500 with power law. Using U50 instead.")
        alpha = np.nan
        U500_power = np.nan

    # Wind dictionary
    wind_dict = {}
    wind_dict["U2"] = U2_nearest
    wind_dict["DIR2"] = DIR2_nearest
    wind_dict["U10"] = U10_nearest
    wind_dict["DIR10"] = DIR10_nearest
    wind_dict["U50"] = U50_nearest
    wind_dict["DIR50"] = DIR50_nearest
    wind_dict["U500_log"] = U500_log
    wind_dict["U500_power"] = U500_power
    wind_dict["z0_fit"] = z0_fit
    wind_dict["alpha"] = alpha_1050 #alpha
    wind_dict["log_law"] = lambda z: log_law(z, z0_fit)
    wind_dict["power_law"] = lambda z: power_law_10m(z)
    wind_dict["surface_pressure_Pa"] = PS_nearest

    return wind_dict

def get_geosfp_tph(lat, lon, date_time, elevation, cache="./../dat/_geosfp/"):
    """
    Get GEOS-FP temp/pressure/height profiles from single-level (non-3d meteorology) files
    """
    # Download geos data if necessary
    mstr = zero_pad_number(date_time.month)
    dstr = zero_pad_number(date_time.day)
    hstr = zero_pad_number(date_time.hour)
    date_str = f"{date_time.year}{mstr}{dstr}"
    time_str = f"{hstr}30"
    geos_pth = f"{cache}/GEOS.fp.asm.tavg1_2d_slv_Nx.{date_str}_{time_str}.V01.nc4"
    if not os.path.exists(geos_pth):
        download_geosfp(date_time, cache=cache)

    # Read the data
    geos_data = xr.load_dataset(geos_pth)

    new_time_str = f"{date_time.year}-{mstr}-{dstr}T{hstr}:30:00"

    # Hx = height at x mb pressure
    H1000 = geos_data.H1000
    H250  = geos_data.H250
    H500  = geos_data.H500
    H850  = geos_data.H850

    Hx    = [float(np.mean(x.sel(time=new_time_str, lon=lon, lat=lat, method="nearest").values)) 
                for x in (H850, H500, H250)]

    Hx    = [elevation] + Hx

    # Tx = temperature at x mb pressure
    T250  = geos_data.T250
    T500  = geos_data.T500
    T850  = geos_data.T850
    
    # TxM = temperature at x meters
    T2M   = geos_data.T2M
    T10M  = geos_data.T10M

    Tx    = [float(np.mean(x.sel(time=new_time_str, lon=lon, lat=lat, method="nearest").values)) 
                for x in (T2M, T850, T500, T250)]

    # Surface pressure in Pa
    PS    = float(np.mean(geos_data.PS.sel(time=new_time_str, lon=lon, lat=lat, method="nearest").values))
    Px    = [PS/100, 850, 500, 250]

    tph = {
        'T': np.array(Tx) - 273.15,
        'P': np.array(Px),
        'H': np.array(Hx)
    }
    return tph