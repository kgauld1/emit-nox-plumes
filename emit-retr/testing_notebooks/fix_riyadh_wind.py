import os, sys, warnings

import earthaccess
from osgeo import gdal

import pandas as pd
import numpy as np
import xarray as xr
import math
import glob

import rasterio as rio

import netCDF4 as nc
from datetime import datetime, timedelta, timezone

from scipy import ndimage as ndi
from scipy.ndimage import binary_fill_holes, center_of_mass, distance_transform_edt
from scipy.interpolate import PchipInterpolator  # monotone, shape-preserving

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from scipy import linalg
from scipy.signal import savgol_filter
import scipy.ndimage as ndi
from scipy.ndimage import gaussian_filter, binary_erosion, binary_dilation, rotate, shift
from skimage.transform import hough_line, hough_line_peaks
from skimage.restoration import (
    denoise_tv_chambolle,
    denoise_bilateral,
    denoise_wavelet,
    estimate_sigma,
    inpaint_biharmonic
)

# This will ignore some warnings caused by holoviews
warnings.simplefilter('ignore') 

sys.path.append('../../EMIT-Data-Resources/python/modules/')
from emit_tools import emit_xarray, ortho_xr

sys.path.append('../')
from config import *
from REFERENCE_PLANTS import REFERENCE_PLANTS

sys.path.append('../../datasets/')
import get_geosfp
import get_campd
import amf_compute
import importlib


# Reload the module
importlib.reload(get_campd)
importlib.reload(get_geosfp)
importlib.reload(amf_compute)

get_emission_rate = get_campd.get_emission_rate
get_emissions = get_campd.get_emissions

get_geosfp_wind = get_geosfp.get_geosfp_wind
get_geosfp_tph = get_geosfp.get_geosfp_tph
get_geosfp_wind_agl = get_geosfp.get_geosfp_wind_agl

PREFIXES = {'OBS': 'EMIT_L1B_OBS_001_',
 'L1B': 'EMIT_L1B_RAD_001_',
 'MASK': 'EMIT_L2A_MASK_001_',
 'L2A': 'EMIT_L2A_RFL_001_'}

loc_name = 'RIYADH_PLANT_9'
loc_data = LOCS[loc_name]
retrieval_log_fn = f'retr_{loc_name}.csv'
retr_log = pd.read_csv(retrieval_log_fn)

wind_multipliers = []

for r in retr_log.iterrows():
    granule_name = r[1]['GRANULE']
    obs_time = datetime.strptime(granule_name.split('_')[0], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)

    aglwind_info = get_geosfp_wind_agl(loc_data['LAT'], loc_data['LON'], obs_time, z_agl=500, cache=f'{CONFIG["geosfp"]}/')
    slvwind_info = get_geosfp_wind(loc_data['LAT'], loc_data['LON'], obs_time, cache=f'{CONFIG["geosfp"]}/')

    U = np.mean([slvwind_info['power_law'](k) for k in [200, 250, 350, 450, 550, 650, 750] ])
    # print(slvwind_info)
    ratio = aglwind_info['speed_ms']/U

    wind_multipliers.append(ratio)
    print(f"{granule_name}, {ratio}")
    # break

retr_log['WIND_CORR'] = wind_multipliers
retr_log.to_csv(f'retr_{loc_name}_wcorr.csv')