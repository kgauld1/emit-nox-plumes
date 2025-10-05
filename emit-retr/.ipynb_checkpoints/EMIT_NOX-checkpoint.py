import earthaccess
import os
import warnings
import csv
from osgeo import gdal
import numpy as np
import math
import rasterio as rio
import xarray as xr
import holoviews as hv
import hvplot.xarray
import netCDF4 as nc
import cv2 as cv

import glob
import sys
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import rasterio
from rasterio.transform import Affine
from scipy import linalg
from lxml import etree
from scipy.ndimage import gaussian_filter

from skimage.morphology import binary_erosion
from skimage.transform import hough_line, hough_line_peaks

# This will ignore some warnings caused by holoviews
warnings.simplefilter('ignore') 

sys.path.append('../EMIT-Data-Resources/python/modules/')
from emit_tools import emit_xarray


wav_min = 418
wav_max = 492

def get_NO2_cross(fp="../enmap/cross_sections/no2c_97.txt"):
    # Get NO2 absorption cross-sections @ 220K from Vandaele et al. (1998)
    # Following TROPOMI NO2 ATBD: https://sentinel.esa.int/documents/247904/2476257/sentinel-5p-tropomi-atbd-no2-data-products
    # Data reference: http://spectrolab.aeronomie.be/no2.htm
    no2_cross_sections = pd.read_csv(
        fp,
        sep=" ",
        usecols=[3, 4],
        names=["vacuum_wavenumber_cm-1", "cross_section_cm^2/molecule_@220K"],
    )
    # add wavelength column (nm)
    no2_cross_sections["vacuum_wavelength_nm"] = 1e7 / no2_cross_sections["vacuum_wavenumber_cm-1"].values
    return no2_cross_sections
    

def gaussian_kernel(x, mu, fwhm_nm):
    """
    Create normalized Gaussian kernel with standard deviation sigma (nm) based on sigma = FWHM/sqrt(2log2)
    """
    # Define standard deviation sigma
    sigma = fwhm_nm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    # Create kernel
    k = np.exp(-0.5 * ((x - mu) / sigma)**2)
    # Normalize kernel to area = 1
    k /= np.trapz(k, x, dx=0.01)
    return k

def convolve_to_sensor_grid(xs_wav_nm, xs, sensor_wav_nm, sensor_fwhm_nm):
    """
    Convolve high-res cross-section xs_sigma(xs_wav_nm) to sensor grid using per-band Gaussian SRF.
    Returns array of length len(sensor_wav_nm).
    """
    out = np.zeros_like(sensor_wav_nm, dtype=float)
    for i, (mu, fwhm) in enumerate(zip(sensor_wav_nm, sensor_fwhm_nm)):
        # Create normalized Gaussian kernel
        k = gaussian_kernel(xs_wav_nm, mu, fwhm)
        # Apply kernel to absorption cross-section spectrum
        out[i] = np.trapz(k * xs, xs_wav_nm, dx=0.01)
    return out


if __name__ == "__main__": 
    bin_size = 4
    polydeg = 3

    fp = '/home/kgauld/orcd/pool/EMIT/EMIT_L1B_RAD_001_20250423T173412_2511311_021.nc'
    ds_geo = emit_xarray(fp, ortho=True)
    no2_cross_sections = get_NO2_cross()