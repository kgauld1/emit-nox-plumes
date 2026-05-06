import earthaccess
import os
import warnings
import csv
import numpy as np
import math
import xarray as xr
import holoviews as hv
import hvplot.xarray
import netCDF4 as nc

import glob
import sys
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from scipy import linalg
from scipy.ndimage import gaussian_filter, binary_erosion, binary_dilation
from skimage.transform import hough_line, hough_line_peaks

from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

from config import CONFIG, LOCS
from REFERENCE_PLANTS import REFERENCE_PLANTS

sys.path.append('../EMIT-Data-Resources/python/modules/')
from emit_tools import emit_xarray, ortho_xr

wav_min = 403
wav_max = 493

####################################################################################################
############################## NOX CROSS SECTION ###################################################
####################################################################################################
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

def get_NOX_cross_sec(emit_ds, wav_min=wav_min, wav_max=wav_max):
    # Get NO2 absorption cross-sections @ 220K from Vandaele et al. (1998)
    # Following TROPOMI NO2 ATBD: https://sentinel.esa.int/documents/247904/2476257/sentinel-5p-tropomi-atbd-no2-data-products
    # Data reference: http://spectrolab.aeronomie.be/no2.htm
    no2_cross_sections = pd.read_csv(
        CONFIG['NOX_CSEC'],
        sep=" ",
        usecols=[3, 4],
        names=["vacuum_wavenumber_cm-1", "cross_section_cm^2/molecule_@220K"],
    )

    # add wavelength column (nm)
    no2_cross_sections["vacuum_wavelength_nm"] = 1e7 / no2_cross_sections["vacuum_wavenumber_cm-1"].values
    
    emit_spec_wlen = emit_ds["wavelengths"].to_numpy().astype(np.float64)
    emit_spec_fwhm = emit_ds["fwhm"].to_numpy().astype(np.float64)
    
    no2_cross_sections_conv = convolve_to_sensor_grid(no2_cross_sections["vacuum_wavelength_nm"], 
                                                      no2_cross_sections["cross_section_cm^2/molecule_@220K"],
                                                      emit_spec_wlen,
                                                      emit_spec_fwhm)
    
    # Select bands and convolved cross-sections in fitting window
    window_sel = (emit_spec_wlen >= wav_min) & (emit_spec_wlen <= wav_max)
    emit_vnir_spectral_bands_fitting_window = emit_spec_wlen[window_sel].copy()
    no2_cross_sections_conv_fitting_window = no2_cross_sections_conv[window_sel].copy()

    return no2_cross_sections_conv_fitting_window, emit_vnir_spectral_bands_fitting_window

####################################################################################################
################################# MATCHED FILTER ###################################################
####################################################################################################

def detrend_spectra(spectra, wavelengths, poly_order=5):
    """Remove low-order polynomial from each spectrum (like DOAS)"""
    n_pixels, n_bands = spectra.shape
    detrended = np.zeros_like(spectra)
    coeffs = np.polynomial.polynomial.polyfit(wavelengths, spectra.T, poly_order)
    baseline = np.polynomial.polynomial.polyval(wavelengths, coeffs)
    detrended = spectra - baseline
    return detrended

def detrend_single(spectrum, wavelengths, poly_order=5):
    """Same for the target vector"""
    coeffs = np.polynomial.polynomial.polyfit(wavelengths, spectrum, poly_order)
    baseline = np.polynomial.polynomial.polyval(wavelengths, coeffs)
    return spectrum - baseline

def destripe(cube, mask=None):
    """Remove per-column mean bias from each band. Optionally exclude points from mean bias using mask"""
    cube_norm = cube.copy()
    cube_norm[np.where(mask==1)] = np.nan
    col_mean = np.nanmean(cube_norm, axis=0, keepdims=True)  # (1, C, B)
    row_mean = np.nanmean(cube_norm, axis=(0,1), keepdims=True)  # (1, 1, B) global mean
    return cube - col_mean + row_mean

def get_clusters(emit, n_clusters=8):
    """K-means clustering on RxCxB hyperspectral image. Returns RxC class map."""
    R, C, B = emit.shape
    pixels = emit.reshape(-1, B)
    mask = np.all(np.isfinite(pixels), axis=1)
    labels = np.full(R * C, -1, dtype=np.int32)

    # reducing each spectrum to a PCA representation to speed up the transform-maintains good representations
    pix_reduced = PCA(n_components=10).fit_transform(pixels[mask])
    labels[mask] = MiniBatchKMeans(n_clusters=n_clusters, batch_size=1024, n_init=10, random_state=0).fit_predict(pix_reduced)
    return labels.reshape(R, C)

def run_matched_filter_clusters(emit, clusters, mask=None, use_destripe=False):
    """Run matched filter with a cluster mapping for computing reference spectra"""
    # Work in log space
    radiance_cube = np.array(emit['radiance'])
    log_cube = np.log(radiance_cube)

    if destripe:
        log_cube = destripe(log_cube, mask=mask)
    
    if mask is None:
        mask = np.zeros(log_cube.shape[:-1])
    mask = mask.flatten()
    clusters = clusters.flatten()
    n_rows, n_cols, n_bands = log_cube.shape
    pixels = log_cube.reshape(-1, n_bands)
    
    # Detrend everything
    wavelengths = np.array(desired_bands)
    pixels_dt = detrend_spectra(pixels, wavelengths, poly_order=3)
    target_dt = -detrend_single(nox_window, wavelengths, poly_order=3)

    field = np.zeros((n_rows, n_cols)).flatten()

    cluster_ids = np.unique(clusters)
    for c_id in cluster_ids:
        valid_idx = np.where(clusters==c_id)
        c_pixels = pixels_dt[valid_idx]  # every pixel in this cluster
        c_mask = mask[valid_idx]
        
        good = c_pixels[c_mask == 0]
        
        mu_c = good.mean(axis=0)
        Sigma_c = np.cov(good, rowvar=False) + 1e-6 * np.eye(n_bands)
        
        try:
            inv_Sigma_c = np.linalg.pinv(Sigma_c)
            
            y = c_pixels - mu_c
            c_field = (y @ inv_Sigma_c @ target_dt) / (target_dt.T @ inv_Sigma_c @ target_dt)
            field[valid_idx] = c_field  # or reshape appropriately
        except:
            field[valid_idx] = np.nan
    return field.reshape((n_rows, n_cols))
    
####################################################################################################
######################################## MASKING ###################################################
####################################################################################################

def crop_about_loc(ds, clat, clon, km_boundary=None, pix_boundary=None):
    """Crop EMIT granule about the target clat,clon by # of km or pix"""
    if km_boundary is None and pix_boundary is None:
        return np.ones_like(ds['radiance'])[...,0]
    
    lat, lon = ds['lat'], ds['lon']
    
    if km_boundary is not None:
        dlat = (km_boundary/2)/111
        dlon = (km_boundary/2)/(111*np.cos(np.radians(clat)))

        lat_min = clat - dlat
        lat_max = clat + dlat
        lon_min = clon - dlon
        lon_max = clon + dlon

        mask = (
            (lat >= lat_min) & (lat <= lat_max) &
            (lon >= lon_min) & (lon <= lon_max)
        )
        
        ys, xs = np.where(mask)
        if len(ys) == 0:
            raise ValueError("No pixels found inside requested box!")

        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
    elif pix_boundary is not None:
        plat, plot = np.argmin(np.abs(lat-clat)), np.argmin(np.abs(lon-clon))
        
        y0, y1 = plat - pix_boundary//2, plat + pix_boundary//2
        x0, x1 = plon - pix_boundary//2, plon + pix_boundary//2

    Ny, Nx = ds["radiance"].shape[:2]
        
    mask = np.zeros((Ny, Nx), dtype=bool)
    mask[y0:y1+1, x0:x1+1] = True

    # Expand to (Ny, Nx, Nlam)
    mask3d = mask[..., None]  # shape (Ny, Nx, 1)
    mask3d = np.broadcast_to(mask3d, ds["radiance"].shape)
    
    ds["radiance"] = ds["radiance"].where(mask3d)
    # ds["radiance"] = ds["radiance"].where(mask)

    return mask

def get_envi_mask(fn):
    mask_fn = fn.replace("L1B_RAD", "L2A_MASK")
    mask_ds = emit_xarray(mask_fn, ortho=False)
    cloud_mask = mask_ds['mask'].values[...,0]
    cloud_mask = binary_dilation(cloud_mask, iterations=2)
    
    cirrus_mask = mask_ds['mask'].values[...,1]
    water_mask = mask_ds['mask'].values[...,2]
    sc_mask = mask_ds['mask'].values[...,3]
    
    agg_mask = np.clip(cloud_mask+cirrus_mask+water_mask+sc_mask, 0, 1)
    del mask_ds
    return agg_mask.astype(np.uint8)


def run_retrieval_MF(loc_name, granule_name, km_boundary=None, n_clusters=64, use_env_mask=False):
    fn = f"{CONFIG['data_folder']}/{loc_name}/EMIT_L1B_RAD_001_{granule_name}.nc"

    if loc_name in REFERENCE_PLANTS.keys():
        loc_data = REFERENCE_PLANTS[loc_name]
    else:
        loc_data = LOCS[loc_name]

    rad_ds = emit_xarray(fn)
    
    nox_window, desired_bands = get_NOX_cross_sec(rad_ds)
    emit_window = rad_ds.sel(wavelengths=desired_bands, method='nearest')

    cropmask = crop_about_loc(emit_window, loc_data['LAT'], loc_data['LON'], km_boundary=km_boundary)
    

    clusters = get_clusters(np.array(rad_ds['radiance']), n_clusters=n_clusters)

    if use_env_mask:
        env_mask = get_envi_mask(fn)
        initial_mask = np.clip(env_mask + (1-cropmask), 0, 1).astype(np.uint8)
    else:
        initial_mask = (1-cropmask)

    initial_field = run_matched_filter_clusters(emit_window, clusters, mask=initial_mask, use_destripe=True)

    high_mask = (initial_field > np.nanpercentile(initial_field, 90)).astype(np.uint8)
    mask_comb = np.clip(initial_mask + high_mask, 0, 1).astype(np.uint8)

    field_mf_cluster = run_matched_filter_clusters(emit_window, clusters, mask=mask_comb, use_destripe=True)

    return field_mf_cluster