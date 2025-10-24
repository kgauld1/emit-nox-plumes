import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter, binary_erosion, binary_dilation
import holoviews as hv
import hvplot.xarray
import panel as pn
from config import *
import numpy as np
from scipy import ndimage as ndi
import sys

sys.path.append('../EMIT-Data-Resources/python/modules/')
from emit_tools import emit_xarray, ortho_xr

hv.extension('bokeh')
pn.extension()

def get_plume_mask(dSCD, thresh=90):
    # Threshold mask
    threshold = np.nanpercentile(dSCD, thresh)
    plume_mask = (dSCD > threshold).astype(np.uint8)
    
    plume_mask = binary_dilation(plume_mask, iterations=3)
    plume_mask = binary_erosion(plume_mask, iterations=6)
    plume_mask = binary_dilation(plume_mask, iterations=3)
    
    return plume_mask.astype(np.uint8)

def two_plume_masks(img, pct=97, open_iters=1, close_iters=2, min_size=500):
    """
    img: 2D array (NaNs allowed)
    pct: percentile used for threshold (e.g., 95–99)
    open_iters/close_iters: morphology to denoise/bridge small gaps
    min_size: drop tiny specks before picking the two largest blobs
    """
    # 1) Threshold on high values (tune pct as needed)
    valid = np.isfinite(img)
    thr = np.nanpercentile(img, pct)
    bw = valid & (img >= thr)

    # 2) Morphology to clean the mask
    # Opening removes salt noise; closing bridges small gaps.
    bw = ndi.binary_opening(bw, iterations=open_iters)
    bw = ndi.binary_closing(bw, iterations=close_iters)

    # Remove very small objects
    lbl, n = ndi.label(bw)
    if n == 0:
        return np.zeros_like(bw, bool), np.zeros_like(bw, bool)

    sizes = ndi.sum(np.ones_like(bw), labels=lbl, index=np.arange(1, n+1))
    keep = {i+1 for i, s in enumerate(sizes) if s >= min_size}
    bw = np.isin(lbl, list(keep))

    # Re-label after size filter, fill holes
    lbl, n = ndi.label(bw)
    if n == 0:
        return np.zeros_like(bw, bool), np.zeros_like(bw, bool)
    lbl = ndi.binary_fill_holes(lbl>0).astype(int) * lbl

    # 3) Keep the two largest components
    sizes = ndi.sum(np.ones_like(lbl, int), labels=lbl, index=np.arange(1, n+1))
    order = np.argsort(sizes)[::-1]            # largest first
    top_two_labels = [order[0]+1] + ([order[1]+1] if n >= 2 else [])

    mask_two = np.isin(lbl, top_two_labels)

    # 4) If you want them separately as "top" and "bottom", split by centroid row
    rows = np.arange(img.shape[0])
    centroids = [ndi.center_of_mass(np.ones_like(lbl), lbl, lab) for lab in top_two_labels]
    # each centroid is (row, col); smaller row ~ nearer the "top" of the array
    sort_idx = np.argsort([c[0] for c in centroids])

    top_label = top_two_labels[sort_idx[0]]
    bottom_label = top_two_labels[sort_idx[-1]]

    mask_top = (lbl == top_label)
    mask_bottom = (lbl == bottom_label)

    # Optional: fill holes in each plume
    mask_top = ndi.binary_fill_holes(mask_top)
    mask_bottom = ndi.binary_fill_holes(mask_bottom)

    return mask_top, mask_bottom


def get_plume_xarray(ds, field=False):
    rad1500 = ds['radiance'].sel(wavelengths=1500, method='nearest').values
    dSCD = np.where(rad1500 <= -8000, np.nan, ds['dSCD'].values[:,:,0])
    plume_mask = get_plume_mask(dSCD)
    mask_top, mask_bottom = two_plume_masks(plume_mask)
    mask_comb = mask_top | mask_bottom
    mask_comb = plume_mask
    if field:
        mask_comb = 1
    plume_data = np.where(mask_comb==0, np.nan, dSCD)
    
    # Build a DataArray for the plume with 2D lon/lat coords
    return xr.DataArray(
        plume_data,
        dims=("latitude", "longitude"),
        coords={
            "latitude": ds['latitude'].values,
            "longitude": ds['longitude'].values,
        },
        name="dSCD"
    )
    
def plot_plume_dSCD(ds, esri=True):
    plume_da = get_plume_xarray(ds)
    
    finite_vals = plume_da.values[np.isfinite(plume_da.values)]
    vmin, vmax = np.nanpercentile(finite_vals, [5, 95]) if finite_vals.size else (np.nanmin(plume_da.values), np.nanmax(plume_da.values))
    
    plot = plume_da.hvplot.quadmesh(
        x='longitude', y='latitude',  # use the 2D coords
        geo=True, tiles='ESRI' if esri else None,
        cmap='inferno', clim=(vmin, vmax),
        alpha=0.75, frame_height=650,
        crs=ccrs.PlateCarree(),      # data are in lon/lat
        title='Plume dSCD (orthorectified) on ESRI satellite' if esri else 'Plume dSCD (orthorectified)'
    )
    return plot
    
def plot_field_dSCD(ds):
    plume_da = get_plume_xarray(ds, field=True)
    
    finite_vals = plume_da.values[np.isfinite(plume_da.values)]
    vmin, vmax = np.nanpercentile(finite_vals, [5, 95]) if finite_vals.size else (np.nanmin(plume_da.values), np.nanmax(plume_da.values))
    
    plot = plume_da.hvplot.quadmesh(
        x='longitude', y='latitude',  # use the 2D coords
        geo=True, tiles=None,
        cmap='RdBu_r', clim=(vmin, vmax),
        alpha=0.75, frame_height=650,
        crs=ccrs.PlateCarree(),      # data are in lon/lat
        title='Field dSCD (orthorectified)'
    )
    return plot


if __name__ == "__main__":
    loc_name = "New_Madrid_Power_Plant"
    granule_name = "EMIT_L1B_RAD_001_20241012T201651_2428613_043"
#    fn = f"{CONFIG['results_folder']}/{loc_name}/{granule_name}"
#    loc_name = 'Bridger'
#    granule_name = 'EMIT_L1B_RAD_001_20230206T180234_2303712_009'
    
    granule_fn = f"{CONFIG['data_folder']}/{loc_name}/{granule_name}.nc"
    dscd_fn = f"{CONFIG['results_folder']}/{loc_name}/dSCD_{granule_name}.npy"
    
    DSCD_NPA = np.load(dscd_fn)
    
    ds = emit_xarray(granule_fn)
    wl_val = float(ds["wavelengths"].isel(wavelengths=0))  # or a specific value
    
    dscd_da = xr.DataArray(
        DSCD_NPA.astype('float32')[..., None],  # -> (downtrack, crosstrack, 1)
        dims=("downtrack", "crosstrack", "wavelengths"),
        coords={
            "downtrack": ds["downtrack"],
            "crosstrack": ds["crosstrack"],
            "wavelengths": [wl_val],
        },
        name="dSCD",
        attrs={
            "long_name": "Differential Slant Column Density (single band)",
            "units": "molec cm^-2",
        },
    )
    ds = ds.assign(dSCD=dscd_da)
    ds = ortho_xr(ds)
    
#    fn = '/Volumes/T9/EMIT-NOX/results/EMIT/New_Madrid_Power_Plant/EMIT_L1B_RAD_001_20250927T182549_2527011_024.nc'
#    fn = '/Volumes/T9/EMIT-NOX/results/EMIT/New_Madrid_Power_Plant/EMIT_L1B_RAD_001_20240815T191458_2422813_045.nc'
    
    plot_type='field'
    esri = True
    
    if plot_type == 'field':
        plot = plot_field_dSCD(ds)
    elif plot_type == 'plume':
        plot = plot_plume_dSCD(ds, esri=esri)
    else:
        raise Exception("Plot type should be field or plume")
    
    pn.panel(plot).show()

