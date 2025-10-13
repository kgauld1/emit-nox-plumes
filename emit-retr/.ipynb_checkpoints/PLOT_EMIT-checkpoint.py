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

hv.extension('bokeh')
pn.extension()

def get_plume_mask(dSCD, thresh=95):
    # Threshold mask
    threshold = np.nanpercentile(dSCD, thresh)
    plume_mask = (dSCD > threshold).astype(np.uint8)
    
    plume_mask = binary_dilation(plume_mask, iterations=2)
    plume_mask = binary_erosion(plume_mask, iterations=4)
    plume_mask = binary_dilation(plume_mask, iterations=2)
    
    return plume_mask.astype(np.uint8)

def get_plume_xarray(ds):
    rad1500 = ds['radiance'].sel(wavelengths=1500, method='nearest').values
    dSCD = np.where(rad1500 <= -8000, np.nan, ds['dSCD'].values[:,:,0])
    plume_mask = get_plume_mask(dSCD)
    plume_data = np.where(plume_mask==0, np.nan, dSCD)
    
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

if __name__ == "__main__":
    loc_name = "RIYADH_PLANT_9"
    granule_name = "EMIT_L1B_RAD_001_20250613T114019_2516407_025.nc"
    fn = f"{CONFIG['results_folder']}/{loc_name}/{granule_name}"
    ds = xr.open_dataset(fn)
    
    plume_da = get_plume_xarray(ds)

    # Choose sane color limits from finite values
    finite_vals = plume_da.values[np.isfinite(plume_da.values)]
    vmin, vmax = np.nanpercentile(finite_vals, [5, 95]) if finite_vals.size else (np.nanmin(plume_da.values), np.nanmax(plume_da.values))
    
    plot = plume_da.hvplot.quadmesh(
        x='longitude', y='latitude',  # use the 2D coords
        geo=True, tiles='ESRI',
        cmap='Inferno', clim=(vmin, vmax),
        alpha=0.75, frame_height=650,
        crs=ccrs.PlateCarree(),      # data are in lon/lat
        title='Plume dSCD (orthorectified) on ESRI satellite'
    )

    pn.panel(plot).show()

