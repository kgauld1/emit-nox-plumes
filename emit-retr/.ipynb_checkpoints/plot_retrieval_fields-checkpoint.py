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
import argparse
import glob

sys.path.append('../EMIT-Data-Resources/python/modules/')
from emit_tools import emit_xarray, ortho_xr

def get_plot(loc_name):
    fns = glob.glob(f'{CONFIG["results_folder"]}/{loc_name}/*.npy')
    
    cols = 5
    rows = (len(fns) + cols - 1)//cols
    fig, axs = plt.subplots(rows,cols, figsize=(cols*4,rows*4))
    axs = axs.flatten()
    
    
    for i in range(min(len(fns), rows*cols)):
        dSCD = np.load(fns[i])
        gn_unmask = fns[i].split('/')[-1]
        radfn = f"{CONFIG['data_folder']}/{loc_name}/{gn_unmask[5:-4]}.nc"
        
        ds_no_orth = emit_xarray(radfn)
        wl_val = float(ds_no_orth["wavelengths"].isel(wavelengths=0))  # or a specific value
        dscd_da = xr.DataArray(
            dSCD.astype('float32')[..., None],  # -> (downtrack, crosstrack, 1)
            dims=("downtrack", "crosstrack", "wavelengths"),
            coords={
                "downtrack": ds_no_orth["downtrack"],
                "crosstrack": ds_no_orth["crosstrack"],
                "wavelengths": [wl_val],
            },
            name="dSCD",
            attrs={
                "long_name": "Differential Slant Column Density (single band)",
                "units": "molec cm^-2",
            },
        )
        ds_no_orth = ds_no_orth.assign(dSCD=dscd_da)
        ds = ortho_xr(ds_no_orth)
        del ds_no_orth
    
        bounds = (np.min(ds['longitude'].values),
                  np.max(ds['longitude'].values),
                  np.min(ds['latitude'].values),
                  np.max(ds['latitude'].values))
    
        # axs[i].imshow(ds['radiance'][:,:,0], cmap='gray', extent=bounds)
        rad1500 = ds['radiance'].sel(wavelengths=1500, method='nearest').values
        dSCD_nan = np.where(rad1500 <= -8000, np.nan, ds['dSCD'].values[:,:,0])
        
        axs[i].imshow(dSCD_nan*1e19, cmap='RdBu_r', origin='upper', 
                      aspect='auto', vmin=-2e17, vmax=2e17, extent=bounds, alpha=0.7)
        axs[i].set_title(fns[i].split('/')[-1][-31:-4])

        del ds
        del dscd_da
    
    
    fig.suptitle(loc_name, fontsize=24, y=0.92)
    plt.savefig(f"{CONFIG['plot_folder']}/{loc_name}_dSCD_retrievals.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--loc_name", type=str, required=False, help="Location name")
    parser.add_argument("--run_all", action="store_true", help="Flag for plant mode")
    parser.add_argument("--loc_tick", action="store_true", help="Include location tick")
    args = parser.parse_args()
    
    if args.run_all:
        folders = glob.glob(f"{CONFIG['results_folder']}/*")
        loc_names = [k.split('/')[-1] for k in folders]

        for ln in loc_names:
            print(f"Starting {ln}")
            get_plot(ln)
            print(f"Done with {ln}")
        
    
    