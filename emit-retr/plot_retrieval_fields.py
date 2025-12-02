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

from skimage.restoration import denoise_tv_chambolle, inpaint_biharmonic

sys.path.append('../EMIT-Data-Resources/python/modules/')
from emit_tools import emit_xarray, ortho_xr

def get_plot(loc_name, loc_tick=False, tv_filter=False):
    fns = glob.glob(f'{CONFIG["results_folder"]}/{loc_name}/*.npy')
    if loc_tick:
        if loc_name in LOCS.keys():
            ltlat, ltlon = LOCS[loc_name]['LAT'], LOCS[loc_name]['LON']
        elif loc_name in POWER_PLANTS.keys():
            ltlat, ltlon = POWER_PLANTS[loc_name]['LAT'], POWER_PLANTS[loc_name]['LON']
        else:
            loc_tick = False
    
    cols = 5
    rows = (len(fns) + cols - 1)//cols
    fig, axs = plt.subplots(rows,cols, figsize=(cols*4,rows*4+5))
    axs = axs.flatten()
    
    
    for i in range(min(len(fns), rows*cols)):
        dSCD = np.load(fns[i])
        gn_unmask = fns[i].split('/')[-1]
        radfn = f"{CONFIG['data_folder']}/{loc_name}/{gn_unmask[5:-4]}.nc"
        
        try:
            ds_no_orth = emit_xarray(radfn)
        except:
            continue
            
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
        
        if not tv_filter:
            axs[i].imshow(dSCD_nan*1e19, cmap='RdBu_r', origin='upper',
                          aspect='auto', vmin=-2e17, vmax=2e17, extent=bounds)
            
        else:
            mask = np.isfinite(dSCD_nan)
            filled = inpaint_biharmonic(dSCD_nan, ~mask)
            tv = denoise_tv_chambolle(filled, weight=0.2)
            tv[~mask] = np.nan
            
            axs[i].imshow(tv, cmap='RdBu_r', origin='upper',
                            aspect='auto', extent=bounds)
        
        axs[i].set_title(fns[i].split('/')[-1][-31:-4])

        del ds
        del dscd_da
        if loc_tick:
            axs[i].scatter([ltlon], [ltlat], marker='x', c='red')
    
    fnout = f'{loc_name}_dSCD_retrievals'
    if loc_tick:
        fnout += '_wtick'
    if tv_filter:
        fnout += '_tv'
    fnout += '.png'
    
    fig.suptitle(loc_name, fontsize=24, y=0.92)
    plt.savefig(f"{CONFIG['plot_folder']}/{fnout}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--loc_name", type=str, required=False, help="Location name")
    parser.add_argument("--run_all", action="store_true", help="Flag for plant mode")
    parser.add_argument("--loc_tick", action="store_true", help="Include location tick")
    parser.add_argument("--tv", action="store_true", help="Use tv filter")
    args = parser.parse_args()
    loc_name, loc_tick, run_all, tv = args.loc_name, args.loc_tick, args.run_all, args.tv
    
    if run_all:
        folders = glob.glob(f"{CONFIG['results_folder']}/*")
        loc_names = [k.split('/')[-1] for k in folders]

        for ln in loc_names:
            print(f"Starting {ln}")
            get_plot(ln, loc_tick, tv)
            print(f"Done with {ln}")
    else:
        print(f"Starting {loc_name}")
        get_plot(loc_name, loc_tick, tv)
        print(f"Done with {loc_name}")
