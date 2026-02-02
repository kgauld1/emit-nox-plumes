import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter, binary_erosion, binary_dilation
import holoviews as hv
import hvplot.xarray
import panel as pn
import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import distance_transform_edt
import sys, os
import argparse
import glob
import gc

from datetime import datetime, timedelta, timezone

from skimage.restoration import denoise_tv_chambolle, inpaint_biharmonic
from config import *
from REFERENCE_PLANTS import REFERENCE_PLANTS

sys.path.append('../EMIT-Data-Resources/python/modules/')
from emit_tools import emit_xarray, ortho_xr

sys.path.append('../datasets/')
from get_geosfp import get_geosfp_wind
from get_hrrr import get_hrrr_wind_10m, get_hrrr_wind_agl

def get_single_granule_plot(dSCD_fn, loc_name, loc_tick=False, wind_tick=False, tv_filter=False, use_ax=None):
    if not use_ax:
        fig = plt.figure(figsize=(16,16))
        ax = plt.gca()
    else:
        ax = use_ax

    if loc_tick:
        if loc_name in LOCS.keys():
            ltlat, ltlon = LOCS[loc_name]['LAT'], LOCS[loc_name]['LON']
        elif loc_name in REFERENCE_PLANTS.keys():
            ltlat, ltlon = REFERENCE_PLANTS[loc_name]['LAT'], REFERENCE_PLANTS[loc_name]['LON']
        else:
            loc_tick = False

    dSCD = np.load(dSCD_fn)
    gn_unmask = dSCD_fn.split('/')[-1]
    radfn = f"{CONFIG['data_folder']}/{loc_name}/{gn_unmask[5:-4]}.nc"
    granule_name = gn_unmask[-31:-4]
    

    try:
        ds_no_orth = emit_xarray(radfn)
    except:
        return
        
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

    rad1500 = ds['radiance'].sel(wavelengths=1500, method='nearest').data
    rad1500 = np.asarray(rad1500, dtype=np.float32)
    
    dscd0 = ds['dSCD'].isel(wavelengths=0).data
    dscd0 = np.asarray(dscd0, dtype=np.float32)
    
    dSCD_nan = dscd0.copy()
    dSCD_nan[rad1500 <= -8000] = np.nan

    if wind_tick:
        obs_time = datetime.strptime(granule_name.split('_')[0], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
        
        def plot_arrow(angle,color):
            theta = np.deg2rad((angle + 180.0) % 360.0)
            L = 0.1  # degrees
            
            ax.annotate(
                "", xy=(ltlon + L*np.sin(theta), ltlat + L*np.cos(theta)), 
                xytext=(ltlon, ltlat),
                arrowprops=dict(arrowstyle="->", linewidth=2, color=color),
                zorder=10
            )
        
        geosfp_info = get_geosfp_wind(ltlat, ltlon, obs_time, cache=f'{CONFIG["geosfp"]}/')
        plot_arrow(float(geosfp_info["DIR50"]), 'blue')

        hrrr_agl_info = get_hrrr_wind_agl(ltlat, ltlon, obs_time, layer=(200,600), cache=f'{CONFIG["hrrr"]}')
        plot_arrow(float(hrrr_agl_info["dir_from_deg"]), 'red')

        hrrr_10m_info = get_hrrr_wind_10m(ltlat, ltlon, obs_time, cache=f'{CONFIG["hrrr"]}')
        plot_arrow(float(hrrr_10m_info["dir_from_deg"]), 'green')

        ax.scatter([],[],c='blue',label=f'geosfp 50m {geosfp_info["U50"]:0.2f}')
        ax.scatter([],[],c='red',label=f'hrrr agl {hrrr_agl_info["speed_ms"]:0.2f}')
        ax.scatter([],[],c='green',label=f'hrrr 10m {hrrr_10m_info["speed_ms"]:0.2f}')

        ax.legend(
            loc="lower left",
            bbox_to_anchor=(0, 1.02),
            fontsize=14,
            markerscale=1.6,
            handlelength=2.5,
            labelspacing=0.4,
            frameon=True,   # or False if you prefer
        )
    
    if not tv_filter:
        ax.imshow(dSCD_nan*1e19, cmap='RdBu_r', origin='upper',
                      aspect='auto', vmin=-2e17, vmax=2e17, extent=bounds)
    else:
        mask = np.isfinite(dSCD_nan)
        # nearest-neighbor fill
        _, idx = distance_transform_edt(~mask, return_indices=True)
        filled = dSCD_nan[tuple(idx)]   # copies nearest valid value into each NaN pixel
        tv = denoise_tv_chambolle(filled, weight=0.2)
        tv[~mask] = np.nan
        
        ax.imshow(tv, cmap='RdBu_r', origin='upper',
                        aspect='auto', extent=bounds)
    
    ax.set_title(granule_name)
    
    try:
        valid_lat = ds['latitude'].values[np.where(~np.isnan(dSCD_nan))[0]]
        valid_lon = ds['longitude'].values[np.where(~np.isnan(dSCD_nan))[1]]

        latmin_bnd, latmax_bnd = valid_lat.min(), valid_lat.max()
        lonmin_bnd, lonmax_bnd = valid_lon.min(), valid_lon.max()

        ax.set_xlim(lonmin_bnd, lonmax_bnd)
        ax.set_ylim(latmin_bnd, latmax_bnd)
    except:
        pass

    del ds
    del dscd_da
    if loc_tick:
        ax.scatter([ltlon], [ltlat], marker='x', c='red')

    if not use_ax:
        fnout = f'{loc_name}_{granule_name}_dSCD'
        if loc_tick:
            fnout += '_ltick'
        if wind_tick:
            fnout += '_wdir'
        if tv_filter:
            fnout += '_tv'
        fnout += '.png'
        fig.suptitle(loc_name, fontsize=24, y=0.92)
        print(f"Saving {fnout}...")
        plt.savefig(f"{CONFIG['plot_folder']}/{loc_name}/{fnout}")
        plt.close(fig)
        gc.collect()

def get_plot(loc_name, loc_tick=False, wind_tick=False, tv_filter=False, combine_plot=False):
    fns = glob.glob(f"{CONFIG['results_folder']}/{CONFIG['retr_subdir']}/{loc_name}/*.npy")
    
    if combine_plot:
        cols = 5
        rows = (len(fns) + cols - 1)//cols
        fig, axs = plt.subplots(rows,cols, figsize=(cols*4,rows*4+5))
        axs = axs.flatten()
        for i in range(min(len(fns), rows*cols)):
            get_single_granule_plot(fns[i], loc_name, loc_tick=loc_tick, wind_tick=wind_tick, tv_filter=tv_filter, use_ax=axs[i])
        fnout = f'{loc_name}_dSCD_retrievals'
        if loc_tick:
            fnout += '_ltick'
        if wind_tick:
            fnout += '_wdir'
        if tv_filter:
            fnout += '_tv'
        fnout += '.png'
        
        fig.suptitle(loc_name, fontsize=24, y=0.92)
        plt.savefig(f"{CONFIG['plot_folder']}/{loc_name}/{fnout}")
    else:
        for i in range(len(fns)):
            get_single_granule_plot(fns[i], loc_name, loc_tick=loc_tick, wind_tick=wind_tick, tv_filter=tv_filter, use_ax=None)

def get_tavg_plot(loc_name):
    mean_stack = np.load(f"{CONFIG['results_folder']}/{CONFIG['tavg_subdir']}/{loc_name}/{loc_name}_mean_stack.npy")

    H,W = mean_stack.shape
    fig, axs = plt.subplots(1, 3, figsize=(15,5*H/W), layout='constrained')
    
    v = mean_stack[np.isfinite(mean_stack)]
    lim = np.nanpercentile(np.abs(v), 99.5)   # tighter than 99.5
    im = axs[0].imshow(mean_stack, cmap='RdBu_r', vmin=-lim, vmax=lim)
    plt.colorbar(im, ax=axs[0])
    axs[0].scatter(mean_stack.shape[1] // 2, mean_stack.shape[0] // 2, marker='x', s=50, c='green')
    axs[0].set_title('Raw dSCD')
    
    im = axs[1].imshow(gaussian_filter(mean_stack, 2), cmap="RdBu_r", vmin=-lim, vmax=lim)
    plt.colorbar(im, ax=axs[1])
    axs[1].scatter(mean_stack.shape[1] // 2, mean_stack.shape[0] // 2, marker='x', s=50, c='green')
    axs[1].set_title('Gaussian filter')
    
    
    mask = np.isfinite(mean_stack)
    # nearest-neighbor fill
    _, idx = distance_transform_edt(~mask, return_indices=True)
    filled = mean_stack[tuple(idx)]   # copies nearest valid value into each NaN pixel
    tv = denoise_tv_chambolle(filled, weight=0.2)
    tv[~mask] = np.nan
    
    
    im = axs[2].imshow(tv, cmap='RdBu_r', origin='upper', vmin=-0.001, vmax=0.001)
    plt.colorbar(im, ax=axs[2])
    axs[2].scatter(mean_stack.shape[1] // 2, mean_stack.shape[0] // 2, marker='x', s=50, c='green')
    axs[2].set_title('TV filter')
    
    fig.suptitle(f"{loc_name} Time Averaged Retrieval", fontsize=20)
    plt.savefig(f"{CONFIG['plot_folder']}/{loc_name}/{loc_name}_Time_Average.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--loc_name", type=str, required=False, help="Location name")
    parser.add_argument("--run_all", action="store_true", help="Flag for plant mode")
    parser.add_argument("--loc_tick", action="store_true", help="Include location tick")
    parser.add_argument("--wind_tick", action="store_true", help="Include wind direction")
    parser.add_argument("--combine_plot", action="store_true", help="Combine all location plots")
    parser.add_argument("--tv", action="store_true", help="Use tv filter")
    parser.add_argument("--tavg", action="store_true", help="Use time avg")
    args = parser.parse_args()

    if args.tavg:
        if args.run_all:
            with open("targets1", "r") as f:
                target_names = f.read().splitlines()
            for ln in target_names:
                print(f"Starting {ln}")
                get_tavg_plot(ln)
                print(f"Done with {ln}")
        else:
            print(f"Starting {args.loc_name}")
            get_tavg_plot(args.loc_name)
            print(f"Done with {args.loc_name}")
        quit()
    
    if args.run_all:
        folders = glob.glob(f"{CONFIG['results_folder']}/{CONFIG['retr_subdir']}/*")
        loc_names = [k.split('/')[-1] for k in folders]

        for ln in loc_names:
            print(f"Starting {ln}")
            os.makedirs(f"{CONFIG['plot_folder']}/{ln}/", exist_ok=True)
            get_plot(ln, args.loc_tick, args.wind_tick, args.tv, args.combine_plot)
            print(f"Done with {ln}")
    else:
        print(f"Starting {args.loc_name}")
        os.makedirs(f"{CONFIG['plot_folder']}/{args.loc_name}/", exist_ok=True)
        get_plot(args.loc_name, args.loc_tick, args.wind_tick, args.tv, args.combine_plot)
        print(f"Done with {args.loc_name}")
