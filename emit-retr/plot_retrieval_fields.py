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
from scipy.ndimage import gaussian_filter
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

def rgb_stretch(rgb_ds, qlo=2, qhi=98, gamma=2.2, white_background=False):
    da = rgb_ds["radiance"]  # or "reflectance" if you have it
    out = []
    for wl in da["wavelengths"].values:
        b = da.sel(wavelengths=wl)

        # mask nonpositive / invalid
        b = b.where(b > 0)
        lo = b.quantile(qlo/100.0)
        hi = b.quantile(qhi/100.0)

        b = (b - lo) / (hi - lo)
        b = b.clip(0, 1)

        # gamma correction (display gamma)
        b = b ** (1/gamma)
        if white_background:
            b = b.fillna(1.0)
        out.append(b)

    da_out = xr.concat(out, dim="wavelengths")
    da_out = da_out.assign_coords(wavelengths=da["wavelengths"])

    rgb_ds = rgb_ds.copy()
    rgb_ds["radiance"] = da_out
    return rgb_ds

def crop_about_loc(ds, clat, clon, km_boundary=None, pix_boundary=None):
    if km_boundary is None and pix_boundary is None:
        return np.ones_like(ds['radiance'])
    
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

def get_rgb(loc_name, granule, km_boundary=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    fp = f"{CONFIG['data_folder']}/{loc_name}/EMIT_L1B_RAD_001_{granule}.nc"
    try:
        lat = REFERENCE_PLANTS[loc_name]['LAT']
        lon = REFERENCE_PLANTS[loc_name]['LON']
    except:
        lat = LOCS[loc_name]['LAT']
        lon = LOCS[loc_name]['LON']
    # Load and orthorectify
    ds_geo = emit_xarray(fp, ortho=False)
    if km_boundary is not None:
        mask=crop_about_loc(ds_geo, lat, lon, km_boundary)
    ds_geo = ortho_xr(ds_geo)

    # Select RGB bands
    rgb = ds_geo.sel(wavelengths=[700, 529, 470], method="nearest")
    rgb = rgb_stretch(rgb, qlo=2, qhi=98, gamma=2.2, white_background=True)

    # Crop if bounds provided
    
    # Convert to numpy image
    img = rgb.transpose("latitude","longitude","wavelengths").to_array().values
    img = np.moveaxis(img, 0, -1).astype(float)
    img = np.squeeze(img)          # drops any trailing 1-dims
    img = img[..., :3]             # defensive: ensure 3 channels if something weird
    extent = [
        float(rgb.longitude.min()),
        float(rgb.longitude.max()),
        float(rgb.latitude.min()),
        float(rgb.latitude.max()),
    ]

    ax.imshow(img, extent=extent, origin="upper", aspect="auto")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    return ax

def get_single_granule_rgb(loc_name, granule_name, loc_tick=False, wind_tick=False):
    fig = plt.figure(figsize=(16,16))
    ax = plt.gca()
    get_rgb(loc_name, granule_name, ax=ax)
    if loc_tick:
        if loc_name in LOCS.keys():
            ltlat, ltlon = LOCS[loc_name]['LAT'], LOCS[loc_name]['LON']
        elif loc_name in REFERENCE_PLANTS.keys():
            ltlat, ltlon = REFERENCE_PLANTS[loc_name]['LAT'], REFERENCE_PLANTS[loc_name]['LON']
        else:
            loc_tick = False
        ax.scatter([ltlon], [ltlat], marker='x', c='red', s=150)
    fnout = f'{loc_name}_{granule_name}_rgb'
    if loc_tick:
        fnout += '_ltick'
    fnout += '.png'
    fig.suptitle(loc_name, fontsize=24, y=0.92)
    print(f"Saving {fnout}...")
    
    os.makedirs(f"{CONFIG['plot_folder']}_rgb", exist_ok=True)
    os.makedirs(f"{CONFIG['plot_folder']}_rgb/{loc_name}", exist_ok=True)
    
    plt.savefig(f"{CONFIG['plot_folder']}_rgb/{loc_name}/{fnout}")
    plt.close(fig)
    gc.collect()
    
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

        # hrrr_agl_info = get_hrrr_wind_agl(ltlat, ltlon, obs_time, layer=(200,600), cache=f'{CONFIG["hrrr"]}')
        # plot_arrow(float(hrrr_agl_info["dir_from_deg"]), 'red')

        # hrrr_10m_info = get_hrrr_wind_10m(ltlat, ltlon, obs_time, cache=f'{CONFIG["hrrr"]}')
        # plot_arrow(float(hrrr_10m_info["dir_from_deg"]), 'green')

        ax.scatter([],[],c='blue',label=f'geosfp 50m {geosfp_info["U50"]:0.2f}')
        # ax.scatter([],[],c='red',label=f'hrrr agl {hrrr_agl_info["speed_ms"]:0.2f}')
        # ax.scatter([],[],c='green',label=f'hrrr 10m {hrrr_10m_info["speed_ms"]:0.2f}')

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

        vmax = np.nanpercentile(tv,99.7)
        
        # ax.imshow(tv, cmap='RdBu_r', origin='upper',
        #           vmin=-vmax, vmax=vmax,
        #                 aspect='auto', extent=bounds)
        im = ax.imshow(tv, cmap='YlOrRd', origin='upper',
                  vmin=0, vmax=vmax,
                        aspect='auto', extent=bounds)
        # Z_s   = ndi.gaussian_filter(dSCD_nan, sigma=1.5)
        # im=ax.imshow(Z_s, cmap='YlOrRd', origin='upper', 
        #           vmin=0, vmax=np.nanpercentile(Z_s, 98), 
        #           aspect='auto', extent=bounds)
        plt.colorbar(im, ax=ax)

    
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

def get_plot(loc_name, loc_tick=False, wind_tick=False, tv_filter=False, combine_plot=False, rgb=False):
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
    elif rgb:
        for i in range(len(fns)):
            gn_unmask = fns[i].split('/')[-1]
            granule_name = gn_unmask[-31:-4]
            get_single_granule_rgb(loc_name, granule_name, loc_tick=loc_tick)
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
    parser.add_argument("--rgb", action="store_true")
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
            if not args.rgb:
                os.makedirs(f"{CONFIG['plot_folder']}/{ln}/", exist_ok=True)
            get_plot(ln, args.loc_tick, args.wind_tick, args.tv, args.combine_plot, args.rgb)
            print(f"Done with {ln}")
    else:
        print(f"Starting {args.loc_name}")
        if not args.rgb:
            os.makedirs(f"{CONFIG['plot_folder']}/{args.loc_name}/", exist_ok=True)
        get_plot(args.loc_name, args.loc_tick, args.wind_tick, args.tv, args.combine_plot, args.rgb)
        print(f"Done with {args.loc_name}")
