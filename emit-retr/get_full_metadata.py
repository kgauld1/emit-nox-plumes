import sys, os, glob, warnings
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

from config import CONFIG
from REFERENCE_PLANTS import REFERENCE_PLANTS

sys.path.append('../datasets/')
from get_geosfp import get_geosfp_wind
from get_hrrr import get_hrrr_wind_10m, get_hrrr_wind_agl
from get_campd import get_emission_rate

FILTERED_LOCATIONS = [
    "Alcoa_Allowance_Management_Inc",
    "Belews_Creek",
    "Colstrip",
    "Fort_Martin_Power_Station",
    "Gerald_Gentleman_Station",
    "Ghent",
    "Intermountain",
    "Labadie",
    "Laramie_River",
    "Limestone",
    "Martin_Lake",
    "Mill_Creek",
    "New_Madrid_Power_Plant",
    "Ninemile_Point",
    "Scherer",
    "Shawnee",
    "W_A_Parish",
    "Thomas_Hill_Energy_Center"
]

SAVE_COLS = ['LOC_NAME', 'GRANULE', 'LON', 'LAT', 'CAMPD_RATE',
             'HRRR_AGL_DIR', 'HRRR_AGL_SPD',
             'HRRR_10M_DIR', 'HRRR_10M_SPD',
             'GEOSFP_50M_DIR', 'GEOSFP_50M_SPD']


csv_savefn = f"{CONFIG['data_folder']}/metadata_FULL.csv"

if os.path.isfile(csv_savefn):
    data = pd.read_csv(csv_savefn)
else:
    data = pd.DataFrame(columns=SAVE_COLS)

def get_infodict(loc_name, granule):
    infodict = {}
    for k in SAVE_COLS: infodict[k] = None
    infodict['LOC_NAME'] = loc_name
    infodict['GRANULE'] = granule

    loc_info = REFERENCE_PLANTS[loc_name]
    ltlon, ltlat = loc_info['LON'], loc_info['LAT']
    infodict['LON'], infodict['LAT'] = ltlon, ltlat

    obs_time = datetime.strptime(granule.split('_')[0], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    
    geosfp_info = get_geosfp_wind(ltlat, ltlon, obs_time, cache=f'{CONFIG["geosfp"]}/')
    infodict['GEOSFP_50M_DIR'], infodict['GEOSFP_50M_SPD'] = float(geosfp_info["DIR50"]), float(geosfp_info["U50"])
    
    hrrr_agl_info = get_hrrr_wind_agl(ltlat, ltlon, obs_time, layer=(200,600), cache=f'{CONFIG["hrrr"]}')
    infodict['HRRR_AGL_DIR'], infodict['HRRR_AGL_SPD'] = float(hrrr_agl_info["dir_from_deg"]), float(hrrr_agl_info["speed_ms"])
    
    hrrr_10m_info = get_hrrr_wind_10m(ltlat, ltlon, obs_time, cache=f'{CONFIG["hrrr"]}')
    infodict['HRRR_10M_DIR'], infodict['HRRR_10M_SPD'] = float(hrrr_10m_info["dir_from_deg"]), float(hrrr_10m_info["speed_ms"])
    
    infodict['CAMPD_RATE'] = get_emission_rate(loc_info, obs_time, timedelta(hours=4))

    return infodict

for loc_name in FILTERED_LOCATIONS:
    print(loc_name)
    fns = glob.glob(f"{CONFIG['data_folder']}/{loc_name}/*RAD*")
    granules = ['_'.join(k.split('/')[-1].split('_')[4:7]).split('.')[0] for k in fns]
    for g in granules:
        if g in data['GRANULE'].values:
            print(f"already populated {g}, skipping")
            continue
    
        g_dat = get_infodict(loc_name, g)
        data = pd.concat([data, pd.DataFrame(g_dat, index=[0])])
        data.to_csv(csv_savefn, index=False)