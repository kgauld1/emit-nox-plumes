from EMIT_NOX import run_retrieval
import argparse
import numpy as np
import glob
import os

from config import CONFIG, POWER_PLANTS, CROSS_SECTIONS, LOCS


def multipass_retrieval(target_name, samples):
    try:
        clat, clon = LOCS[target_name]['LAT'], LOCS[target_name]['LON']
    except:
        clat, clon = POWER_PLANTS[target_name]['LAT'], POWER_PLANTS[target_name]['LON']

    dSCDs, masks, lats, lons, s = [], [], [], [], []
    
    for i, fn in enumerate(samples):
        try:
            result_dSCD, result_mask, result_lat, result_lon = run_retrieval(fn, clat=clat, clon=clon, km_boundary = 10)
            dSCDs.append(result_dSCD)
            masks.append(result_mask)
            lats.append(result_lat)
            lons.append(result_lon)
            s.append(True)
        except ValueError as e:
            print(f"FAILED {fn}: {e}")
            dSCDs.append(None)
            masks.append(None)
            lats.append(None)
            lons.append(None)
            s.append(False)

    return dSCDs, masks, lats, lons, s
                                                                         

def singlepass_retrieval(target_name, fn, save_file=True):
    granule_name = fn.split('/')[-1]
    result_dSCD, lat, lon, result_mask = run_retrieval(fn)

    if save_file:
        save_path = f"{CONFIG['results_folder']}/{target_name}"
        os.makedirs(save_path, exist_ok=True)
        np.save(f"{save_path}/dSCD_{granule_name.split('.')[0]}.npy", result_dSCD)
        
        print(f"Saved product to {save_path}/dSCD_{granule_name.split('.')[0]}.npy !")

    return result_dSCD, lat, lon, result_mask

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--loc_name", type=str, required=True, help="Location name")
    parser.add_argument("--multipass", action="store_true", help="Flag for plant mode")
    args = parser.parse_args()
    
    target_name = args.loc_name
    samples = glob.glob(f"{CONFIG['data_folder']}/{target_name}/*_RAD_*.nc")
    
    if multipass:
        multipass_retrieval(target_name, samples)
    else:
        for fn in samples:
            singlepass_retrieval(target_name, fn)