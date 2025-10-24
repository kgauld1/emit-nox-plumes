from EMIT_NOX import run_retrieval
import argparse
import numpy as np
import glob
import os

from config import CONFIG, POWER_PLANTS, CROSS_SECTIONS, LOCS

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--loc_name", type=str, required=True, help="Location name")
    # parser.add_argument("--plant", action="store_true", help="Flag for plant mode")
    args = parser.parse_args()
    
    target_name = args.loc_name
    
    # if args.plant:
    #     target_data =  POWER_PLANTS[target_name]
    # else:
    #     target_data = LOCS[target_name]

    samples = glob.glob(f"{CONFIG['data_folder']}/{target_name}/*_RAD_*.nc")

    for fn in samples:
        granule_name = fn.split('/')[-1]
        
        result_dSCD = run_retrieval(fn)
        
        save_path = f"{CONFIG['results_folder']}/{target_name}"
        os.makedirs(save_path, exist_ok=True)
        np.save(f"{save_path}/dSCD_{granule_name.split('.')[0]}.npy", result_dSCD)
        
        print(f"Saved product to {save_path}/dSCD_{granule_name.split('.')[0]}.npy !")