import os
import earthaccess
import sys

import argparse

from config import CONFIG, POWER_PLANTS, LOCS
from datetime import timedelta, datetime


def get_L1B_granules(lat, lon, starttime, endtime, cloud_cover=(0,90), count=100):
    
    return results


def download_granules(results, data_folder, fs):
    for rf in results:
        result_urls = rf.data_links()
        for url in result_urls:
            granule_asset_id = url.split('/')[-1]
            
            # Download the Granule Asset if it doesn't exist
            fp = f'{data_folder}/{granule_asset_id}'
            if not os.path.isfile(fp):
                with fs.get(url,stream=True) as src:
                    with open(fp,'wb') as dst:
                        for chunk in src.iter_content(chunk_size=64*1024*1024):
                            dst.write(chunk)
            else:
                print(f"{url} already downloaded. Skipping...")
                continue
            print(f"DOWNLOADED {url} to {fp}")
    
def process_plants(args, fs):
    if not args.download:
        print("DOWNLOAD NOT REQUESTED.")
    
    for loc_name, md in POWER_PLANTS.items():
        lon, lat = md['LON'], md['LAT']
        
        results = earthaccess.search_data(
                short_name='EMITL1BRAD',
                point=(lon, lat),
                temporal=(args.starttime, args.endtime),
                cloud_cover=(0, args.cloud_cover),
                count=args.count
            )
        
        print(f"FOUND {len(results)} EMIT GRANULES FOR {loc_name} AT {lat:0.3f}, {lon:0.3f}")
        if args.download:
            save_path = f"{args.data_folder}/{loc_name}"
            os.makedirs(save_path, exist_ok=True)
            
            download_granules(results, save_path, fs)
            print("DONE DOWNLOADING GRANULES")

if __name__ == "__main__":
    auth = earthaccess.login(persist=True)
    fs = earthaccess.get_requests_https_session()
    
    if auth.authenticated:
        print("Authenticated to EarthData")
    else:
        raise Exception("Unable to authenticate to EarthData")
    
    parser = argparse.ArgumentParser(description="Download satellite data for a given location and time range.")

    # Required
    parser.add_argument("--lat", type=float, help="Latitude of the location")
    parser.add_argument("--lon", type=float, help="Longitude of the location")
    parser.add_argument("--loc_name", type=str, help="Location name")\

    # Optional
    parser.add_argument(
        "--starttime",
        type=str,
        default="20220101",
        help="Start date in YYYYMMDD format (default: 20220101)"
    )
    parser.add_argument(
        "--endtime",
        type=str,
        default=datetime.now().strftime("%Y%m%d"),
        help="End date in YYYYMMDD format (default: current date)"
    )
    parser.add_argument(
        "--cloud_cover",
        type=int,
        default=90,
        help="Maximum cloud cover percentage (default: 90)"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Maximum number of results to fetch (default: 100)"
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        default=CONFIG['data_folder'],
        help='Path to data folder (default to config file)'
    )
    parser.add_argument(
        "--download",
        type=lambda x: str(x).lower() in ["true", "1", "yes", "y"],
        default=False,
        help="Whether to download the data (default: True)"
    )
    
    parser.add_argument(
        "--grab_all_plants",
        type=lambda x: str(x).lower() in ["true", "1", "yes", "y"],
        default=False,
        help="Whether to use all power plants"
    )

    args = parser.parse_args()
    
    if args.grab_all_plants:
        process_plants(args, fs)
        quit()
    
    if not ((args.lat and args.lon) or args.loc_name):
        raise Exception("Must specify either lat/lon or a location name")
        
    if args.loc_name:
        lon = LOCS[args.loc_name]['LON']
        lat = LOCS[args.loc_name]['LAT']
        loc_name = args.loc_name
    else:
        lon = args.lon
        lat = args.lat
        loc_name = f"LOC_{lat:0.3f}_{lon:0.3f}"
    
    results = earthaccess.search_data(
            short_name='EMITL1BRAD',
            point=(lon, lat),
            temporal=(args.starttime, args.endtime),
            cloud_cover=(0, args.cloud_cover),
            count=args.count
        )
    
    print(f"FOUND {len(results)} EMIT GRANULES FOR {loc_name} AT {lat:0.3f}, {lon:0.3f}")
    if args.download:
        save_path = f"{args.data_folder}/{loc_name}"
        os.makedirs(save_path, exist_ok=True)
        
        download_granules(results, save_path, fs)
        print("DONE DOWNLOADING GRANULES")
    else:
        print("DOWNLOAD NOT REQUESTED. EXITING.")
