import os
import earthaccess
import numpy as np
import pandas as pd
import geopandas as gp
from shapely.geometry.polygon import orient
import xarray as xr
import sys

from datetime import timedelta
import numpy as np
import pandas as pd
import glob


#### FROM JOE PALMO

#### planet4589.org data
df = pd.read_csv('https://planet4589.org/space/gcat/tsv/launch/launch.tsv', sep='\t',)
sites_df = pd.read_csv('https://planet4589.org/space/gcat/tsv/tables/sites.tsv', sep='\t')

#### FUNCTIONS
def clean_and_parse_date(date_str):
    try:
        # Remove invalid characters like "?"
        cleaned_date = date_str.replace('?', '').strip()

        # Check if the format includes seconds
        if len(cleaned_date.split()) == 4 and ':' in cleaned_date:
            return pd.to_datetime(cleaned_date, format='%Y %b %d %H%M:%S')
        # Check if the format includes hours and minutes
        elif len(cleaned_date.split()) == 4:
            return pd.to_datetime(cleaned_date, format='%Y %b %d %H%M')
        # Check if it's just year, month, day
        elif len(cleaned_date.split()) == 3:
            return pd.to_datetime(cleaned_date, format='%Y %b %d')
        # Check if it's just year, month 
        
        ## IGNORE ANYTHING COARSER THAN A DAY
        # elif len(cleaned_date.split()) == 2 and 'Q' not in cleaned_date:
        #     return pd.to_datetime(cleaned_date, format='%Y %b')
        # # Check if it's just year, quarter
        # elif len(cleaned_date.split()) == 2 and 'Q' in cleaned_date:
        #     # Extract the year for quarters
        #     return int(cleaned_date.split()[0])  # Return the year only
        # Check if it's just year
        # elif len(cleaned_date.split()) == 1:
        #     return pd.to_datetime(cleaned_date, format='%Y')
        else:
            return np.nan  # Return NaN for invalid entries
    except Exception:
        return np.nan

# given the Launch_Site, find the corresponding lat and lon
def get_lat(site):
    try:
        return sites_df[sites_df['#Site'] == site]['Latitude'].values[0]
    except Exception:
        return np.nan
    
def get_lon(site):
    try:
        return sites_df[sites_df['#Site'] == site]['Longitude'].values[0]
    except Exception:
        return np.nan

#### DATA CLEANING

def get_launch_df():
    auth = earthaccess.login(persist=True)
    
    # Apply the cleaning and parsing function
    df['timestamp'] = df['Launch_Date'].apply(clean_and_parse_date)
    df.dropna(subset=['timestamp'], inplace=True)
    # set timestamp index
    df.set_index('timestamp', inplace=True)
    df.index = pd.DatetimeIndex(df.index)
    df.sort_index(inplace=True)

    # only launches during TEMPO mission
    recent_df = df.loc['2022-07-14':]

    # get the lat and lon for each launch site
    recent_df['lat'] = recent_df['Launch_Site'].apply(get_lat).astype(float)
    recent_df['lon'] = recent_df['Launch_Site'].apply(get_lon).astype(float)

    return recent_df

if __name__ == '__main__':
    rocket_df = get_launch_df()
    
    fs = earthaccess.get_requests_https_session()

    for t, row in rocket_df.iterrows():
        starttime = (t-timedelta(minutes=5)).strftime("%Y%m%dT%H%M")
        endtime = (t+timedelta(hours=1)).strftime("%Y%m%dT%H%M")
        try:
            results = earthaccess.search_data(
                short_name='EMITL1BRAD',
                point=(row['lon'], row['lat']),
                temporal=(starttime, endtime),
                cloud_cover=(0,100),
                count=100
            )
            rocket_df.loc[t, 'emit'] = len(results)
        except:
            rocket_df.loc[t, 'emit'] = -1
        
        rocket_df.to_csv('rocket_df_withemit_temp.csv')