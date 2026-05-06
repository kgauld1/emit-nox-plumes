CONFIG = {
    'data_folder': '/orcd/data/dvaron/001/kgauld/EMIT/data',
    'results_folder': '/orcd/data/dvaron/001/kgauld/EMIT/results6',
    'plot_folder': '/orcd/data/dvaron/001/kgauld/EMIT/plots4_rgb',
    'geosfp': '/orcd/data/dvaron/001/kgauld/GEOS_FP',
    'hrrr': '/orcd/data/dvaron/001/kgauld/HRRR',
    'tropomi': '/orcd/data/dvaron/001/kgauld/TROPOMI',
    'campd_key': '/orcd/pool/005/dvaron_shared/kgauld/emit-nox-plumes/secrets/CAMPD_APIKEY',

    'retr_subdir': 'Retrievals',
    'tavg_subdir': 'Time_Average',
    'ps_subdir': 'Point_Source',

    'NOX_CSEC': '/orcd/home/002/kgauld/emit-nox-plumes/cross_sections/no2c_97.txt',
    'AMF_LUT': '/orcd/data/dvaron/001/kgauld/AMF/differential_amf.nc'
}

LOCS = {
    'ARIZONA_MINE': {
        'LAT': 33.048083799737455,
        'LON': -109.35166351714791
    },
    'SLC_MINE': {
        'LAT': 40.528234085750384,
        'LON': -112.14439046068979
    },
    'RIYADH_PLANT_9': {
        'LAT': 24.95005929724472, 
        'LON': 47.065018186187366
    },
    'RIYADH_PLANT_10': {
        'LAT': 24.420813526042448, 
        'LON': 47.01991290531597
    },
    'RIYADH_PLANT_7': {
        'LAT': 24.567585604791905, 
        'LON': 46.8816170455378
    },
    'RIYADH_QURAYYAH': {
        'LAT': 25.844068671174213, 
        'LON': 50.125993222806876
    },
    'XAI_CENTER': {
        'LAT': 35.059787111932955, 
        'LON': -90.15624976124313
    },
    'SPACEX_PAD': {
        'LAT': 25.997199872699298, 
        'LON': -97.15631326442413
    },
    'TURKMENISTAN': {
        'LAT': 39.39108288758638,
        'LON': 53.832978540033835
    },
    'MATLA_AFRICA': {
        'LON': 29.141089005950896,
        'LAT': -26.281600366206224
    },
    'OPENAI_STARGATE': {
        'LON': -99.78911808370637,
        'LAT': 32.501439448057695
    },
    'ANTHROPIC_CARLISLE': {
        'LON': -86.46509966844667,
        'LAT': 41.691712949609105
    },
    'META_PROMETHEUS': {
        'LON': -82.75000326402957, 
        'LAT': 40.06702853903578
    },
    'XAI_PP': {
        'LAT': 34.9809316261756, 
        'LON': -90.03976114417924
    }
}


PREFIXES = {'OBS': 'EMIT_L1B_OBS_001_',
 'L1B': 'EMIT_L1B_RAD_001_',
 'MASK': 'EMIT_L2A_MASK_001_',
 'L2A': 'EMIT_L2A_RFL_001_'}