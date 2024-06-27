from scipy.interpolate import RegularGridInterpolator
from astropy.table import Table
import numpy as np

def read_cache(table, ignore_cols = []):
    """
    Args:
        table (filepath): a table that contains keys `teff`, `logg`, and photometry with names equivalent to those from pyphot
    Returns:
        table_interp (RegularGridInterpolator): interpolates teff and logg to a vector containing photometry
        key_dict (dict): sends pyphot photometry keys to the index of the vector output of table_interp
    """
    cache = Table.read(table) # read the table

    ignore_cols = ignore_cols + ['teff', 'logg']
    photo_keys = [i for i in cache.keys() if i not in ignore_cols] # list the keys that are not teff or logg (i.e. photo bands)

    teff = np.unique(np.array(cache['teff'])) # create a list of unique teffs
    logg = np.unique(np.array(cache['logg'])) # create a list of unique loggs
    
    key_dict = {} # instantiate empty key dictionary
    raw_data = np.zeros((len(teff), len(logg), len(photo_keys)))
    
    # iterate over all unique teffs and loggs
    for i in range(len(teff)):
        for j in range(len(logg)):
            # find the location in the cache table that has that unique logg and teff
            teff_loc = np.where(cache['teff'] == teff[i])[0]
            logg_loc = np.where(cache['logg'] == logg[j])[0]
            k = np.intersect1d(teff_loc, logg_loc)[0]

            # read the photometries into the raw_data
            raw_data[i,j] = [cache[key][k] for key in photo_keys]
            
    # now, create the cache table
    for ii, key in enumerate(photo_keys):
        key_dict[key] = ii # save the index of that key
            
    table_interp = RegularGridInterpolator((teff, logg), raw_data) # create the interpolator object
    return table_interp, key_dict