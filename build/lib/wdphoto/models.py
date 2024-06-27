import re
import pickle
import glob
import os
import numpy as np
import scipy

from scipy.interpolate import RegularGridInterpolator
from astropy.table import Table


# convert air wavelengths to vacuum
def air2vac(wv):
    _tl=1.e4/np.array(wv)
    return (np.array(wv)*(1.+6.4328e-5+2.94981e-2/\
                          (146.-_tl**2)+2.5540e-4/(41.-_tl**2)))

def refactor_array(fl):
    # re-format an array of strings into floats
    for jj, num in enumerate(fl):
        if 'E' not in num: # if there is no exponential
            if '+' in num:
                num = num.split('-')
                num = 'E'.join(num)
            elif ('-' in num) and (num[0] != '-'):
                num = num.split('-')
                num = 'E-'.join(num)
            elif ('-' in num) and (num[0] == '-'):
                num = num.split('-')
                num = 'E-'.join(num)
                num = num[1:]
            fl[jj] = num

    try:
        fl = np.array([float(val) for val in fl])
    except:
        fl = fl[1:]
        fl = np.array([float(val) for val in fl])
    return fl


# create the Warwick DA interpolator
def build_warwick_da(path = '/data/warwick_da', outpath = None, flux_unit = 'flam'):
    dirpath = os.path.dirname(os.path.realpath(__file__)) # identify the current directory
    files = glob.glob(dirpath + path + '/*') # get a list of the files corresponding to the Warwick DA spectra

    # read in the first file
    with open(files[0]) as f:
        lines = f.read().splitlines()
        
    for ii in range(len(lines)):        
        if 'Effective temperature' in lines[ii]: # iterate over lines until the first line with actual data is found
            # create an array of all the data up until that line to create the base wavelength grid
            base_wavl = np.array(re.split('\s+', ''.join(lines[1:ii])))[1:].astype(float) 
            break

    # instantiate holder objects
    dat = []        
                
    for file in files:
        # read each file in
        with open(file) as f:
            lines = f.read().splitlines()
                
        prev_ii = 0 # at what index do the file's wavelengths end?
        first = True # is this the first pass of that file?
        for ii in range(len(lines)):        
            if 'Effective temperature' in lines[ii]:
                if first: # if not done already, generate an array of the file's wavelength coverage
                    wavl = np.array(re.split('\s+', ''.join(lines[1:ii])))[1:].astype(float)

                    # update variables
                    first = False
                    prev_ii = ii

                    # store the temperature and logg of this part of the file
                    teff = float(re.split('\s+', lines[prev_ii])[4])
                    logg = np.log10(float(re.split('\s+', lines[prev_ii])[7]))

                else: # if this is not the first time,
                    # store the temperature and logg of this part of the file
                    teff = float(re.split('\s+', lines[prev_ii])[4])
                    logg = np.log10(float(re.split('\s+', lines[prev_ii])[7]))

                    # create a list of all the string fluxes from the last break to this one
                    fl = re.split('\s+', ''.join(lines[prev_ii+1:ii]))
                    fl = refactor_array(fl) # refactor the array from strings to floats
                    fl = np.interp(base_wavl, wavl, fl) # interpolate flux onto a consistent wavelength grid
                        
                    dat.append([logg, teff, fl, base_wavl]) # append to a holder for later use                                             
                    prev_ii = ii # update to the new end of the flux data
                
    table = Table() # create a table to hold this in 
    table['logg'] = np.array(dat, dtype=object).T[0]
    table['teff'] = np.array(dat, dtype=object).T[1] 
    table['fl'] = np.array(dat, dtype=object).T[2]   
    table['wl'] = air2vac(np.array(dat, dtype=object).T[3]) # convert air wavelengths to vacuum
    
    if flux_unit == 'flam': # convert to flam if desired
        table['fl'] = (2.99792458e18*table['fl'] / table['wl']**2)
    
    # create a sorted list of teffs and loggs for use in the array
    teffs = sorted(list(set(table['teff'])))
    loggs = sorted(list(set(table['logg'])))
    
    # instantiate the interpolation object
    values = np.zeros((len(teffs), len(loggs), len(base_wavl)))
    
    for i in range(len(teffs)):
        for j in range(len(loggs)):
            try:
                # append the value of flux corresponding to the current (teff, logg)
                values[i,j] = table[np.all([table['teff'] == teffs[i], table['logg'] == loggs[j]], axis = 0)]['fl'][0]
            except:
                # if that isn't included, append zeros
                values[i,j] = np.zeros(len(base_wavl))
    
    #NICOLE BUG FIX
    high_logg_grid=values[:,4:]
    high_loggs=loggs[4:]

    low_logg_grid=values[16:33,:]
    low_loggs_teffs=teffs[16:33]

    model_spec = RegularGridInterpolator((teffs, high_loggs), high_logg_grid)
    model_spec_low_logg = RegularGridInterpolator((low_loggs_teffs, loggs), low_logg_grid)
    
    if outpath is not None:
        # open a file, where you ant to store the data
        interp_file = open(outpath + '/warwick_da.pkl', 'wb')
        
        # dump information to that file
        pickle.dump(model_spec, interp_file)
        np.save(outpath + '/warwick_da_wavl', base_wavl)
        
    return base_wavl, model_spec, model_spec_low_logg, table

def read_cache(table):
    cache = Table.read(table)
    
    teff = np.unique(np.array(cache['teff']))
    logg = np.unique(np.array(cache['logg']))
    
    key_dict = {}
    raw_data = np.zeros((len(teff), len(logg), 271))
    
    #test.remove_columns(['teff', 'logg'])
    for i in range(len(teff)):
        for j in range(len(logg)):
            teff_loc = np.where(cache['teff'] == teff[i])[0]
            logg_loc = np.where(cache['logg'] == logg[j])[0]
    
            k = np.intersect1d(teff_loc, logg_loc)[0]
            raw_data[i,j] = [cache[key][k] for key in cache.keys()[2:]]
            
    for ii, key in enumerate(cache.keys()[2:]):
        key_dict[key] = ii
            
    model_sed = RegularGridInterpolator((teff, logg), raw_data)
    
    return model_sed, key_dict
    
    