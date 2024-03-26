import numpy as np
from bisect import bisect_left
import scipy
import matplotlib.pyplot as plt
from astropy import constants as c

import re
import pickle
from scipy.interpolate import RegularGridInterpolator
from astropy.table import Table
import glob
import os

def air2vac(wv):
    _tl=1.e4/np.array(wv)
    return (np.array(wv)*(1.+6.4328e-5+2.94981e-2/\
                          (146.-_tl**2)+2.5540e-4/(41.-_tl**2)))

def build_warwick_da(path = '/models/warwick_da', outpath = None, flux_unit = 'flam'):
    dirpath = os.path.dirname(os.path.realpath(__file__))
    files = glob.glob(dirpath + path + '/*')
    

    with open(files[0]) as f:
        lines = f.read().splitlines()
    
    table = Table()
    dat = []
    
    with open(files[0]) as f:
        lines = f.read().splitlines()
        
    base_wavl = []
        
    for ii in range(len(lines)):        
        if 'Effective temperature' in lines[ii]:
            base_wavl = np.array(re.split('\s+', ''.join(lines[1:ii])))[1:].astype(float)
            break                
                
    for file in files:
        with open(file) as f:
            lines = f.read().splitlines()
                
        prev_ii = 0
            
        first = True
        
        for ii in range(len(lines)):        
            if 'Effective temperature' in lines[ii]:
                if first:
                    wavl = np.array(re.split('\s+', ''.join(lines[1:ii])))[1:].astype(float)
                    first = False
                    prev_ii = ii
                    continue
                    
                #print(prev_ii)
                                
                teff = float(re.split('\s+', lines[prev_ii])[4])
                logg = np.log10(float(re.split('\s+', lines[prev_ii])[7]))
                
                if not first:
                    fl = re.split('\s+', ''.join(lines[prev_ii+1:ii]))
                    for jj, num in enumerate(fl):
                        if 'E' not in num:
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
                        
                    dat.append([logg, teff, np.interp(base_wavl, wavl, fl), base_wavl])                                                   
                    #fls[str(teff) + ' ' + str(logg)] = fl
                    
                    prev_ii = ii
        
        
    default_centres =  dict(a = 6564.61, b = 4862.68, g = 4341.68, d = 4102.89,
                     e = 3971.20, z = 3890.12, n = 3835.5,
                 t = 3799.5)
    default_windows = dict(a = 100, b = 100, g = 85, d = 70, e = 30,
                      z = 25, n = 15, t = 10)
    default_edges = dict(a = 25, b = 25, g = 20, d = 20, 
                    e = 5, z = 5, n = 5, t = 4)
    
    default_names = ['n', 'z', 'e', 'd', 'g', 'b', 'a']
                
    table['teff'] = np.array(dat, dtype=object).T[1]
    table['logg'] = np.array(dat, dtype=object).T[0]
    wavls = air2vac(np.array(dat, dtype=object).T[3])
    fls = np.array(dat, dtype=object).T[2] # convert from erg cm^2 s^1 Hz^-1 ---> erg cm^2 s^1 A^-1
    
    table['wl'] = wavls
    table['fl'] = fls
    if flux_unit == 'flam':
        table['fl'] = (2.99792458e18*table['fl'] / table['wl']**2)
    
    teffs = sorted(list(set(table['teff'])))
    loggs = sorted(list(set(table['logg'])))
    
    values = np.zeros((len(teffs), len(loggs), len(wavls[0])))
    
    for i in range(len(teffs)):
        for j in range(len(loggs)):
            try:
                values[i,j] = table[np.all([table['teff'] == teffs[i], table['logg'] == loggs[j]], axis = 0)]['fl'][0]
            except:
                values[i,j] = np.zeros(len(wavls[0]))
    
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
        
    return wavls[0], model_spec, model_spec_low_logg, table