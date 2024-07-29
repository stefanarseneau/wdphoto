from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator, RegularGridInterpolator
from scipy.interpolate import griddata, interp1d
from dataclasses import dataclass
import numpy as np
import os

from astropy.table import Table, vstack
import pyphot

if __name__ != "__main__":
    from . import utils

lib = pyphot.get_library()

limits = {
    'CO_Hrich': ((4800, 79000), (8.84, 9.26)),
    'CO_Hdef': ((4500, 79000), (8.85, 9.28)),
    'ONe_Hrich': ((3750, 78800), (8.86, 9.30)),
    'ONe_Hdef': ((4250, 78800), (8.86, 9.31)),
}

class Interpolator:
    def __init__(self, interp_obj, teff_lims, logg_lims):
        self.interp_obj = interp_obj
        self.teff_lims = teff_lims
        self.logg_lims = logg_lims

class WarwickDAInterpolator:
    """
    Input:
        bands
    """
    def __init__(self, bands, precache=True):
        self.bands = bands # pyphot library objects
        self.precache = precache # use precaching?

        if not self.precache:
            self.teff_lims = (4001, 129000)
            self.logg_lims = (4.51, 9.49)

            # generate the interpolator 
            base_wavl, warwick_model, warwick_model_low_logg, table = utils.build_warwick_da(flux_unit = 'flam')
            self.interp = lambda teff, logg: np.array([lib[band].get_flux(base_wavl * pyphot.unit['angstrom'], warwick_model((teff, logg)) * pyphot.unit['erg/s/cm**2/angstrom'], axis = 1).to('erg/s/cm**2/angstrom').value for band in self.bands])
            
        else:
            self.teff_lims = (4001, 90000)
            self.logg_lims = (7, 9)

            dirpath = os.path.dirname(os.path.realpath(__file__)) # identify the current directory
            table = Table.read(f'{dirpath}/data/warwick_da/warwick_cache_table.csv') 
            self.interp = MultiBandInterpolator(table, self.bands, self.teff_lims, self.logg_lims)

    def __call__(self, teff, logg):
        return self.interp(teff, logg)

class LaPlataUltramassive:
    def __init__(self, bands, massive_params = (None, None)):        
        self.bands = bands
        self.core, self.layer = massive_params[0], massive_params[1]

        dirpath = os.path.dirname(os.path.realpath(__file__)) # identify the current directory
        model = f'{self.core}_{self.layer}'
        path = f'{dirpath}/data/laplata/{model}_Massive.csv' if (self.core is not None) else f'{dirpath}/data/laplata/allwd.csv'
        self.table = Table.read(path)
    
        self.teff_lims = limits[model][0] if (self.core is not None) else (5000, 79000)
        self.logg_lims = limits[model][1] if (self.core is not None) else (7, 9.5)
        self.interp = MultiBandInterpolator(self.table, self.bands, self.teff_lims, self.logg_lims)

        self.mass_array, self.logg_array, self.age_cool_array, self.teff_array, self.Mbol_array = self.read_cooling_tracks()

    def read_cooling_tracks(self):
        dirpath = os.path.dirname(os.path.realpath(__file__))

        # initialize the array
        mass_array  = np.zeros(0)
        logg        = np.zeros(0)
        age_cool    = np.zeros(0)
        teff        = np.zeros(0)
        Mbol        = np.zeros(0)

        if self.core == 'ONe':
            masses = ['110','116','122','129']
        else:
            masses = ['110','116','123','129']

        # load in values from each track
        for mass in masses:
            if self.core == 'CO':
                Cool = Table.read(dirpath+'/data/laplata/high_mass/'+self.core+'_'+mass+'_'+self.layer+'_0_02.dat', format='ascii') 
                Cool = Cool[::10] # (Cool['LOG(TEFF)'] > logteff_min) * (Cool['LOG(TEFF)'] < logteff_max)
                mass_array  = np.concatenate(( mass_array, np.ones(len(Cool)) * int(mass)/100 ))
                logg        = np.concatenate(( logg, Cool['logg(CGS)'] ))
                age_cool    = np.concatenate(( age_cool, Cool['tcool(gyr)']))
                teff        = np.concatenate(( teff, Cool['Teff'] ))
                Mbol        = np.concatenate(( Mbol, 4.75 - 2.5 * Cool['log(L/Lsun)'] ))
                del Cool
            else:
                Cool = Table.read(dirpath+'/data/laplata/high_mass/'+self.core+'_'+mass+'_'+self.layer+'_0_02.dat', format='ascii') 
                Cool = Cool[::10] # (Cool['LOG(TEFF)'] > logteff_min) * (Cool['LOG(TEFF)'] < logteff_max)
                mass_array  = np.concatenate(( mass_array, np.ones(len(Cool)) * int(mass)/100 ))
                logg        = np.concatenate(( logg, Cool['Log(grav)'] ))
                age_cool    = np.concatenate(( age_cool, (10**Cool['Log(edad/Myr)'] -
                                                      10**Cool['Log(edad/Myr)'][0]) * 1e6 ))
                teff     = np.concatenate(( teff, 10**Cool['LOG(TEFF)']))
                Mbol        = np.concatenate(( Mbol, 4.75 - 2.5 * Cool['LOG(L)'] ))
                del Cool

        if self.core == "ONe":
            age_cool *= 1e-3

        select = ~np.isnan(mass_array + logg + age_cool + teff + Mbol) * (age_cool > 1)
        return mass_array[select], logg[select], age_cool[select], teff[select], Mbol[select]

    def interp_xy_z(self, x, y, z, interp_type='linear'):
        selected    = ~np.isnan(x + y + z)

        if interp_type == 'linear':
            interpolator = LinearNDInterpolator
        elif interp_type == 'cubic':
            interpolator = CloughTocher2DInterpolator

        return interpolator((x[selected], y[selected]), z[selected], rescale=True)

    def __call__(self, teff, logg):
        return self.interp(teff, logg)    
    

class SingleBandInterpolator:
    def __init__(self, table, band, teff_lims, logg_lims):
        self.table = table
        self.band = band
        self.teff_lims = teff_lims
        self.logg_lims = logg_lims

        self.eval = self.build_interpolator()

    def __call__(self, teff, logg):
        return self.eval(teff, logg)

    def build_interpolator(self):
        def interpolate_2d(x, y, z, method):
            if method == 'linear':
                interpolator = LinearNDInterpolator
            elif method == 'cubic':
                interpolator = CloughTocher2DInterpolator
            return interpolator((x, y), z, rescale=True)
            #return interp2d(x, y, z, kind=method)

        def interp(x, y, z):
            grid_z      = griddata(np.array((x, y)).T, z, (grid_x, grid_y), method='linear')
            z_func      = interpolate_2d(x, y, z, 'linear')
            return z_func

        logteff_logg_grid=(self.teff_lims[0], self.teff_lims[1], 1000, self.logg_lims[0], self.logg_lims[1], 0.01)
        grid_x, grid_y = np.mgrid[logteff_logg_grid[0]:logteff_logg_grid[1]:logteff_logg_grid[2],
                                    logteff_logg_grid[3]:logteff_logg_grid[4]:logteff_logg_grid[5]]

        band_func = interp(self.table['teff'], self.table['logg'], self.table[self.band])

        photometry = lambda teff, logg: float(band_func(teff, logg))
        return photometry

class MultiBandInterpolator:
    def __init__(self, table, bands, teff_lims, logg_lims):
        self.table = table
        self.bands = bands
        self.teff_lims = teff_lims
        self.logg_lims = logg_lims

        self.interpolator = [SingleBandInterpolator(self.table, band, self.teff_lims, self.logg_lims) for band in self.bands]

    def __call__(self, teff, logg):
        return np.array([interp(teff, logg) for interp in self.interpolator])

    
