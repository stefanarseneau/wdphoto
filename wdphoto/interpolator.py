from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator, RegularGridInterpolator
from scipy.interpolate import griddata, interp1d
import argparse
import numpy as np
import os

from astropy.table import Table, vstack
import pyphot

if __name__ != "__main__":
    from . import utils

lib = pyphot.get_library()

class Interpolator:
    def __init__(self, interp_obj, teff_lims, logg_lims):
        self.interp_obj = interp_obj
        self.teff_lims = teff_lims
        self.logg_lims = logg_lims

class WarwickDAInterpolator:
    def __init__(self, bands, precache=True):
        self.bands = bands # pyphot library objects
        self.precache = precache # use precaching?

        if not self.precache:
            # generate the interpolator 
            base_wavl, warwick_model, warwick_model_low_logg, table = utils.build_warwick_da(flux_unit = 'flam')
            self.interp = lambda teff, logg: np.array([band.get_flux(base_wavl * pyphot.unit['angstrom'], 4*np.pi*warwick_model((teff, logg)) * pyphot.unit['erg/s/cm**2/angstrom'], axis = 1).to('erg/s/cm**2/angstrom').value for band in self.bands])
            
            self.key = {}
            for i, band in enumerate(self.bands):
                self.key[band.name] = i
        else:
            dirpath = os.path.dirname(os.path.realpath(__file__)) # identify the current directory
            table = Table.read(f'{dirpath}/data/warwick_da/warwick_cache_table.csv') 
            self.interp = MultiBandInterpolator(table, self.bands)
            self.key = {band:i for i, band in enumerate(self.bands)}

        self.teff_lims = (3001, 90000)
        self.logg_lims = (4.51, 9.49)

    def __call__(self, teff, logg):
        return self.interp(teff, logg)


class LaPlataInterpolator:
    def __init__(self, core, layer, z, bands):
        self.masses = ['110', '116', '123', '129'] # solar masses (e.g. 1.10 Msun)
        
        self.core = core
        self.layer = layer
        self.z = z
        self.bands = bands

        self.table = self.build_table_over_mass()

        self.interp = MultiBandInterpolator(self.table, self.bands)
        self.key = {band:i for i, band in enumerate(self.bands)}
        self.teff_lims = (4000, 73000)
        self.logg_lims = (8.85, 9.35)

    def build_table_over_mass(self):
        table = Table()
        dirpath = os.path.dirname(os.path.realpath(__file__)) # identify the current directory

        for mass in self.masses:
            # read in the table from the correct file
            path = f'{dirpath}/data/laplata/{self.core}_{mass}_{self.layer}_{self.z}.dat'
            temp_table = Table.read(path, format='ascii')

            # put the columns into a pyphot-readable standard
            temp_table.rename_columns(['Teff', 'logg(CGS)'], ['teff', 'logg'])
            temp_table.remove_columns(['g_1', 'r_1', 'i_1', 'z_1', 'y', 'U', 'B',\
                                'V', 'R', 'I', 'J', 'H', 'K', 'FUV', 'NUV'])
            
            mag_cols = ['Gaia_G_mag', 'Gaia_BP_mag', 'Gaia_RP_mag', 'SDSS_u_mag', 'SDSS_g_mag', 'SDSS_r_mag', 'SDSS_i_mag', 'SDSS_z_mag']
            temp_table.rename_columns(['G3', 'Bp3', 'Rp3', 'u', 'g', 'r', 'i', 'z' ], mag_cols)
            
            # now convert from absolute magnitude to surface flux
            for col in mag_cols:
                temp_table[col[:-4]] = (4*np.pi)**-1 * 10**(-0.4*(temp_table[col] + lib[col[:-4]].Vega_zero_mag)) * ((10*3.086775e16) / (6.957e8*temp_table['R/R_sun']))**2
            
            # stack the tables
            table = vstack([table, temp_table])
        return table
    
    def __call__(self, teff, logg):
        return self.interp(teff, logg)
    
class SingleBandInterpolator:
    def __init__(self, table, band):
        self.table = table
        self.band = band
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

        logteff_logg_grid=(3500, 80000, 1000, 8.77, 9.3, 0.01)
        grid_x, grid_y = np.mgrid[logteff_logg_grid[0]:logteff_logg_grid[1]:logteff_logg_grid[2],
                                    logteff_logg_grid[3]:logteff_logg_grid[4]:logteff_logg_grid[5]]

        mask = self.table['teff'] > logteff_logg_grid[0]
        table = self.table[mask]

        band_func = interp(self.table['teff'], self.table['logg'], self.table[self.band])

        photometry = lambda teff, logg: float(band_func(teff, logg))
        return photometry

class MultiBandInterpolator:
    def __init__(self, table, bands):
        self.table = table
        self.bands = bands
        self.interpolator = [SingleBandInterpolator(self.table, band) for band in self.bands]

    def __call__(self, teff, logg):
        return np.array([interp(teff, logg) for interp in self.interpolator])

class LaPlataTests:
    def __init__(self, basepath = './data/laplata'):
        self.basepath = basepath
        self.cores = ['CO', 'ONe'] # La Plata core models
        self.Hlayers = ['Hrich', 'Hdef'] # available H layer thicknesses
        self.zs = ['0_001', '0_02', '0_04', '0_06'] # metallicities

        self.teffs = np.arange(4000, 73000, 500)
        self.loggs = np.arange(8.85, 9.35, 0.01)

        self.reset_grid()

    def reset_grid(self):
        self.grid = np.zeros((len(self.teffs), len(self.loggs), 3))

    def test_metallicity(self):
        """
        Holding all other variables constant, understand the effect of different metallicities
        """

        print('TEST METALLICITY VARIATION')

        # ONe have no z variation, so only test CO
        core = self.cores[0]

        for layer in self.Hlayers:
            print(f'CORE: {core} | H-LAYER: {layer} ========================')
            grids = {}
            
            for z in self.zs:
                interp = LaPlataInterpolator(core, layer, z, bands = ['Gaia_G', 'Gaia_BP', 'Gaia_RP'])

                for i in range(len(self.teffs)):
                    for j in range(len(self.loggs)):
                        self.grid[i,j] = interp(self.teffs[i], self.loggs[j])

                grids[z] = (interp.interp, self.grid)
                self.reset_grid()
            self.print_test(grids)

    def test_hlayers(self):
        """
        Holding all other variables constant, understand the effect of different H layers
        """

        print('TEST HLAYER VARIATION')

        for core in self.cores:
            for z in self.zs:
                if (core == 'ONe') and (z != '0_02'):
                    continue

                print(f'CORE: {core} | Z: {z} ========================')
                grids = {}
                
                for layer in self.Hlayers:
                    interp = LaPlataInterpolator(core, layer, z, bands = ['Gaia_G', 'Gaia_BP', 'Gaia_RP'])

                    for i in range(len(self.teffs)):
                        for j in range(len(self.loggs)):
                            self.grid[i,j] = interp(self.teffs[i], self.loggs[j])

                    grids[layer] = (interp.interp, self.grid)
                    self.reset_grid()
                self.print_test(grids)


    def test_cores(self):
        """
        Holding all other variables constant, understand the effect of different H layers
        """

        print('TEST CORE VARIATION')

        z = self.zs[1]

        for layer in self.Hlayers:

            print(f'HLAYER: {layer} | Z: {z} ========================')
            grids = {}
            
            for core in self.cores:
                interp = LaPlataInterpolator(core, layer, z, bands = ['Gaia_G', 'Gaia_BP', 'Gaia_RP'])

                for i in range(len(self.teffs)):
                    for j in range(len(self.loggs)):
                        self.grid[i,j] = interp(self.teffs[i], self.loggs[j])

                grids[core] = (interp.interp, self.grid)
                self.reset_grid()
            self.print_test(grids)

    def print_test(self, grids):
        for key1 in grids.keys():
            for key2 in grids.keys():
                if key1 != key2:
                    print(f'    {key1} and {key2}:')
                    mean = np.nanmean(np.abs(grids[key1][1] - grids[key2][1]) / np.nanmean(grids[key1][1]))*100
                    median = np.nanmedian(np.abs(grids[key1][1] - grids[key2][1]) / np.nanmedian(grids[key1][1]))*100
                    print(f'    mean absolute difference: {mean:2.2f}%')
                    print(f'    median absolute difference: {median:2.2f}%\n')

    def run_tests(self, metallicity, layers, core_comps):
        if metallicity:
            self.test_metallicity()
        if layers:
            self.test_hlayers()
        if core_comps:
            self.test_cores()
            
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-metallicities", action="store_true")
    parser.add_argument("--test-hlayers", action="store_true")
    parser.add_argument("--test-cores", action="store_true")
    parser.add_argument("path")

    args = parser.parse_args()

    tester = LaPlataTests(args.path)
    tester.run_tests(args.test_metallicities, args.test_hlayers, args.test_cores)