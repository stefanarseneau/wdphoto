import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import lmfit
import pyphot

from . import interpolator

#physical constants in SI units
speed_light = 299792458 #m/s
radius_sun = 6.957e8 #m
mass_sun = 1.9884e30 #kg
newton_G = 6.674e-11 #N m^2/kg^2
pc_to_m = 3.086775e16

class PhotometryEngine:
    def __init__(self, interpolator):
        self.interpolator = interpolator
        self.bands = self.interpolator.bands

        lib = pyphot.get_library()
        self.filters = [lib[band] for band in self.bands]

    def mag_to_flux(self, mag, e_mag = None):
        """
        convert from magntiudes on the AB system to flux for a particular filter (Gaia magnitudes are on the Vega system)
        
        Args:
            mag (array[N]):    magnitude of the observation
            e_mag (array[N]):  magnitude uncertainty
        Returns:
            flux in flam if e_mag is not specified, a tuple containing flux and flux uncertainty if it is
        """
        if e_mag is not None:
            flux = [10**(-0.4*(mag[i] + filter.Vega_zero_mag)) for i, filter in enumerate(self.filters)]
            e_flux = [np.abs((np.log(10)*(-0.4)*10**(-0.4 * (mag[i] + filter.Vega_zero_mag)) * e_mag[i])) for i, filter in enumerate(self.filters)]
            return flux, e_flux
        else:
            flux = [10**(-0.4*(mag[i] + filter.Vega_zero_mag)) for i, fiter in enumerate(self.filters)]
            return flux

    def get_model_flux(self, params):
        #get model photometric flux for a WD with a given radius, located a given distance away
        teff, logg, radius, distance = params['teff'], params['logg'], params['radius'], params['distance']
            
        fl= 4 * np.pi * self.interpolator(teff, logg) # flux in physical units

        #convert to SI units
        radius = radius * radius_sun # Rsun to meter
        distance = distance * pc_to_m # Parsec to meter

        return (radius / distance)**2 * fl # scale down flux by distance

    def residual(self, params, obs_flux = None, e_obs_flux = None):
        # calculate the chi2 between the model and the fit
        model_flux = self.get_model_flux(params)
        
        chisquare = ((model_flux - obs_flux) / e_obs_flux)**2
        return chisquare

    def __call__(self, obs_mag, e_obs_mag, distance, p0 = [10000, 8, 0.003]):    
        obs_flux, e_obs_flux = self.mag_to_flux(obs_mag,  e_obs_mag) # convert magnitudes to fluxes

        #use lmfit.minimize to fit the model to the data
        params = lmfit.Parameters()
        params.add('teff', value = p0[0], min = self.interpolator.teff_lims[0], max = self.interpolator.teff_lims[1], vary = True)
        params.add('logg', value = p0[1], min = self.interpolator.logg_lims[0], max = self.interpolator.logg_lims[1], vary=False)
        params.add('radius', value = p0[2], min = 0.000001, max = 0.1, vary = True)
        params.add('distance', value = distance, vary = False)
                
        result = lmfit.minimize(self.residual, params, kws = dict(obs_flux = obs_flux, e_obs_flux = e_obs_flux), method = 'leastsq')

        radius = result.params['radius'].value
        e_radius = result.params['radius'].stderr
        teff = result.params['teff'].value
        e_teff = result.params['teff'].stderr

        return radius, e_radius, teff, e_teff