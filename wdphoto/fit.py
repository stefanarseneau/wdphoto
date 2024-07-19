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
    def __init__(self, interpolator, assume_mrr = False, vary_logg = False):
        self.interpolator = interpolator
        self.bands = interpolator.bands
        self.teff_lims = interpolator.teff_lims
        self.logg_lims = interpolator.logg_lims

        self.assume_mrr = assume_mrr
        self.vary_logg = vary_logg

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
            flux = [10**(-0.4*(mag[i] + filter.Vega_zero_mag)) for i, filter in enumerate(self.filters)]
            return flux

    def get_model_flux(self, params):
        #get model photometric flux for a WD with a given radius, located a given distance away        
        fl= 4 * np.pi * self.interpolator(params['teff'], params['logg']) # flux in physical units

        #convert to SI units
        if self.assume_mrr:
            radius = self.interpolator.radius_interp(params['teff'], params['logg']) * radius_sun
        else:            
            radius = params['radius'] * radius_sun # Rsun to meter
        distance = params['distance'] * pc_to_m # Parsec to meter

        return (radius / distance)**2 * fl # scale down flux by distance

    def residual(self, params, obs_flux = None, e_obs_flux = None):
        # calculate the chi2 between the model and the fit
        model_flux = self.get_model_flux(params)
        chisquare = ((model_flux - obs_flux) / e_obs_flux)**2

        return chisquare

    def __call__(self, obs_mag, e_obs_mag, distance, p0 = [], method = 'leastsq', fit_kws = None):    
        obs_flux, e_obs_flux = self.mag_to_flux(obs_mag,  e_obs_mag) # convert magnitudes to fluxes

        # if an initial guess is not specified, set it to the mean of the parameter range
        if len(p0) == 0:
            p0 = [np.average(self.teff_lims), np.average(self.logg_lims), 0.001]
        
        # initialize the parameter object
        params = lmfit.Parameters()
        params.add('teff', value = p0[0], min = self.teff_lims[0], max = self.teff_lims[1], vary = True)
        params.add('distance', value = distance, vary = False)

        if self.assume_mrr:
            # if we're assuming an MRR, we only need to vary logg
            params.add('logg', value = p0[1], min = self.logg_lims[0], max = self.logg_lims[1], vary = True)
        else:
            # if not assume_mrr, initialize a specific radius variable
            params.add('logg', value = p0[1], min = self.logg_lims[0], max = self.logg_lims[1], vary = self.vary_logg)
            params.add('radius', value = p0[2], min = 0.000001, max = 0.1, vary = True)

        # run the fit with the defined parameters
        result = lmfit.minimize(self.residual, params, kws = dict(obs_flux = obs_flux, e_obs_flux = e_obs_flux), method = method, **fit_kws = fit_kws)

        # save the variables that are the same for all versions
        teff = result.params['teff'].value
        e_teff = result.params['teff'].stderr
        logg = result.params['logg'].value
        e_logg = result.params['logg'].stderr

        if self.assume_mrr:
            radius = self.interpolator.radius_interp(teff, logg)
            e_radius = None
        else:
            radius = result.params['radius'].value
            e_radius = result.params['radius'].stderr            

        return radius, e_radius, teff, e_teff, logg, e_logg, result