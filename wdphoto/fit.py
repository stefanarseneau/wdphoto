import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import lmfit
import pyphot

from . import models

#physical constants in SI units
speed_light = 299792458 #m/s
radius_sun = 6.957e8 #m
mass_sun = 1.9884e30 #kg
newton_G = 6.674e-11 #N m^2/kg^2
pc_to_m = 3.086775e16

# generate the interpolator 
base_wavl, warwick_model, warwick_model_low_logg, table = models.build_warwick_da(flux_unit = 'flam')

def mag_to_flux_spec_Vega(mag, filters, e_mag = None):
    """
    convert from magntiudes on the AB system to flux for a particular filter (Gaia magnitudes are on the Vega system)
    
    Args:
        mag (array[N]):    magnitude of the observation
        filt (array[N]):   pyphot object for the filter in question
        e_mag (array[N]):  magnitude uncertainty
    Returns:
        flux in flam if e_mag is not specified, a tuple containing flux and flux uncertainty if it is
    """
    if e_mag is not None:
        flux = [10**(-0.4*(mag[i] + filters[i].Vega_zero_mag)) for i in range(len(filters))]
        e_flux = [np.abs((np.log(10)*(-0.4)*10**(-0.4 * (mag[i] + filters[i].Vega_zero_mag)) * e_mag[i])) for i in range(len(filters))]
        return flux, e_flux
    else:
        flux = [10**(-0.4*(mag[i] + filters[i].Vega_zero_mag)) for i in range(len(filters))]
        return flux
    
def template(teff, logg, radius, distance):
    fl= 4*np.pi*warwick_model((teff,logg)) # flux in physical units
    
    #convert to SI units
    radius = radius * radius_sun # Rsun to meter
    distance = distance * pc_to_m # Parsec to meter
    
    fl = (radius / distance)**2 * fl # scale down flux by distance
    return fl

def get_model_flux(params, filters, cache = None, key = None):
    #get model photometric flux for a WD with a given radius, located a given distance away
    teff, logg, radius, distance = params['teff'], params['logg'], params['radius'], params['distance']
        
    if cache is None:
        fl= 4*np.pi*warwick_model((teff,logg))#flux in physical units
        fl = np.array([filt.get_flux(base_wavl * pyphot.unit['angstrom'], fl * pyphot.unit['erg/s/cm**2/angstrom'], axis = 1).to('erg/s/cm**2/angstrom').value for filt in filters])
    else:            
        indx = [key[filt.name] for filt in filters]
        fl = 4 * np.pi * cache((teff, logg))[indx]   
    
    #convert to SI units
    radius = radius * radius_sun # Rsun to meter
    distance = distance * pc_to_m # Parsec to meter
    
    fl = (radius / distance)**2 * fl
    return fl

def residual(params, obs_flux = None, e_obs_flux = None, filters = None, cache = None, key = None):
    #calculate the chi2 between the model and the fit
    model_flux = get_model_flux(params, filters, cache, key)
    
    chisquare = ((model_flux - obs_flux) / e_obs_flux)**2
    return chisquare

def get_parameters(obs_mag, e_obs_mag, filters, distance, cache_obj = (None, None), p0 = [10000, 8, 0.003]):    
    obs_flux, e_obs_flux = mag_to_flux_spec_Vega(obs_mag, filters, e_obs_mag) # convert magnitudes to fluxes
    cache, key = cache_obj

    #use lmfit.minimize to fit the model to the data
    params = lmfit.Parameters()
    params.add('teff', value = p0[0], min = 3001, max = 90000, vary = True)
    params.add('logg', value = p0[1], min=4.51, max=9.49, vary=False)
    params.add('radius', value = p0[2], min = 0.000001, max = 0.1, vary = True)
    params.add('distance', value = distance, min = 1, max = 2000, vary = False)
            
    result = lmfit.minimize(residual, params, kws = dict(obs_flux = obs_flux, e_obs_flux = e_obs_flux, filters = filters, cache = cache, key = key), method = 'leastsq')

    radius = result.params['radius'].value
    e_radius = result.params['radius'].stderr
    teff = result.params['teff'].value
    e_teff = result.params['teff'].stderr

    return radius, e_radius, teff, e_teff