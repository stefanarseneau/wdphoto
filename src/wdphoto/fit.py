import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import lmfit
import pyphot

from . import models
from . import dered

#physical constants in SI units
speed_light = 299792458 #m/s
radius_sun = 6.957e8 #m
mass_sun = 1.9884e30 #kg
newton_G = 6.674e-11 #N m^2/kg^2
pc_to_m = 3.086775e16

base_wavl, warwick_model, warwick_model_low_logg, table = models.build_warwick_da(flux_unit = 'flam')

def mag_to_flux_spec_Vega(mag, filt, e_mag = None):
    #convert from magntiudes on the Vega system to flux for a particular filter
    #Gaia magnitudes are on the Vega system
    if e_mag is not None:
        #compute the flux error given the error on the magnitude
        #assume the error on the zero point is negligible
        return (10**(-0.4*(mag + filt.Vega_zero_mag)), np.abs((np.log(10)*(-0.4)*10**(-0.4 * (mag + filt.Vega_zero_mag)) * e_mag)))
    else:
        return 10**(-0.4*(mag + filt.Vega_zero_mag))
    
def mag_to_flux_spec_AB(mag, filt, e_mag = None):
    #convert from magntiudes on the AB system to flux for a particular filter
    #SDSS magnitudes are on the AB system
    if e_mag is not None:
        #compute the flux error given the error on the magnitude
        #assume the error on the zero point is negligible
        return (10**(-0.4*(mag + filt.AB_zero_mag)), np.abs((np.log(10)*(-0.4)*10**(-0.4 * (mag + filt.AB_zero_mag)) * e_mag)))
    else:
        return 10**(-0.4*(mag + filt.AB_zero_mag))
    
def template(teff, logg, radius, distance):
    fl= 4*np.pi*warwick_model((teff,logg))#flux in physical units
    
    #convert to SI units
    radius = radius * radius_sun # Rsun to meter
    distance = distance * pc_to_m # Parsec to meter
    
    fl = (radius / distance)**2 * fl
    
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

def get_parameters(obs_flux, e_obs_flux, filters, cache, key, p0 = [10000, 8, 0.003, 100]):          
    #use lmfit.minimize to fit the model to the data
    params = lmfit.Parameters()
    params.add('teff', value = p0[0], min = 3001, max = 100000, vary = True)
    params.add('logg', value = p0[1], min=4.51, max=9.49, vary=False)
    params.add('radius', value = p0[2], min = 0.000001, max = 0.1, vary = True)
    params.add('distance', value = p0[3], min = 1, max = 2000, vary = False)
            
    result = lmfit.minimize(residual, params, kws = dict(obs_flux = obs_flux, e_obs_flux = e_obs_flux, filters = filters, cache = cache, key = key), method = 'leastsq')
        
    return result

def fit_parameters(catalog, source_id_key, coord_keys, photo_keys, e_photo_keys, photo_bands, distance_set = 'med',
                  cachefile = None, verbose_output = False, plot = False):
    if cachefile is not None:
        try:
            cache, key = models.read_cache(cachefile)
        except:
            print('Fatal Error: Cache table not found {} \nProceeding using Warwick models'.format(cachefile))
            cache = None
            key = None
    else:
        cache = None
        key = None
    
    lib = pyphot.get_library()
    filters = [lib[photo_bands[i]] for i in range(len(photo_bands))]
    
    photo = dered.edenhofer_dered(catalog, source_id_key, coord_keys, photo_keys, photo_bands, distance_set)
    distance_col = 'r_{}_geo'.format(distance_set)
    for i in range(len(e_photo_keys)):
        photo[e_photo_keys[i]] = catalog[e_photo_keys[i]]
        
    photo_keys_dered = [photo_keys[i] + '_dered' for i in range(len(photo_keys))]
    
    figs = []
    
    teff = []
    e_teff = []
    radius = []
    e_radius = []
    chi2 = []
    
    for i in tqdm(range(len(photo))):
        mags = [photo[photo_keys_dered[j]][i] for j in range(len(photo_keys_dered))]
        e_mags = [photo[e_photo_keys[j]][i] for j in range(len(photo_keys_dered))]
        
        obs_flux = [mag_to_flux_spec_Vega(mags[j], filters[j], e_mags[j])[0] for j in range(len(mags))]
        e_obs_flux = [mag_to_flux_spec_Vega(mags[j], filters[j], e_mags[j])[1] for j in range(len(mags))]
        
        try:
            result_8 = get_parameters(obs_flux, e_obs_flux, filters, cache, key, p0 = [10000, 8, 0.003, photo[distance_col][i]])
            result_7 = get_parameters(obs_flux, e_obs_flux, filters, cache, key, p0 = [10000, 8, 0.003, photo[distance_col][i]])
            result_9 = get_parameters(obs_flux, e_obs_flux, filters, cache, key, p0 = [10000, 8, 0.003, photo[distance_col][i]])
        except:
            print('Failed to fit source Gaia DR3 {}, appending -9999!'.format(photo[source_id_key][i]))
            
            teff.append(-9999)
            e_teff.append(-9999)
            radius.append(-9999)            
            e_radius.append(-9999)
            chi2.append(-9999)
            
            continue
            
        if plot:
                # plot the model spectrum, model photometric flux, and observed photometric flux
                # with the fit results
                #plt.style.use('stefan.mplstyle')
                fig, ax1 =plt.subplots(1,1,figsize=(7,7))
                ax1.errorbar([filters[j].lpivot.to('angstrom').value for j in range(len(filters))], obs_flux, yerr = e_obs_flux, 
                             linestyle = 'none', marker = 'None', color = 'k', capsize = 5, label = 'Observed SED', zorder = 10)
                
                if cache is None:
                    ax1.plot([filters[j].lpivot.to('angstrom').value for j in range(len(filters))], get_model_flux(result_8.params, filters), 'co', markersize = 10, label = 'Model SED')
                else:
                    ax1.plot([filters[j].lpivot.to('angstrom').value for j in range(len(filters))], get_model_flux(result_8.params, filters, cache, key), 'co', markersize = 10, label = 'Model SED')
                
                model_fl = template(result_8.params['teff'].value, result_8.params['logg'].value, result_8.params['radius'].value, result_8.params['distance'].value)
                mask = (3600 < base_wavl)*(base_wavl<9000)
                ax1.plot(base_wavl[mask], model_fl[mask], c = 'k', label = 'Model Spectrum')
                
                ax1.text(0.5, 0.70, "$T_{eff}$" + " = {:2.6} $K$".format(result_8.params['teff'].value), transform = plt.gca().transAxes, fontsize = 18)
                ax1.text(0.5, 0.65, "$\log g$" + " = {:2.5f}".format(result_8.params['logg'].value), transform = plt.gca().transAxes, fontsize = 18)
                ax1.text(0.5, 0.60, r'$Radius = ${:2.5f} $R_\odot$'.format(result_8.params['radius'].value), transform = plt.gca().transAxes, fontsize = 18)
                ax1.text(0.5, 0.55, '$Distance = ${:2.1f} $pc$'.format(result_8.params['distance'].value), transform = plt.gca().transAxes, fontsize=18)
                ax1.text(0.5, 0.50, r'$\chi_r^2 = ${:2.2f}'.format(result_8.redchi), transform = plt.gca().transAxes, fontsize = 18)
                
                ax1.set_xlim((3450, 9000))
                ax1.set_xlabel(r'Wavelength $[\AA]$')
                ax1.set_ylabel('Flux $[erg/s/cm^2/\AA]$')
                ax1.set_title('Gaia DR3 {}'.format(photo[source_id_key][i]))
                ax1.legend()
                
                figs.append(fig)
                plt.close()
        
        teff.append(result_8.params['teff'].value)
        e_teff.append(result_8.params['teff'].stderr)
        radius.append(result_8.params['radius'].value)            
        e_radius.append(np.sqrt(result_8.params['radius'].stderr**2 + np.abs(result_7.params['radius'].value - result_9.params['radius'].value)**2 ))
        chi2.append(result_8.redchi)

    
    photo['radius'] = radius
    photo['e_radius'] = e_radius
    photo['teff'] = teff
    photo['e_teff'] = e_teff
    photo['redchi'] = chi2
    
    
    if not verbose_output:
        drop = []
        keys = photo.keys()
        
        for key in keys:
            if key not in [source_id_key, 'radius', 'e_radius', 'teff', 'e_teff', 'redchi']:
                drop.append(key)
                
        photo.remove_columns(drop)
    
    if not plot:
        return photo
    else:
        return photo, figs