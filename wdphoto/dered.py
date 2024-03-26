from astropy.table import Table, join, unique, vstack
from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
import astropy.units as u
from dustmaps.edenhofer2023 import Edenhofer2023Query

import numpy as np
import pyphot
import extinction

def bailer_jones_distance(catalog, key = 'source_id', distance_set = 'med'):
    """
    Input:
        catalog : list of Gaia DR3 source ids
        key     : column in the catalog containing Gaia DR3 source ids
        med     : which Bailer-Jones column to access. Can be 'lo','med', or 'hi'
    Output:
        catalog : astropy table containing the source id and the Bailer-Jones distance in pc
    """
        
    stardats = []
    # Astroquery row limits mean that if the length of the table is more than 2000, split the query into multiple calls.
    iters = (len(catalog)+2000) // 2000
    
    for i in range(iters):
        if len(catalog) != 1:
            ADQL_CODE = """SELECT dist.source_id, dist.r_{}_geo
                FROM gaiadr3.gaia_source as gaia
                JOIN external.gaiaedr3_distance as dist
                ON gaia.source_id = dist.source_id      
                WHERE gaia.source_id in {}""".format(distance_set, tuple(catalog[key][2000*i:2000*i+2000]))
        else:
            ADQL_CODE = """SELECT dist.source_id, dist.r_{}_geo
                FROM gaiadr3.gaia_source as gaia
                JOIN external.gaiaedr3_distance as dist
                ON gaia.source_id = dist.source_id      
                WHERE gaia.source_id = {}""".format(distance_set, str(catalog[key][0]))
            
        stardats.append(Gaia.launch_job(ADQL_CODE,dump_to_file=False).get_results())
        
    gaia_data = vstack(stardats)
        
    return gaia_data

def edenhofer_dered(catalog, source_id_key, coord_keys, photo_keys, photo_bands, distance_set = 'med'):    
    """
    Inputs:
        catalog       : Astropy table on which to perform dereddening
        source_id_key : Name of the column containing Gaia DR3 source ids
        coord_keys    : list, (name of ra column, name of dec column)
        photo_keys    : list whose elements are the name of the photometry columns
        photo_bands   : the names of the above photometric bands in pyphot
    Returns:
        catalog       : an astropy table containing dereddened photometry, among others
    """    
    ra, dec = coord_keys[0], coord_keys[1]
    distance_col = 'r_{}_geo'.format(distance_set)
    keep_keys = list(np.concatenate(([source_id_key, ra, dec], photo_keys)))
    
    catalog = catalog[keep_keys].copy()
    distance = bailer_jones_distance(catalog, key = source_id_key, distance_set = distance_set)
    distance.rename_column('source_id', source_id_key)
    
    catalog = join(catalog, distance, keys = source_id_key)
    coords = SkyCoord(frame="icrs", ra = [catalog[ra][i] for i in range(len(catalog))]*u.deg, dec = [catalog[dec][i] for i in range(len(catalog))]*u.deg, 
                      distance = [catalog[distance_col][i] for i in range(len(catalog))] * u.pc)
    
    # Query Edenhofer2023Query to get mean E(B-V) in arbitrary units
    ehq = Edenhofer2023Query(integrated=True) #integrated=True gives integrated extinction from coordinate
    ehq_mean = ehq.query(coords, mode='mean')
    
    # Convert E to Av
    catalog['AV_'+distance_set] = 2.8*ehq_mean
    #all NaNs from objects too near (<69pc)
    #below 69 pc, just set extinction to 0 since very near
    for i in range(len(catalog)):
        if np.isnan(catalog['AV_'+distance_set][i]):
            catalog['AV_'+distance_set][i] = 0
            

            
    lib = pyphot.get_library()
    filters = [lib[photo_bands[i]] for i in range(len(photo_bands))]
    phot_wavl = np.array([x.lpivot.to('angstrom').value for x in filters])
    
    # A(V)=R*E(B-V) where the mean value of R for the diffuse interstellar medium is 3.1
    Rv = 3.1
    
    ext_all=[]
    for i in range(len(catalog)):
        av = catalog['AV_'+distance_set][i]
        #returns extinction in magnitudes at each SDSS and Gaia photometric band wavelength
        ext_all.append(extinction.fitzpatrick99(phot_wavl, av, Rv))
        
    ext_all = np.array(ext_all)
    
    for i in range(len(photo_keys)):
        catalog[photo_keys[i] + '_ext'] = ext_all.T[i]
    for i in range(len(photo_keys)):
        catalog[photo_keys[i] + '_dered'] = catalog[photo_keys[i]] - catalog[photo_keys[i] + '_ext']
        
    drop_keys = list(np.concatenate(([ra, dec], photo_keys)))
    catalog.remove_columns(drop_keys)
    
    return catalog

    