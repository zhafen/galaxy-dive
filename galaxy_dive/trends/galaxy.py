#!/usr/bin/env python
'''Compilation of observed and derived astrophysical trends for galaxies.

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import math
import numpy as np
import os
import pandas as pd
import scipy.interpolate as interp
from scipy.optimize import newton
import sys

import galaxy_dive.config as gd_config
import galaxy_dive.utils.constants as constants
import galaxy_dive.utils.utilities as utilities

########################################################################

def circular_velocity( r_vir, m_vir ):
  '''Calculate the circular velocity of a halo in km/s.

  Args:
    r_vir (float or array-like) : The virial radius in pkpc.
    m_vir (float or array-like) : The halo mass in Msun.

  Returns:
    v_c : Circular velocity of the halo in km/s, indexed the same way that ahf_reader.mtree_halos[i] is.
  '''
  
  # Convert r_vir and m_vir to cgs
  r_vir_cgs = r_vir*constants.CM_PER_KPC
  m_vir_cgs = m_vir*constants.MSUN

  # Get v_c
  v_c_cgs = np.sqrt( constants.G_UNIV * m_vir_cgs / r_vir_cgs )

  # Convert to km/s
  v_c = v_c_cgs / constants.CM_PER_KM

  return v_c

########################################################################
# Metal Trends
########################################################################

def stellar_mass_metallicity_relation( m_star, apply_correction=True ):
    '''Mass-metallicity relation (MZR) in stars at z=0, taken from
        Gallazzi+2005 with a correction to the B-band weighted stellar
        metallicities by Peeples&Somerville2013
        (and put together in S3.1 of Peeples+14).

    Args:
        m_star (float or array-like) :
            Mass of the target galaxy at z=0 in units of Msun.

        apply_correction (bool) :
            If True, apply the correction from B-band weighted stellar
            metallicities to mass-weighted stellar metallicities calculated
            by Peeples&Somerville2013
    
    Returns:
        z (dict of np.ndarrays): Mass metallicity relation percentiles
    '''

    # Values from Table 2 of Gallazzi+2005
    m_star_gallazzi = 10.**np.arange( 8.91, 11.91 + .2, .2 )
    z_data = utilities.SmartDict({
        50 : 10.**np.array([
            -0.60,
            -0.61,
            -0.65,
            -0.61,
            -0.52,
            -0.41,
            -0.23,
            -0.11,
            -0.01,
            0.04,
            0.07,
            0.10,
            0.12,
            0.13,
            0.14,
            0.15,
        ]),
        16 : 10.**np.array([
            -1.11,
            -1.07,
            -1.10,
            -1.03,
            -0.97,
            -0.90,
            -0.80,
            -0.65,
            -0.41,
            -0.24,
            -0.14,
            -0.09,
            -0.06,
            -0.04,
            -0.03,
            -0.03,
        ]),
        84 : 10.**np.array([
            0.,
            0.,
            -0.05,
            -0.01,
            0.05,
            0.09,
            0.14,
            0.17,
            0.20,
            0.22,
            0.24,
            0.25,
            0.26,
            0.28,
            0.29,
            0.30,
        ]),
    })

    # Interpolate
    z = utilities.SmartDict( {} )
    for key, item in z_data.items():
        z[key] = interp.interp1d( m_star_gallazzi, item )( m_star )

    # Apply correction
    if apply_correction:
        for key, item in z.items():
            z[key] = 10.**( 1.08 * np.log10( item ) - 0.16 )

    return z

########################################################################

def ism_oxygen_mass_frac(
    m_star,
):
    '''Observations of Oxygen mass in the ISM compiled by Peeples+14.
    Data is from Kewley&Ellison2008.
    Args:
        m_star (float or array-like) :
            Mass of the target galaxy at z=0 in units of Msun.

    Returns:
        m_o (np.ndarray): log(O/H) + 12
    '''

    log_m_star = np.log10( m_star )
    log_moxy_mg = (
        27.76
        - 7.056 * log_m_star
        + 0.8184 * log_m_star**2.
        - 0.03029 * log_m_star**3.
    )

    return 10.**log_moxy_mg

########################################################################

def ism_mass_metallicity_relation(
    m_star,
    z_sun = 0.02,
):
    '''Get the ISM mass metallicity relation, as compiled by Peeples+14
    (S3.2.2). This is really just assuming that there is a fixed ratio
    between the oxygen mass and the stellar mass.

    Returns: ISM mass metallicity relation, in *mass fraction*
    '''

    return ism_oxygen_mass_frac( m_star ) / 0.44 * 1.366/16. * 1e-12 / z_sun

########################################################################

def galaxy_metal_mass(
    m_star,
    z_sun = 0.02,
    apply_stellar_z_correction = True,
    cold_gas_correction_factor = None,
    use_powerlaw_median_for_cold_gas = False,
    mass_sources = [ 'stars', 'ISM', 'dust' ],
):
    '''Mass of metals in galaxies, as assembled by Peeples+2014.

    Args:
        m_star (float or array-like) :
            Mass of the target galaxy at z=0 in units of Msun.

        z_sun (float) :
            Solar metallicity. Some (e.g. Peeples+14) use the Caffau+2011
            value of 0.0153, while others use the Asplund+2010 value of 0.014.
        apply_stellar_z_correction (bool) :
            If True, apply the correction from B-band weighted stellar
            metallicities to mass-weighted stellar metallicities calculated
            by Peeples&Somerville2013

        cold_gas_correction

    Returns:
        m_z (dict of dicts) :
            Means and percentages for each category, and the total.
    '''

    # Setup base dictionary
    m_z = {}

    for mass_source in mass_sources:

        # Stellar metal masses
        if mass_source == 'stars':

            # Get stellar metal mass out
            z_star = stellar_mass_metallicity_relation(
                m_star,
                apply_correction = apply_stellar_z_correction,
            )
            m_z[mass_source] = z_star * m_star * z_sun

        # ISM gas metal masses
        if mass_source == 'ISM':

            # Get gas mass out.
            m_gas = cold_gas_fraction(
                m_star,
                use_powerlaw_median = use_powerlaw_median_for_cold_gas,
            ) * m_star

            # Correct for only accounting for cold gas
            if cold_gas_correction_factor is not None:
                m_gas *= cold_gas_correction_factor

            # Get metallicity
            z_ism = ism_mass_metallicity_relation( m_star, z_sun )

            m_z[mass_source] = m_gas * z_ism

        # Dust mass
        if mass_source == 'dust':

            m_dust = ism_dust_mass( m_star )

            m_z[mass_source] = utilities.SmartDict( {
                16 : m_dust / 1.5,
                50 : m_dust,
                84 : m_dust * 1.5,
            } )

    # Total
    # Create the array
    m_z['total'] = utilities.SmartDict( {} )
    for key, item in m_z[mass_sources[0]].items():
        m_z['total'][key] = np.zeros( m_star.shape )
    # Sum up
    for mass_source in mass_sources:
        for key, item in m_z[mass_source].items():
            m_z['total'][key] += item

    return m_z

########################################################################

def galaxy_available_metals(
    m_star,
    alpha = 'Peeples14',
    y = 'Peeples14',
):
    '''Available metals in a galaxy, assuming a power law of the form
    y * m_star ** alpha . y is basically the yield, and alpha should usually
    be close to 1.

    Args:
        m_star (float or array-like) :
            Mass of the target galaxy at z=0 in units of Msun.

        alpha, y (float or str) :
            'Peeples14' uses a fit to digitized data from Peeples+14.

    Returns:
        galaxy_available_metals (same as m_star): Results
    '''

    if alpha == 'Peeples14':
        alpha = 1.014
    if y == 'Peeples14':
        y = 0.0479

    return y * m_star ** alpha

########################################################################

def halo_metal_budget(
    data = 'MzMstar_Ma2016.csv',
    data_dir = gd_config.DATA_DIR,
    log_mstar = False,
):
    '''Plot the halo metal budget, i.e. the mass of metals in the halo divided
    by the estimate of the mass of available metals produced by the galaxy.

    Args:
        data (str) :
            Filename of the data to use.
            'MsMstar_Ma2016.csv' contains values from the FIRE-1 sims.

        data_dir (str) :
            Location of the data file.

        log_mstar (boolean) :
            If True, the data .csv file has Mstar in log-space.

    Retuns:
        m_star (array-like) :
            Stellar mass values

        met_budget (array-like) :
            Budget for a given stellar mass
    '''

    # Load the data
    data_filepath = os.path.join( data_dir, data )
    df = pd.read_csv( data_filepath )

    if log_mstar:
        df['Mstar'] = 10.**df['Mstar']

    # Return the data
    return df['Mstar'].values, df['MetalBudget'].values

########################################################################

def cold_gas_fraction(
    m_star = None,
    use_powerlaw_median = True,
):
    '''Mass of cold gas in galaxies, as given by a compilation of data by
    Peeples+2014, consisting of data from McGaugh2005,2012, Leroy+2008,
    and Saintonge+2011.

    Args:
        m_star (float or array-like) :
            Mass of the target galaxy at z=0 in units of Msun.

    Returns:
        f_g (dict of np.ndarrays) :
            Cold gas fraction.
    '''

    m_star_peeples = 10.**np.array([
        6.7,
        7.1,
        7.6,
        8.2,
        8.6,
        9.1,
        9.6,
        10.1,
        10.6,
        11.0,
        11.4,
    ])
    f_g_data = {
        50 : np.array([
            7.4,
            3.7,
            5.6,
            5.8,
            3.6,
            1.3,
            0.6,
            0.5,
            0.23,
            0.17,
            0.067,
        ]),
        16 : np.array([
            4.6,
            2.1,
            2.2,
            4.8,
            2.6,
            1.0,
            0.32,
            0.083,
            0.056,
            0.082,
            0.057,
        ]),
        84 : np.array([
            12.0,
            19.5,
            15.7,
            8.4,
            4.3,
            2.2,
            1.7,
            1.2,
            0.52,
            0.34,
            0.12,
        ]),
    }

    # Interpolate
    if m_star is not None:
        f_g = utilities.SmartDict( {} )
        for key, item in f_g_data.items():
            f_g[key] = interp.interp1d( m_star_peeples, item )( m_star )
    else:
        f_g = utilities.SmartDict( f_g_data )

        return m_star_peeples, f_g

    # Replace the median with a fit powerlaw, if requested
    if use_powerlaw_median:
        f_g[50] = 10.**( -0.48 * np.log10( m_star ) + 4.39 )

    return f_g
        
########################################################################

def ism_dust_mass( m_star ):
    '''ISM dust mass, compiled by Peeples+2014, using data from Skibba+2011 and
    Draine+2007.

    Args:
        m_star (float or array-like) :
            Mass of the target galaxy at z=0 in units of Msun.

    Returns:
        ism_dust_mass (same as m_star):
            Units of Msun.
    '''

    return 10.**( 0.86 * np.log10( m_star ) - 1.31 )

########################################################################

def sneii_metal_budget( m_star, y_z_ii=0.030 ):
    '''Get the total mass of metal produced by SNe by z=0 in a galaxy with a
    z=0 stellar mass of m_star.
    This is taken from Peeple+2014, who use the star formation histories of
    Leitner+2012.

    Args:
        m_star (float or array-like) :
            Mass of the target galaxy at z=0 in units of Msun.

        y_z_ii (float) :
            Nucleosynthetic yield of all heavy 

    Returns:
        sneii_metal_budget (same as m_star):
            Units of Msun.
    '''

    return 1.2856 * y_z_ii * m_star ** 1.0146

########################################################################

def smhm_m_halo(log_m_star, z, paramfile=None ):
    '''Adapted from Behroozi+2018 (UniverseMachine EDR) by S.Wellons 7/11/18

    Args:
        log_m_star (float or array-like):
            log10 of the stellar mass in units of Msun.

        z (float) :
            Redshift.

        paramfile (str):
            What data to use.

    Returns:
        log_m_halo (same as log_m_star) :
            log10 of the halo mass in Msun
    '''

    if paramfile is None:
        dirname = os.path.dirname( __file__ )
        paramfile = os.path.join(
            dirname,
            "params",
            "smhm_true_med_params.txt"
        )

    log_m_halo = np.zeros(len(log_m_star))
    for i in range(0,len(log_m_star)):
        log_m_halo[i] = newton(_smhm_m_star, 12., args=(z, log_m_star[i]))
    return log_m_halo

def _smhm_m_star(log_m_halo, z, target):

    dirname = os.path.dirname( __file__ )
    paramfile = os.path.join(
        dirname,
        "params",
        "smhm_true_med_params.txt"
    )

    return smhm_m_star(np.asarray([log_m_halo]), z, paramfile=paramfile)[0] - target
            
########################################################################

def smhm_m_star(log_m_halo, z, paramfile=None):
    '''Adapted from Behroozi+2018 (UniverseMachine EDR) by S.Wellons 7/11/18

    Args:
        log_m_halo (same as log_m_star) :
            log10 of the halo mass in Msun

        z (float) :
            Redshift.

        paramfile (str):
            What data to use.

    Returns:
        log_m_star (float or array-like):
            log10 of the stellar mass in units of Msun.
    '''

    if paramfile is None:
        dirname = os.path.dirname( __file__ )
        paramfile = os.path.join(
            dirname,
            "params",
            "smhm_true_med_params.txt"
        )

    #Load params
    param_file = open(paramfile, "r") 
    param_list = []
    allparams = []
    for line in param_file:
        param_list.append(float((line.split(" "))[1]))
        allparams.append(line.split(" "))

    if (len(param_list) != 20):
        print("Parameter file not correct length.  (Expected 20 lines, got %d)." % len(param_list))
        quit()

    names = "EFF_0 EFF_0_A EFF_0_A2 EFF_0_Z M_1 M_1_A M_1_A2 M_1_Z ALPHA ALPHA_A ALPHA_A2 ALPHA_Z BETA BETA_A BETA_Z DELTA GAMMA GAMMA_A GAMMA_Z CHI2".split(" ");
    params = dict(zip(names, param_list))

    #Decide whether to print tex or evaluate SMHM parameter
    try:
        z = float(z)
    except:
        #print TeX
        for x in allparams[0:8:1]:
            sys.stdout.write('& $%.3f^{+%.3f}_{-%.3f}$' % tuple(float(y) for y in x[1:4]))
        sys.stdout.write("\\\\\n & & & ")
        for x in allparams[8:16:1]:
            sys.stdout.write('& $%.3f^{+%.3f}_{-%.3f}$' % tuple(float(y) for y in x[1:4]))
        sys.stdout.write("\\\\\n & & & ")    
        for x in allparams[16:19:1]:
            sys.stdout.write('& $%.3f^{+%.3f}_{-%.3f}$' % tuple(float(y) for y in x[1:4]))
        sys.stdout.write(' & %.0f' % float(allparams[19][1]))
        if (float(allparams[19][1])>200):
            sys.stdout.write('$\dag$')
        print('\\\\[2ex]')
        quit()

    #Print SMHM relation
    a = 1.0/(1.0+z)
    a1 = a - 1.0
    lna = math.log(a)
    zparams = {}
    zparams['m_1'] = params['M_1'] + a1*params['M_1_A'] - lna*params['M_1_A2'] + z*params['M_1_Z']
    zparams['sm_0'] = zparams['m_1'] + params['EFF_0'] + a1*params['EFF_0_A'] - lna*params['EFF_0_A2'] + z*params['EFF_0_Z']
    zparams['alpha'] = params['ALPHA'] + a1*params['ALPHA_A'] - lna*params['ALPHA_A2'] + z*params['ALPHA_Z']
    zparams['beta'] = params['BETA'] + a1*params['BETA_A'] + z*params['BETA_Z']
    zparams['delta'] = params['DELTA']
    zparams['gamma'] = 10**(params['GAMMA'] + a1*params['GAMMA_A'] + z*params['GAMMA_Z'])

    smhm_max = 14.5-0.35*z
    #print('#Log10(Mpeak/Msun) Log10(Median_SM/Msun) Log10(Median_SM/Mpeak)')
    #print('#Mpeak: peak historical halo mass, using Bryan & Norman virial overdensity.')
    #print('#Overall fit chi^2: %f' % params['CHI2'])
    # if (params['CHI2']>200):
    #     print('#Warning: chi^2 > 200 implies that not all features are well fit.  Comparison with the raw data (in data/smhm/median_raw/) is crucial.')
    log_m_star = log_m_halo*0.
    for i in range(len(log_m_halo)): #m in [x*0.05 for x in range(int(10.5*20),int(smhm_max*20+1),1)]:
        m = log_m_halo[i]
        dm = m-zparams['m_1'];
        dm2 = dm/zparams['delta'];
        sm = zparams['sm_0'] - math.log10(10**(-zparams['alpha']*dm) + 10**(-zparams['beta']*dm)) + zparams['gamma']*math.exp(-0.5*(dm2*dm2));
        log_m_star[i] = sm
        #print("%.2f %.6f %.6f" % (m,sm,sm-m))

    return log_m_star

########################################################################

def smhm_slope(log_m_halo, z, paramfile=None):
    '''Adapted from Behroozi+2018 (UniverseMachine EDR) by S.Wellons 7/11/18

    Args:
        log_m_halo (same as log_m_star) :
            log10 of the halo mass in Msun

        z (float) :
            Redshift.

        paramfile (str):
            What data to use.

    Returns:
        slope (float or array-like):
            Returns dlogMstar/dlogMhalo
    '''

    if paramfile is None:
        dirname = os.path.dirname( __file__ )
        paramfile = os.path.join(
            dirname,
            "params",
            "smhm_true_med_params.txt"
        )

    #Load params
    param_file = open(paramfile, "r") 
    param_list = []
    allparams = []
    for line in param_file:
        param_list.append(float((line.split(" "))[1]))
        allparams.append(line.split(" "))

    if (len(param_list) != 20):
        print("Parameter file not correct length.  (Expected 20 lines, got %d)." % len(param_list))
        quit()

    names = "EFF_0 EFF_0_A EFF_0_A2 EFF_0_Z M_1 M_1_A M_1_A2 M_1_Z ALPHA ALPHA_A ALPHA_A2 ALPHA_Z BETA BETA_A BETA_Z DELTA GAMMA GAMMA_A GAMMA_Z CHI2".split(" ");
    params = dict(zip(names, param_list))

    z = float(z)

    #Print SMHM relation
    a = 1.0/(1.0+z)
    a1 = a - 1.0
    lna = math.log(a)
    zparams = {}
    zparams['m_1'] = params['M_1'] + a1*params['M_1_A'] - lna*params['M_1_A2'] + z*params['M_1_Z']
    zparams['sm_0'] = zparams['m_1'] + params['EFF_0'] + a1*params['EFF_0_A'] - lna*params['EFF_0_A2'] + z*params['EFF_0_Z']
    zparams['alpha'] = params['ALPHA'] + a1*params['ALPHA_A'] - lna*params['ALPHA_A2'] + z*params['ALPHA_Z']
    zparams['beta'] = params['BETA'] + a1*params['BETA_A'] + z*params['BETA_Z']
    zparams['delta'] = params['DELTA']
    zparams['gamma'] = 10**(params['GAMMA'] + a1*params['GAMMA_A'] + z*params['GAMMA_Z'])

    smhm_max = 14.5-0.35*z
    slope = log_m_halo*0.
    for i in range(len(log_m_halo)): #m in [x*0.05 for x in range(int(10.5*20),int(smhm_max*20+1),1)]:
        m = log_m_halo[i]
        dm = m-zparams['m_1'];
        
        term1 = (zparams['alpha']*10.**(zparams['beta']*dm)+zparams['beta']*10.**(zparams['alpha']*dm))/(10.**(zparams['beta']*dm) + 10.**(zparams['alpha']*dm))
        term2 = -zparams['gamma']*dm*math.exp(-(dm/zparams['delta'])**2/2.)/zparams['delta']**2
        slope[i] = term1 + term2

    return slope
