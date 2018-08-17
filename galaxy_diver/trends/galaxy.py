#!/usr/bin/env python
'''Compilation of observed and derived astrophysical trends for galaxies.

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import numpy as np
import scipy.interpolate as interp

import galaxy_diver.utils.constants as constants
import galaxy_diver.utils.utilities as utilities

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
):
    '''Get the ISM mass metallicity relation, as compiled by Peeples+14
    (S3.2.2). This is really just assuming that there is a fixed ratio
    between the oxygen mass and the stellar mass.
    '''

    return ism_oxygen_mass_frac( m_star ) / 0.44 * 1.366/16. * 1e-12

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
            The FIRE simulations use a value of 0.02 .

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
            z_ism = ism_mass_metallicity_relation( m_star )

            m_z[mass_source] = m_gas * z_ism * z_sun

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

    m_star_peeples = 10.**np.arange( 6.7, 11.4 + 0.5, 0.5 )
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
