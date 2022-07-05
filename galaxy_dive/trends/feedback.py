#!/usr/bin/env python
'''FIRE feedback prescription information.

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

from tkinter import W
import numpy as np

########################################################################
########################################################################

def rate_Ia( age ):
    '''Rate of Type Ia supernova used in FIRE.
    
    Args:
        age (np.ndarray):
            Age in Myr.

    Returns:
        r_Ia (np.ndarray)
            Supernova rate at each age in units of SNe / Myr / Msun.
    '''

    result = np.full( age.shape, 0. )

    SNe_occur = age > 37.53
    result[SNe_occur] = (
        5.3e-8 +
        1.6e-5 * np.exp( - ( age[SNe_occur] - 50. )**2. / 200. )
    )

    return result

########################################################################

def rate_II( age ):
    '''Rate of Type II supernova used in FIRE.
    
    Args:
        age (np.ndarray):
            Age in Myr.

    Returns:
        r_II (np.ndarray)
            Supernova rate at each age in units of SNe / Myr / Msun.
    '''

    result = np.full( age.shape, 0. )

    first_stage = ( 3.401 < age ) & ( age < 10.37 )
    result[first_stage] = 5.408e-4

    second_stage = ( 10.37 < age ) & ( age < 37.53 )
    result[second_stage] = 2.516e-4

    return result

########################################################################

def SNe_specific_energy_rate( age, SNe_type='Ia' ):
    '''Energy deposition rate for SNe.
    
    Args:
        age (np.ndarray):
            Age in Myr.

        SNe_type (str):
            Supernova type to get energy rate for.

    Returns:
        edot_SNe (np.ndarray)
            Supernova energy rate per stellar mass, in units of Lsun/Msun.
    '''

    if SNe_type == 'Ia':
        rate_fn = rate_Ia
    elif SNe_type == 'II':
        rate_fn = rate_II
    else:
        raise KeyError( 'Unknown  SNe type key, {}'.format( SNe_type ) )
    rate = rate_fn( age )

    # 1e51 ergs in Lsun * Myr
    energy_per_SNe = 8280.13791848

    return energy_per_SNe * rate

########################################################################

def SNe_specific_energy_rate_for_N( N_SNe=1, mass=1., time_window=1., ):
    '''Calculate the energy rate for an average of N_SNe per mass per time_window.
    Can be used to determine when there are low-number-statistics number of
    SNe.

    Args:
        N_SNe (int):
            Number of supernova.

        mass (float):
            Stellar mass of source in Msun.

        time_window (float):
            Window over which this occurs, in Msun.

    Returns:
        edot_sm (float):
            Energy rate in Lsun/Msun.
    '''

    # 1e51 ergs in Lsun * Myr
    energy_per_SNe = 8280.13791848

    return energy_per_SNe * N_SNe / mass / time_window

########################################################################

def wind_kinetic_light_to_mass( age, metallicity ):
    '''Energy deposition rate for OB/AGB mass loss.
    
    Args:
        age (np.ndarray):
            Age in Myr.

        metallicity (int or np.ndarray):
            Metallicity in Zsun.

    Returns:
        LperM_wind (np.ndarray)
            Stellar wind kinetic luminosity per stellar mass, in units of Lsun/Msun.
    '''

    # Convert to array if necessary
    try:
        len( metallicity )
    except TypeError:
        metallicity = np.full( age.shape, metallicity )

    # <v_w^2> in Lsun * Myr / Msun
    windspeed_square = (
        5.94e4 / ( 1. + ( age/2.5 )**1.4 + ( age/ 10 )**5. )
        + 4.83
    ) * 0.01646436

    # Wind mass loss fraction in star mass per Gyr
    f_wind = np.full( age.shape, np.nan )
    stage_1 = age < 1.
    f_wind[stage_1] = 4.763 * ( 0.01 + metallicity[stage_1] )
    stage_2 = ( 1. < age ) & ( age < 3.5 )
    f_wind[stage_2] = (
        4.763 * ( 0.01 + metallicity[stage_2] )
        * age[stage_2]**( 1.45 + 0.8 * np.log( metallicity[stage_2] ) )
    )
    stage_3 = ( 3.5 < age ) & ( age < 100. )
    f_wind[stage_3] = 29.4 * ( age[stage_3] / 3.5 )**-3.25 + 0.0042
    stage_4 = 100. < age
    f_wind[stage_4] = (
        0.42 * ( age[stage_4] / 1e3 )**-1.1
        / ( 19.81 - np.log( age[stage_4] ) )
    )
    # Convert to star mass per Myr
    f_wind /= 1e3

    result = windspeed_square * f_wind

    return result

########################################################################

def radiation_bolometric_light_to_mass( age ):
    '''Energy deposition rate for radiation.
    Bolometric luminosity rate is given.
    
    Args:
        age (np.ndarray):
            Age in Myr.

    Returns:
        LperM_rad (np.ndarray)
            Radiation luminosity per stellar mass, in units of Lsun/Msun.
    '''

    result = np.full( age.shape, 0. )

    stage_1 = age < 3.5
    result[stage_1] = 1136.59
    stage_2 = 3.5 < age
    x = np.log10( age[stage_2] / 3.5 )
    result[np.invert(stage_1)] = 1500. * np.exp(
        -4.145 * x + 0.691 * x**2. - 0.0576 * x**3.
    )

    return result

########################################################################

def radiation_light_to_mass( age, band='bolometric' ):
    '''Energy deposition rate for radiation.
    
    Args:
        age (np.ndarray):
            Age in Myr.

        band (str):
            Which band to choose. Options include...
                'bolometric'
                'mid/far IR'
                'optical/near IR'
                'photo-electric FUV'
                'ionizing'
                'NUV'

    Returns:
        LperM_rad (np.ndarray)
            Radiation luminosity per stellar mass, in units of Lsun/Msun.
    '''

    if band == 'mid/far IR':
        return np.zeros( age.shape )

    elif band == 'optical/near IR':
        f_opt = np.full( age.shape, np.nan )
        stage_1 = age < 2.5
        f_opt[stage_1] = 0.09
        stage_2 = ( 2.5 < age ) & ( age < 6. )
        f_opt[stage_2] = 0.09 * ( 1 + ( ( age[stage_2] - 2.5 ) / 4 )**2 )
        stage_3 = 6. < age
        f_opt[stage_3] = 1 - 0.841 / ( 1 + ( age[stage_3] - 6) / 300 )
        
        LperM_bol = radiation_bolometric_light_to_mass( age )
        return f_opt * LperM_bol

    elif band == 'photo-electric FUV':
        LperM_FUV = np.full( age.shape, np.nan )
        stage_1 = age < 3.4
        LperM_FUV[stage_1] = 271. * ( 1 + ( age[stage_1] / 3.4 )**2 )
        stage_2 = 3.4 < age
        LperM_FUV[stage_2] = 572. * ( age[stage_2] / 3.4 )**-1.5

        return LperM_FUV
    
    elif band == 'ionizing':
        LperM_ion = np.zeros( age.shape )
        stage_1 = age < 3.5
        LperM_ion[stage_1] = 500.
        stage_2 = ( 3.5 < age ) & ( age < 25. )
        LperM_ion[stage_2] = (
            60 * ( age[stage_2] / 3.5 )**-3.6
            + 470 * ( age[stage_2] / 3.5 ) ** ( 0.045 - 1.82 * np.log10( age[stage_2] ) )
        )

        return LperM_ion

    elif band == 'NUV':
        LperM_NUV = radiation_bolometric_light_to_mass( age )
        for band_i in bands:
            LperM_NUV -= radiation_light_to_mass( age, band=band_i )

        return LperM_NUV

    elif band == 'bolometric':
        return radiation_bolometric_light_to_mass( age )

    else:
        raise KeyError( 'Unrecognized band, band={}'.format( band ) )

########################################################################

def feedback_specific_energy_rate(
    age,
    feedback_source = 'mechanical',
    metallicity = 1.,
    *args,
    **kwargs
):
    '''Energy deposition rate for a given feedback source.
    
    Args:
        age (np.ndarray):
            Age in Myr.
        
        feedback_source (str):
            What type of feedback. Options are...
                'all'
                'mechanical'
                'radiation'
                'heating and ionizing radiation'
                'SNe'
                'SNe Ia'
                'SNe II'
                'stellar wind'
                'bolometric radiation'
                'mid/far IR radiation'
                'optical/near IR radiation'
                'photo-electric FUV radiation'
                'ionizing radiation'
                'NUV radiation'

        metallicity (int or np.ndarray):
            Metallicity to use for metallicity-dependent feedback in Zsun.
            Currently only used by stellar wind feedback.

        *args, **kwargs:
            Passed to specific energy function for the feedback source.

    Returns:
        edot (np.ndarray)
            Supernova energy rate per stellar mass, in units of Lsun/Msun.
    '''

    radiation_bands = [
        'bolometric',
        'mid/far IR',
        'optical/near IR',
        'photo-electric FUV',
        'ionizing',
        'NUV',
    ]
    radiation_band = ' '.join( feedback_source.split( ' ' )[:-1] )

    if feedback_source == 'all':
        result = (
            feedback_specific_energy_rate( age, 'mechanical', *args, **kwargs ) +
            feedback_specific_energy_rate( age, 'radiaton', *args, **kwargs )
        )
        return result
    elif feedback_source == 'mechanical':
        result = (
            feedback_specific_energy_rate( age, 'SNe', *args, **kwargs ) +
            feedback_specific_energy_rate(
                age,
                'stellar wind',
                metallicity=metallicity,
                *args,
                **kwargs
            )
        )
        return result
    elif feedback_source == 'radiation':
        return radiation_light_to_mass( age, *args, **kwargs )
    elif feedback_source == 'heating and ionizing radiation':
        result = (
            radiation_light_to_mass( age, band='photo-electric FUV', *args, **kwargs ) +
            radiation_light_to_mass( age, band='ionizing', *args, **kwargs )
        )
        return result
    elif feedback_source == 'SNe':
        result = (
            SNe_specific_energy_rate( age, SNe_type='Ia', *args, **kwargs ) +
            SNe_specific_energy_rate( age, SNe_type='II', *args, **kwargs )
        )
        return result
    elif feedback_source == 'SNe Ia':
        return SNe_specific_energy_rate( age, SNe_type='Ia', *args, **kwargs )
    elif feedback_source == 'SNe II':
        return SNe_specific_energy_rate( age, SNe_type='II', *args, **kwargs )
    elif feedback_source == 'stellar wind':
        return wind_kinetic_light_to_mass( age, metallicity=metallicity, *args, **kwargs )
    elif radiation_band in radiation_bands:
        return radiation_light_to_mass( age, band=radiation_band, *args, **kwargs )
    else:
        raise KeyError( 'Unrecognized feedback source, feedback_source={}'.format( feedback_source ) )