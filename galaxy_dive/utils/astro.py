#!/usr/bin/env python
'''General astronomy utilities

@author: Daniel Angles-Alcazar, Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import numpy as np

import galaxy_dive.utils.constants as constants

########################################################################

def hubble_parameter( redshift, h=0.702, omega_matter=0.272, omega_lambda=0.728, units='km/s/Mpc' ):
  '''Return Hubble factor in 1/sec for a given redshift.

  Args:
    redshift (float): The input redshift.
    h (float): The hubble parameter.
    omega_matter (float): Cosmological mass fraction of matter.
    omega_lambda (float): Cosmological mass fraction of dark energy.
    units (str): The units the hubble parameter should be returned in.

  Returns:
    hubble_a (float): Hubble factor in specified units
  '''

  zp1 = (1. + redshift)

  e_z = np.sqrt( omega_matter*zp1**3. + omega_lambda )

  hubble_z = h*100.*e_z

  if units == 'km/s/Mpc':
    pass
  elif units == 'km/s/kpc':
    hubble_z /= 1e3
  elif units == '1/s':
    hubble_z /= constants.KM_PER_KPC*1e3
  else:
    raise KeyError( "Unspecified units, {}".format( units ) )
    

  return hubble_z

########################################################################

def age_of_universe( redshift, h=0.71, omega_matter=0.27 ):
  '''Get the exact solution to the age of universe (for a flat universe) to a given redshift

  Args:
    redshift (float): The input redshift.
    h (float): The hubble parameter.
    omega_matter (float): Cosmological mass fraction of matter.

  Returns:
    t (float): Age of the universe in Gyr
  '''

  a = 1./(1.+redshift)
  x = omega_matter / (1. - omega_matter) / (a*a*a)

  t = (2./(3.*np.sqrt(1. - omega_matter))) * np.log( np.sqrt(x) / (-1. + np.sqrt(1.+x)) )

  t *= 13.777 * (0.71/h) ## in Gyr

  return t

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

def get_sneii_metal_budget( m_star, y_z_ii=0.030 ):
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
