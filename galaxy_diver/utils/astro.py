#!/usr/bin/env python
'''General astronomy utilities

@author: Daniel Angles-Alcazar, Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import numpy as np

import constants

########################################################################

def hubble_z( redshift, h=0.702, omega_matter=0.272, omega_lambda=0.728 ):
  '''Return Hubble factor in 1/sec for a given redshift.

  Args:
    redshift (float): The input redshift.
    h (float): The hubble parameter.
    omega_matter (float): TODO
    omega_lambda (float): TODO

  Returns:
    hubble_a (float): Hubble factor in 1/sec
  '''

  ascale = 1. / ( 1. + redshift )
  hubble_a = constants.HUBBLE * h * np.sqrt( omega_matter / ascale**3 + (1. - omega_matter - omega_lambda) / ascale**2 + omega_lambda )     # in 1/sec !!

  return hubble_a

########################################################################

def age_of_universe( redshift, h=0.71, omega_matter=0.27 ):
  '''Get the exact solution to the age of universe (for a flat universe) to a given redshift

  Args:
    redshift (float): The input redshift.
    h (float): The hubble parameter.
    omega_matter (float): TODO

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
