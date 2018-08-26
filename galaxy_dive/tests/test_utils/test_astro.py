'''Testing for astro.py
'''

from mock import patch
import numpy as np
import numpy.testing as npt
import pdb
import unittest

import galaxy_dive.utils.astro as astro
import galaxy_dive.utils.constants as constants

########################################################################

class TestCircularVelocity( unittest.TestCase ):

  ########################################################################

  def test_circular_velocity( self ):

    r_vir = np.array([ 268.00569698,  255.8877719 ,  239.19529116]) 
    m_vir = np.array([  1.07306268e+12,   1.04949145e+12,   1.01265385e+12])

    # What our actual circular velocity is
    result = astro.circular_velocity( r_vir, m_vir )

    # Make sure we have the right dimensions
    assert result.shape == ( 3, )

    # We expect the circular velocity of a 1e12 Msun galaxy to be roughly ~100 km/s
    expected = 100.
    actual = result[0]
    npt.assert_allclose( expected, actual, rtol=0.5 )

  ########################################################################

  def test_hubble_parameter( self ):

    expected = 75.71

    actual = astro.hubble_parameter( 0.16946, .702, 0.272, 0.728 )

    npt.assert_allclose( expected, actual, rtol=1e-4 )

  ########################################################################


  def test_hubble_parameter_1s( self ):

    expected = 75.71*3.24078e-20

    actual = astro.hubble_parameter( 0.16946, .702, 0.272, 0.728, units='1/s' )

    npt.assert_allclose( expected, actual, rtol=1e-4 )
