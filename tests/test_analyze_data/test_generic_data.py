'''Testing for particle_data.py
'''

from mock import patch
import numpy as np
import numpy.testing as npt
import pdb
import unittest

import galaxy_diver.analyze_data.generic_data as generic_data
import galaxy_diver.read_data.snapshot as read_snapshot

########################################################################

default_kwargs = {
  'sdir' : './tests/test_data/test_sdir',
  'analysis_dir' : './tests/test_data/test_analysis_dir',
  'snum' : 500,
  'ahf_index' : 600,
}

########################################################################

class TestGenericData( unittest.TestCase ):

  def setUp( self ):

    self.generic_data = generic_data.GenericData( **default_kwargs )

    self.generic_data.data_attrs = {
      'hubble' : 0.70199999999999996,
      'redshift' : 0.16946,
    }

  ########################################################################

  def test_retrieve_halo_data( self ):

    self.generic_data.retrieve_halo_data()

    # Make sure we have the right redshift
    expected = 0.16946
    actual = self.generic_data.redshift

    # Make sure we have the position right for mt halo 0, snap 500
    actual = self.generic_data.halo_coords
    expected = np.array( [ 29414.96458784,  30856.75007114,  32325.90901812] )/(1. + 0.16946 )/0.70199999999999996
    npt.assert_allclose( expected, actual )
