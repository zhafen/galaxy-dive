'''Testing for particle_data.py
'''

import copy
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

    self.g_data = generic_data.GenericData( **default_kwargs )

    self.g_data.data_attrs = {
      'hubble' : 0.70199999999999996,
      'redshift' : 0.16946,
    }

  ########################################################################

  def test_retrieve_halo_data( self ):

    self.g_data.retrieve_halo_data()

    # Make sure we have the right redshift
    expected = 0.16946
    actual = self.g_data.redshift

    # Make sure we have the position right for mt halo 0, snap 500
    actual = self.g_data.halo_coords
    expected = np.array( [ 29414.96458784,  30856.75007114,  32325.90901812] )/(1. + 0.16946 )/0.70199999999999996
    npt.assert_allclose( expected, actual )

########################################################################

class TestGetData( unittest.TestCase ):

  def setUp( self ):

    self.g_data = generic_data.GenericData( **default_kwargs )

    self.g_data.data_attrs = {
      'hubble' : 0.70199999999999996,
      'redshift' : 0.16946,
    }

    # Setup some necessary data
    self.g_data.data = {
      'P' : np.random.rand( 3, 4 ),
    }

  ########################################################################

  def test_get_position_data( self ):

    rx_before = copy.copy( self.g_data.data['P'][0] )

    actual = self.g_data.get_position_data( 'Rx' )

    expected = rx_before - self.g_data.halo_coords[0]

    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_get_velocity_data( self ):
    
    assert False, "TODO"

########################################################################

class TestCenterCoords( unittest.TestCase ):

  def setUp( self ):

    self.g_data = generic_data.GenericData( **default_kwargs )

    self.g_data.data_attrs = {
      'hubble' : 0.70199999999999996,
      'redshift' : 0.16946,
    }

    # Setup some necessary data
    self.g_data.data = {
      'P' : np.random.rand( 3, 4 ),
    }

  ########################################################################

  def test_center_coords_origin_passed( self ):

    self.g_data.center_method = np.array([ 0.5, 0.5, 0.5 ])

    expected = copy.copy( self.g_data.data['P'] - self.g_data.center_method[:,np.newaxis] )

    self.g_data.center_coords()
    actual = self.g_data.data['P']

    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_center_coords_already_centered( self ):

    self.g_data.centered = True

    self.g_data.center_method = np.array([ 0.5, 0.5, 0.5 ])

    expected = copy.copy( self.g_data.data['P'] )

    self.g_data.center_coords()
    actual = self.g_data.data['P']

    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_center_coords_halo_method( self ):

    pos_before = copy.copy( self.g_data.data['P'] )

    self.g_data.center_coords()
    actual = self.g_data.data['P']

    expected = pos_before - self.g_data.halo_coords[:,np.newaxis]

    npt.assert_allclose( expected, actual )
