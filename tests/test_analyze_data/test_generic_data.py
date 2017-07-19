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

    actual = self.g_data.halo_velocity
    expected = np.array( [-48.89,  73.77,  97.25] )

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

########################################################################

class TestCenterVelCoords( unittest.TestCase ):

  def setUp( self ):

    self.g_data = generic_data.GenericData( **default_kwargs )

    self.g_data.data_attrs = {
      'hubble' : 0.70199999999999996,
      'redshift' : 0.16946,
    }

    # Setup some necessary data
    self.g_data.data = {
      'V' : np.random.rand( 3, 4 ),
    }

  ########################################################################

  def test_center_vel_coords_origin_passed( self ):

    self.g_data.vel_center_method = np.array([ 0.5, 0.5, 0.5 ])

    expected = copy.copy( self.g_data.data['V'] - self.g_data.vel_center_method[:,np.newaxis] )

    self.g_data.center_vel_coords()
    actual = self.g_data.data['V']

    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_center_vel_coords_already_centered( self ):

    self.g_data.vel_centered = True

    self.g_data.vel_center_method = np.array([ 0.5, 0.5, 0.5 ])

    expected = copy.copy( self.g_data.data['V'] )

    self.g_data.center_vel_coords()
    actual = self.g_data.data['V']

    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_center_vel_coords_halo_method( self ):

    vel_before = copy.copy( self.g_data.data['V'] )

    self.g_data.center_vel_coords()
    actual = self.g_data.data['V']

    expected = vel_before - self.g_data.halo_velocity[:,np.newaxis]

    npt.assert_allclose( expected, actual )

########################################################################

class TestProperties( unittest.TestCase ):

  def setUp( self ):

    self.g_data = generic_data.GenericData( **default_kwargs )

    self.g_data.data_attrs = {
      'hubble' : 0.70199999999999996,
      'omega_matter' : 0.272,
      'omega_lambda' : 0.728,
      'redshift' : 0.16946,
    }

    self.g_data.retrieve_halo_data()

  ########################################################################

  def test_calc_com_velocity( self ):

    self.g_data.data = {

      # Have two particles inside and one outside the region.
      'P' :  np.array( [
        [  0.,   2.,   1., ],
        [  1.,   0.,   0., ],
        [  0.,   0.,   0., ],
        [  0.,   0.,   0., ],
        ] )*self.g_data.R_vir*self.g_data.averaging_frac,

      # Have the particle outside have an insane velocity so we notice if it's affecting things.
      'V' :  np.array( [
        [  0., 200.,  10., ],
        [ 10.,   0.,   0., ],
        [  0.,   10.,   0., ],
        [  0.,   0.,   0., ],
        ] ),

      'M' : np.array( [ 1., 1., 1. ] )

    }

    actual = self.g_data.calc_com_velocity()

    expected = np.array( [ 10./4., ]*3 )

    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_hubble_z( self ):

    # Hubble parameter in km/s/kpc
    expected = 75.71*1e-3

    actual = self.g_data.hubble_z

    npt.assert_allclose( expected, actual, rtol=1e-4 )

########################################################################

class TestHubbleFlow( unittest.TestCase ):

  def setUp( self ):

    self.g_data = generic_data.GenericData( **default_kwargs )

    self.g_data.data_attrs = {
      'hubble' : 0.70199999999999996,
      'redshift' : 0.16946,
    }

    # Setup some necessary data
    self.g_data.data = {
      'V' : np.random.rand( 3, 4 ),
    }

########################################################################

