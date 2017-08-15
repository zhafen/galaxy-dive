'''Testing for particle_data.py
'''

import copy
from mock import patch, sentinel
import numpy as np
import numpy.testing as npt
import pdb
import unittest

import galaxy_diver.analyze_data.generic_data as generic_data
import galaxy_diver.read_data.snapshot as read_snapshot

########################################################################

default_kwargs = {
  'data_dir' : './tests/data/sdir',
  'ahf_data_dir' : './tests/data/analysis_dir',
  'snum' : 500,
  'ahf_index' : 600,
}

########################################################################

class TestDataMasker( unittest.TestCase ):

  def setUp( self ):

    g_data = generic_data.GenericData( **default_kwargs )

    # Make it so we don't have to load in the halo data
    g_data.r_scale = 1.
    g_data.centered = True

    g_data.base_data_shape = ( 4, )

    # Setup some necessary data
    g_data.data = {
      'P' : np.random.rand( 3, 4 ),
      'Den' : np.array( [ 1e-6, 1e-4, 1e2, 1e-2 ] ),
      'R' : np.array( [ 0.25, 0.4999, 1.0, 0.5111 ] )*g_data.length_scale,
    }

    self.data_masker = generic_data.DataMasker( g_data )

  ########################################################################

  def test_mask_data( self ):

    self.data_masker.mask_data( 'logDen', -5., -1., )

    actual = self.data_masker.masks[0]['mask']
    expected = np.array( [ 1, 0, 1, 0 ] ).astype( bool )

    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_mask_data_returns( self ):

    actual = self.data_masker.mask_data( 'Rf', 0., 0.5, 'return' )
    expected = np.array( [ 0, 0, 1, 1 ] ).astype( bool )

    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_get_total_mask( self ):

    # Setup some masks first.
    self.data_masker.mask_data( 'logDen', -5., -1., )
    self.data_masker.mask_data( 'Rf', 0., 0.5, )

    actual = self.data_masker.get_total_mask()
    expected = np.array( [ 1, 0, 1, 1 ] ).astype( bool )

    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_get_masked_data( self ):

    # Setup some masks first.
    self.data_masker.mask_data( 'logDen', -5., -1., )
    self.data_masker.mask_data( 'Rf', 0., 0.5, )

    actual = self.data_masker.get_masked_data( 'Den' )

    expected = np.array( [ 1e-4 ] )

    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_get_masked_data_specified_mask( self ):

    mask = np.array( [ 1, 1, 0, 1 ] ).astype( bool )

    actual = self.data_masker.get_masked_data( 'Den', mask=mask )

    expected = np.array( [ 1e2 ] )

    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_get_masked_data_multi_dim( self ):
    '''Test we can get masked data even when we request P.'''

    mask = np.array( [ 1, 1, 0, 1 ] ).astype( bool )

    actual = self.data_masker.get_masked_data( 'P', mask=mask )

    expected = np.array( [ [ self.data_masker.generic_data.data['P'][i,2], ] for i in range(3) ] )

    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_get_masked_data_multi_dim_weird_shape( self ):
    '''Test we can get masked data even when we request P.'''

    mask = np.array( [ [ 1, 0, ], [ 1, 0 ] ] ).astype( bool )

    pos = np.random.rand( 3, 2, 2, )
    self.data_masker.generic_data.data['P'] = pos

    actual = self.data_masker.get_masked_data( 'P', mask=mask )

    expected = np.array( [ [ pos[i,0,1], pos[i,1,1] ] for i in range(3) ] )

    npt.assert_allclose( expected, actual )

########################################################################

class TestDataKeyParser( unittest.TestCase ):

  def setUp( self ):

    self.key_parser = generic_data.DataKeyParser()

  ########################################################################

  def test_is_position_data_key( self ):

    for data_key in [ 'Rx', 'Ry', 'Rz', 'R', 'P' ]:
      assert self.key_parser.is_position_key( data_key )

  ########################################################################

  def test_is_velocity_data_key( self ):

    for data_key in [ 'Vx', 'Vy', 'Vz', 'Vr', ]:
      assert self.key_parser.is_velocity_key( data_key )














