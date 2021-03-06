'''Testing for particle_data.py
'''

import copy
import mock
import numpy as np
import numpy.testing as npt
import pdb
import unittest

import galaxy_dive.analyze_data.generic_data as generic_data
import galaxy_dive.read_data.snapshot as read_snapshot

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
            'R' : np.array( [ 0.25, 0.4999, 1.0, 0.5111 ] ),
            'PType' : np.array( [ 0, 4, 4, 0, ] ),
        }

        self.data_masker = generic_data.DataMasker( g_data )

    ########################################################################

    def test_mask_data( self ):

        self.data_masker.mask_data( 'logDen', -5., -1., )

        actual = self.data_masker.masks[0]['mask']
        expected = np.array( [ 1, 0, 1, 0 ] ).astype( bool )

        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_mask_data_int( self ):

        self.data_masker.mask_data( 'PType', data_value=4 )

        actual = self.data_masker.masks[0]['mask']
        expected = np.array( [ 1, 0, 0, 1 ] ).astype( bool )

        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_mask_custom( self ):

        self.data_masker.mask_data( 'arbitrary', custom_mask=np.array( [ True, False, False, True, ] ), )

        actual = self.data_masker.masks[0]['mask']
        expected = np.array( [ 1, 0, 0, 1 ] ).astype( bool )

        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_mask_data_returns( self ):

        actual = self.data_masker.mask_data( 'R', 0., 0.5, return_or_store='return' )
        expected = np.array( [ 0, 0, 1, 1 ] ).astype( bool )

        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_mask_data_optional_masks( self ):
        '''Test that we can store a data mask as an optional mask.'''

        self.data_masker.mask_data( 'logDen', -5., -1., mask_name='test_mask_a', optional_mask=True )

        actual = self.data_masker.optional_masks['test_mask_a']['mask']
        expected = np.array( [ 1, 0, 1, 0 ] ).astype( bool )

        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_get_total_mask( self ):

        # Setup some masks first.
        self.data_masker.mask_data( 'logDen', -5., -1., )
        self.data_masker.mask_data( 'R', 0., 0.5, )

        actual = self.data_masker.get_total_mask()
        expected = np.array( [ 1, 0, 1, 1 ] ).astype( bool )

        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_get_total_mask_optional_masks_included( self ):
        '''Test that we can get the total mask out even when we use some optional masks
        '''

        # Setup some masks first.
        self.data_masker.mask_data( 'logDen', -5., -1., optional_mask=True )
        self.data_masker.mask_data( 'R', 0., 0.5, )

        actual = self.data_masker.get_total_mask( optional_masks=[ 'logDen', ] )
        expected = np.array( [ 1, 0, 1, 1 ] ).astype( bool )

        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_get_selected_data( self ):

        # Setup some masks first.
        self.data_masker.mask_data( 'logDen', -5., -1., )
        self.data_masker.mask_data( 'R', 0., 0.5, )

        actual = self.data_masker.get_selected_data( 'Den' )

        expected = np.array( [ 1e-4 ] )

        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_get_selected_data_optional_masks_included( self ):
        '''Test we can use get_selected_data even when we have optional masks included.
        '''

        # Setup some masks first.
        self.data_masker.mask_data( 'logDen', -5., -1., optional_mask=True )
        self.data_masker.mask_data( 'R', 0., 0.5, )

        actual = self.data_masker.get_selected_data( 'Den', optional_masks=[ 'logDen', ] )

        expected = np.array( [ 1e-4 ] )

        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_get_selected_data_specified_mask( self ):

        mask = np.array( [ 1, 1, 0, 1 ] ).astype( bool )

        actual = self.data_masker.get_selected_data( 'Den', mask=mask )

        expected = np.array( [ 1e2 ] )

        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_get_selected_data_multi_dim( self ):
        '''Test we can get masked data even when we request P.'''

        mask = np.array( [ 1, 1, 0, 1 ] ).astype( bool )

        actual = self.data_masker.get_selected_data( 'P', mask=mask )

        expected = np.array( [ [ self.data_masker.data_object.data['P'][i,2], ] for i in range(3) ] )

        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_get_selected_data_multi_dim_weird_shape( self ):
        '''Test we can get masked data even when we request P.'''

        mask = np.array( [ [ 1, 0, ], [ 1, 0 ] ] ).astype( bool )

        pos = np.random.rand( 3, 2, 2, )
        self.data_masker.data_object.data['P'] = pos

        actual = self.data_masker.get_selected_data( 'P', mask=mask )

        expected = np.array( [ [ pos[i,0,1], pos[i,1,1] ] for i in range(3) ] )

        npt.assert_allclose( expected, actual )

    ########################################################################

########################################################################
########################################################################

class TestGetPreprocessedData( unittest.TestCase ):

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
            'R' : np.array( [ 0.25, 0.4999, 1.0, 0.5111 ] ),
            'PType' : np.array( [ 0, 4, 4, 0, ] ),
        }

        self.g_data = g_data

    ########################################################################

    @mock.patch( 'galaxy_dive.analyze_data.generic_data.GenericData.get_data_alt', create=True )
    def test_different_get_data_method( self, mock_get_data_alt ):

        mock_get_data_alt.side_effect = [ np.array([ 1., 10., 100., ]), ]

        actual = self.g_data.get_processed_data( 'logZ', sl=1, data_method='get_data_alt' )
        expected = np.array([ 0., 1., 2., ])

        npt.assert_allclose( actual, expected )

        mock_get_data_alt.assert_called_once_with( 'Z', sl=1 )

########################################################################
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

########################################################################
########################################################################

class TestMetaMethods( unittest.TestCase ):

    def setUp( self ):

        self.g_data = generic_data.GenericData( **default_kwargs )

        self.data_masker = generic_data.DataMasker( self.g_data )

    ########################################################################

    @mock.patch( 'galaxy_dive.analyze_data.generic_data.GenericData.test_method', create=True )
    def test_iterate_over_method( self, mock_test_method ):

        method_args = { 'b' : 2, }
        self.g_data.iterate_over_method( 'test_method', 'a', [ 1, 2, ], method_args )

        calls = [ mock.call( a=1, b=2 ), mock.call( a=2, b=2 ), ]
        mock_test_method.assert_has_calls( calls )












