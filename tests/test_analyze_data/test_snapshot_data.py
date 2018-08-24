'''Testing for particle_data.py
'''

import copy
from mock import patch, sentinel
import numpy as np
import numpy.testing as npt
import unittest

import galaxy_dive.analyze_data.simulation_data as simulation_data

########################################################################

default_kwargs = {
    'data_dir': './tests/data/sdir',
    'halo_data_dir': './tests/data/analysis_dir',
    'snum': 500,
    'ahf_index': 600,
    'length_scale_used' : 'r_scale',
    'averaging_frac' : 0.5,
}

########################################################################

class TestSnapshotData( unittest.TestCase ):

    def setUp( self ):

        self.g_data = simulation_data.SnapshotData( **default_kwargs )

        self.g_data.data_attrs = {
            'hubble': 0.70199999999999996,
            'redshift': 0.16946,
        }

    ########################################################################

    def test_retrieve_halo_data( self ):

        self.g_data.retrieve_halo_data()

        # Make sure we have the right redshift
        expected = 0.16946
        actual = self.g_data.redshift

        # Make sure we have the position right for mt halo 0, snap 500
        actual = self.g_data.halo_coords
        expected = np.array( [ 29414.96458784, 30856.75007114, 32325.90901812] )/(1. + 0.16946 )/0.70199999999999996
        npt.assert_allclose( expected, actual )

        actual = self.g_data.halo_velocity
        expected = np.array( [-48.89, 73.77, 97.25] )

########################################################################

class TestGetData( unittest.TestCase ):

    def setUp( self ):

        self.g_data = simulation_data.SnapshotData( **default_kwargs )

        self.g_data.data_attrs = {
            'hubble': 0.70199999999999996,
            'omega_matter': 0.272,
            'omega_lambda': 0.728,
            'redshift': 0.16946,
        }

        # Setup some necessary data
        self.g_data.data = {
            'P': np.random.rand( 3, 4 ),
            'V': np.random.rand( 3, 4 ),
            'Den': np.random.rand( 4 ),
            'Z': np.random.uniform( 0., 1. ),
        }

    ########################################################################

    def test_get_position_data( self ):

        rx_before = copy.copy( self.g_data.data['P'][0] )

        actual = self.g_data.get_position_data( 'Rx' )

        expected = rx_before - self.g_data.halo_coords[0]

        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_get_velocity_data( self ):

        # So we don't try and find the hubble velocity
        self.g_data.hubble_corrected = True

        vx_before = copy.copy( self.g_data.data['V'][0] )

        actual = self.g_data.get_velocity_data( 'Vx' )

        expected = vx_before - self.g_data.halo_velocity[0]

        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_get_data_fails( self ):

        self.assertRaises( KeyError, self.g_data.get_data, 'NonexistentData' )

    ########################################################################

    @patch( 'galaxy_dive.analyze_data.simulation_data.SnapshotData.handle_data_key_error' )
    def test_fails_after_too_many_attempts( self, mock_handle_data_key_error ):
        '''By mocking handle_data_key error, we can emulate it trying to do something'''

        self.assertRaises( KeyError, self.g_data.get_data, 'NonexistentData' )

    ########################################################################

    def test_get_processed_data_standard( self ):
        '''When nothing changes and processed is the regular.'''

        expected = self.g_data.get_data( 'Rx' )
        actual = self.g_data.get_processed_data( 'Rx' )
        npt.assert_allclose( expected, actual )

        expected = self.g_data.get_data( 'Den' )
        actual = self.g_data.get_processed_data( 'Den' )
        npt.assert_allclose( expected, actual )

        expected = self.g_data.get_data( 'Z' )
        actual = self.g_data.get_processed_data( 'Z' )
        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_get_processed_data_log( self ):

        expected = np.log10( self.g_data.get_data( 'Den' ) )
        actual = self.g_data.get_processed_data( 'logDen' )
        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_get_processed_data_fraction( self ):

        expected = self.g_data.get_data( 'Rx' )/self.g_data.length_scale
        actual = self.g_data.get_processed_data( 'Rxf' )
        npt.assert_allclose( expected, actual )

        expected = self.g_data.get_data( 'Z' )/self.g_data.metallicity_scale
        actual = self.g_data.get_processed_data( 'Zf' )
        npt.assert_allclose( expected, actual )


########################################################################

class TestHandleDataKeyError( unittest.TestCase ):

    def setUp( self ):

        self.g_data = simulation_data.SnapshotData( **default_kwargs )

    ########################################################################

    @patch.multiple( 'galaxy_dive.analyze_data.simulation_data.SnapshotData',
        calc_radial_distance=sentinel.DEFAULT, calc_radial_velocity=sentinel.DEFAULT,
        calc_inds=sentinel.DEFAULT, calc_ang_momentum=sentinel.DEFAULT,
        calc_phi=sentinel.DEFAULT, calc_abs_phi=sentinel.DEFAULT,
        calc_num_den=sentinel.DEFAULT, calc_H_den=sentinel.DEFAULT,
        calc_HI_den=sentinel.DEFAULT )
    def test_handle_data_key_error( self, **mocks ):
        '''Make sure that passing a data_key successfully calls the right function.'''

        keys_to_check = [ 'R', 'Vr', 'ind', 'L', 'Phi', 'AbsPhi', 'NumDen', 'HDen', 'HIDen', ]
        for key in keys_to_check:
            self.g_data.handle_data_key_error( key )

        for key in mocks.keys():
            mocks[key].assert_called_once()

    ########################################################################

    def test_fails( self ):

        self.assertRaises( KeyError, self.g_data.handle_data_key_error, 'NonexistentData' )

########################################################################

class TestCenterCoords( unittest.TestCase ):

    def setUp( self ):

        self.g_data = simulation_data.SnapshotData( **default_kwargs )

        self.g_data.data_attrs = {
            'hubble': 0.70199999999999996,
            'redshift': 0.16946,
        }

        # Setup some necessary data
        self.g_data.data = {
            'P': np.random.rand( 3, 4 ),
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

        self.g_data = simulation_data.SnapshotData( **default_kwargs )

        self.g_data.data_attrs = {
            'hubble': 0.70199999999999996,
            'redshift': 0.16946,
        }

        # Setup some necessary data
        self.g_data.data = {
            'V': np.random.rand( 3, 4 ),
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

        self.g_data = simulation_data.SnapshotData( **default_kwargs )

        self.g_data.data_attrs = {
            'hubble': 0.70199999999999996,
            'omega_matter': 0.272,
            'omega_lambda': 0.728,
            'redshift': 0.16946,
        }

    ########################################################################

    def test_v_com( self ):
        '''Get the com velocity'''

        # So that we don't deal with cosmological junk when testing.
        self.g_data.centered = True
        self.g_data.vel_centered = True
        self.g_data.hubble_corrected = True

        self.g_data.data = {

            # Have two particles inside and one outside the region.
            'P': np.array( [
                [ 1., 0., 0., 0., ],
                [ 2., 0., 0., 0., ],
                [ 0., 1., 0., 0., ],
            ] )*self.g_data.length_scale*self.g_data.averaging_frac*0.9,

            # Have the particle outside have an insane velocity so we notice if it's affecting things.
            'V': np.array( [
                [ 10., 0., -5., 0., ],
                [ 200., 0., 10., -10., ],
                [ 0., 10., 0., 0., ],
                ] ),

            'M': np.array( [ 1., 1., 1., 1. ] ),

            'Den': np.random.rand( 4 ),

        }

        actual = self.g_data.v_com

        expected = np.array( [ -5./3., 0., 10./3. ] )

        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_hubble_z( self ):

        # Hubble parameter in km/s/kpc
        expected = 75.71*1e-3

        actual = self.g_data.hubble_z

        npt.assert_allclose( expected, actual, rtol=1e-4 )

        try:
            self.g_data.hubble_z = 2.
            assert False

        # We're actually looking for an error here, to test for read only
        except AttributeError:
            pass

    ########################################################################

    def test_redshift( self ):

        # By hand
        expected = 0.16946

        actual = self.g_data.redshift

        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_redshift_halo_data( self ):

        del self.g_data.data_attrs['redshift']

        # By hand
        expected = 0.16946

        actual = self.g_data.redshift

        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_r_vir( self ):

        # By hand
        expected = 239.19530785947771

        actual = self.g_data.r_vir

        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_r_scale( self ):

        # By hand
        expected = 25.718158280533068

        actual = self.g_data.r_scale

        npt.assert_allclose( expected, actual, rtol=1e-3 )

    ########################################################################

    def test_v_c( self ):

        # By hand
        expected = 134.93489906417346

        actual = self.g_data.v_c

        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_length_scale( self ):
        '''Default options.'''

        expected = self.g_data.r_scale

        actual = self.g_data.length_scale

        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_length_scale_r_vir( self ):
        '''Default options.'''

        self.g_data.length_scale_used = 'R_vir'

        expected = self.g_data.r_vir

        actual = self.g_data.length_scale

        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_velocity_scale( self ):
        '''Default options.'''

        expected = self.g_data.v_c

        actual = self.g_data.velocity_scale

        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_metallicity_scale( self ):
        '''Default options.'''

        expected = self.g_data.z_sun

        actual = self.g_data.metallicity_scale

        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_base_data_shape( self ):

        self.g_data.data = { 'Den': np.random.rand( 5, 3, 2 ), }

        expected = ( 5, 3, 2 )
        actual = self.g_data.base_data_shape
        npt.assert_allclose( expected, actual )

        def try_to_change_shape():
            self.g_data.base_data_shape = ( 5, )

        self.assertRaises( AssertionError, try_to_change_shape, )

########################################################################

class TestHubbleFlow( unittest.TestCase ):

    def setUp( self ):

        self.g_data = simulation_data.SnapshotData( **default_kwargs )

        self.g_data.data_attrs = {
            'hubble': 0.70199999999999996,
            'redshift': 0.16946,
            'omega_matter': 0.272,
            'omega_lambda': 0.728,
        }

        # Setup some necessary data
        self.g_data.data = {
            'V': np.random.rand( 3, 4 ),
            'P': np.random.rand( 3, 4 ),
        }

    ########################################################################

    def test_add_hubble_flow( self ):

        self.g_data.center_method = np.random.rand( 3 )
        self.g_data.vel_center_method = np.random.rand( 3 )

        self.g_data.add_hubble_flow()

        assert self.g_data.hubble_corrected
        assert self.g_data.centered
        assert self.g_data.vel_centered

########################################################################

class TestCalcData( unittest.TestCase ):

    def setUp( self ):

        self.g_data = simulation_data.SnapshotData( **default_kwargs )

        self.g_data.centered = True
        self.g_data.r_scale = 1.

        self.g_data.data = {

            # Have two particles inside and one outside the region.
            'P':  np.array( [
                [  0.,  2.,  0.5, 0., ],
                [  0.5,  0.,  0., 0., ],
                [  0.,  0.5,  0.5, 0., ],
                ] )

        }

    ########################################################################

    def test_calc_radial_distance( self ):

        self.g_data.calc_radial_distance()

        actual = self.g_data.data['R']

        # By hand
        expected = np.array( [ 0.5, 2.0615528128088303, 0.70710678118654757, 0., ] )

        npt.assert_allclose( expected, actual )













