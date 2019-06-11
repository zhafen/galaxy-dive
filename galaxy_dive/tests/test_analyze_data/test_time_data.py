'''Testing for particle_data.py
'''

import mock
import numpy as np
import numpy.testing as npt
import pandas as pd
import unittest

import galaxy_dive.analyze_data.simulation_data as simulation_data
import galaxy_dive.utils.astro as astro

########################################################################


class SaneEqualityArray(np.ndarray):
    '''Numpy array subclass that allows you to test if two arrays are equal.'''

    def __eq__(self, other):
        return (isinstance(other, np.ndarray) and self.shape == other.shape
                and np.allclose(self, other))


def sane_eq_array(list_in):
    '''Wrapper for SaneEqualityArray, that takes in a list.'''

    arr = np.array(list_in)

    return arr.view(SaneEqualityArray)

########################################################################


default_kwargs = {
    'data_dir': './tests/data/sdir',
    'halo_data_dir': './tests/data/analysis_dir',
    'snum': [600, 550, 500],
    'ahf_index': 600,
    'store_ahf_reader': True,
}

########################################################################


class TestTimeData( unittest.TestCase ):

    def setUp( self ):

        self.t_data = simulation_data.TimeData( **default_kwargs )

        self.t_data.data = {
            'P': np.zeros( ( 3, 4, 5 ) ),
            'V': np.zeros( ( 3, 4, 5 ) ),
        }

        self.t_data.data_attrs = {
            'hubble': 0.70199999999999996,
            'omega_matter': 0.272,
            'omega_lambda': 0.728,
        }

    ########################################################################

    def test_retrieve_halo_data( self ):

        # Setup Mock Data
        self.t_data.data = { 'snum': np.array([ 600, 550, 500 ]), }

        self.t_data.retrieve_halo_data()

        # Make sure we have the right redshift
        expected = np.array([ 0.             ,  0.0698467,  0.16946  ])
        actual = self.t_data.redshift
        npt.assert_allclose( expected, actual )

        # Make sure we have the position right for mt halo 0, snap 500
        actual = self.t_data.halo_coords
        expected = np.array([
            [ 41792.16330006,  44131.23097357,  46267.67030715],
            [ 39109.186634  ,  41182.20415729,  43161.67681807],
            [ 35829.92061045,  37586.13756083,  39375.69771064]
        ]).transpose()
        npt.assert_allclose( expected, actual )

        actual = self.t_data.halo_velocity
        expected = np.array([
            [-48.53,    72.1 ,  96.12],
            [-49.05,    72.73,  96.86],
            [-48.89,    73.77,  97.25]
        ]).transpose()
        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_center_coords( self ):

        self.t_data.center_method = np.array( [ [ 1., 2., 3., 4., 5. ], ]*3 )

        self.t_data.center_coords()
        expected = np.array( [ [ [-1., -2., -3., -4., -5.], ]*4, ]*3 )
        actual = self.t_data.data['P']
        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_center_vel_coords( self ):

        self.t_data.vel_center_method = np.array( [ [ 1., 2., 3., 4., 5. ], ]*3 )

        self.t_data.center_vel_coords()
        expected = np.array( [ [ [-1., -2., -3., -4., -5.], ]*4, ]*3 )
        actual = self.t_data.data['V']
        npt.assert_allclose( expected, actual )

    ########################################################################

    @mock.patch( 'galaxy_dive.analyze_data.simulation_data.SimulationData.get_data' )
    @mock.patch( 'galaxy_dive.analyze_data.simulation_data.SimulationData.hubble_z', new_callable=mock.PropertyMock )
    def test_add_hubble_flow( self, mock_hubble_z, mock_get_data ):

        # Change the mock data we're using
        self.t_data.vel_centered = True
        self.t_data.data['P'] = np.array( [ [ [1., 2., 3., 4., 5.], ]*4, ]*3 )

        # Create side effects for the mock functions
        mock_hubble_z.side_effect = [ pd.Series( [ 1., ]*5 ), ]*2
        mock_get_data.side_effect = [ self.t_data.data['P'], ]

        self.t_data.add_hubble_flow()

        actual = self.t_data.data['V']
        expected = np.array( [ [ [1., 2., 3., 4., 5.], ]*4, ]*3 )
        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_get_data( self ):

        # Change the mock data
        self.t_data.data['P'] = np.array( [ [ [1., 2., 3., 4., 5.], ]*4, ]*3 )

        self.t_data.centered = True
        self.t_data.vel_centered = True

        expected = np.array( [ [1., 2., 3., 4., 5.], ]*4 )
        actual = self.t_data.get_data( 'Rx' )
        npt.assert_allclose( expected, actual )

    ########################################################################

    @mock.patch( 'galaxy_dive.analyze_data.ahf.HaloData.get_mt_data' )
    def test_get_processed_data( self, mock_get_mt_data ):
        '''Test that our processed data method works, especially when scaling data.'''

        # Setup the mock data
        self.t_data.data['P'] = np.array( [ [ [1., 2., 3., 4., 5.], ]*4, ]*3 )
        self.t_data.data['snum'] = sane_eq_array( [ 600, 550, 500, ] )
        mock_get_mt_data.side_effect = [ np.array([1., 2., 3., np.nan, np.nan ]) ]

        # So we don't have to do extra calculations that change things
        self.t_data.centered = True
        self.t_data.vel_centered = True

        expected = np.array( [ [1., 1., 1., np.nan, np.nan], ]*4 )*self.t_data.data_attrs['hubble']
        actual = self.t_data.get_processed_data( 'Rx', scale_key='Rvir', scale_a_power=1., scale_h_power=-1., )
        npt.assert_allclose( expected, actual )

        mock_get_mt_data.assert_called_once_with( 'Rvir', mt_halo_id=0, a_power=1., snums=self.t_data.data['snum'] )

    ########################################################################

    def test_hubble_z( self ):

        self.t_data.redshift = np.array([ 0.             ,  0.0698467,  0.16946  ])

        expected = np.array( [ astro.hubble_parameter( redshift, units='km/s/kpc' ) for redshift in self.t_data.redshift ] )
        actual = self.t_data.hubble_z
        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_redshift_pd_df( self ):
        '''Sometimes we have a pandas DataFrame with NaNs in it as the redshift, and that causes trouble.
        '''

        data = {
            'redshift': np.array( [1., 2., np.nan ] ),
        }
        df = pd.DataFrame( data, index=np.array( [ 1, 2, 3 ] ) )

        self.t_data._redshift = df['redshift']
        self.t_data.redshift = np.array( [1., 2., 3.])

    ########################################################################

    def test_redshift_pd_df_nan_alternate( self ):
        '''Having nans in either array (due to how things are defined when redshift is too low) can cause problems.
        '''

        data = {
            'redshift': np.array( [1., 2., 3. ] ),
        }
        df = pd.DataFrame( data, index=np.array( [ 1, 2, 3 ] ) )

        self.t_data._redshift = df['redshift']
        self.t_data.redshift = np.array( [1., 2., np.nan ] )

########################################################################
########################################################################

class TestGetData( unittest.TestCase ):

    def setUp( self ):

        self.t_data = simulation_data.TimeData( **default_kwargs )

        self.t_data.data = {
            'P': np.zeros( ( 3, 4, 5 ) ),
            'V': np.zeros( ( 3, 4, 5 ) ),
        }

        self.t_data.data_attrs = {
            'hubble': 0.70199999999999996,
            'omega_matter': 0.272,
            'omega_lambda': 0.728,
        }

    ########################################################################

    def test_get_selected_data_over_time( self ):
        '''Make sure we can get data based on its classification at a
        particular time.
        '''

        # Mock test data
        self.t_data.data['M'] = np.array([
            [ 1., 2., 3., np.nan, np.nan ],
            [ 4., 5., 6., np.nan, np.nan ],
            [ 7., 8., 9., np.nan, np.nan ],
            [ 11., 12., 13., np.nan, np.nan ],
        ])
        self.t_data.data['T'] = np.array([
            [ 11., 12., 13., np.nan, np.nan ],
            [ 7., 8., 9., np.nan, np.nan ],
            [ 4., 5., 6., np.nan, np.nan ],
            [ 1., 2., 3., np.nan, np.nan ],
        ])
        self.t_data.data['snum'] = np.array([
            600, 550, 500, 450, 400,
        ])

        # Ignore this warning, because we want it to show up.
        with np.errstate(divide='ignore',invalid='ignore'):
        # Mask some data
            self.t_data.data_masker.mask_data( 'T', 4.5, 8.5 )

        # Actual calculation
        actual = self.t_data.get_selected_data_over_time(
            'M',
            snum = 550,
        )

        expected = np.array([
            [ 4., 5., 6., np.nan, np.nan ],
            [ 7., 8., 9., np.nan, np.nan ],
        ])

        npt.assert_allclose( actual, expected )

    ########################################################################

    def test_get_selected_data_over_time_sample( self ):
        '''Make sure we can get data based on its classification at a
        particular time.
        '''

        # Mock test data
        n_particles = int( 1e3 )
        n_snapshots = 550
        self.t_data.data['M'] = np.random.randn( n_particles, n_snapshots )
        self.t_data.data['T'] = np.random.randn( n_particles, n_snapshots )
        self.t_data.data['snum'] = np.arange( 600, 600-n_snapshots, -1 )

        # Mask some data
        self.t_data.data_masker.mask_data( 'T', 0., np.inf )

        # Actual calculation
        n_samples = 10
        first = self.t_data.get_selected_data_over_time(
            'M',
            snum = 550,
            n_samples = n_samples,
        )

        # Make sure we have the right shape
        self.assertEqual( first.shape, (n_samples, n_snapshots) )

        # Our data shouldn't look the same, unless by rare chance.
        second = self.t_data.get_selected_data_over_time(
            'M',
            snum = 550,
            n_samples = n_samples,
        )
        assert not np.allclose( first, second )

    ########################################################################

    def test_get_selected_data_over_time_sample_seed( self ):
        '''Make sure we can get data based on its classification at a
        particular time.
        '''

        # Mock test data
        n_particles = int( 1e3 )
        n_snapshots = 550
        self.t_data.data['M'] = np.random.randn( n_particles, n_snapshots )
        self.t_data.data['T'] = np.random.randn( n_particles, n_snapshots )
        self.t_data.data['snum'] = np.arange( 600, 600-n_snapshots, -1 )

        # Mask some data
        self.t_data.data_masker.mask_data( 'T', 0., np.inf )

        seed = np.random.randint( 1e7 )

        # Actual calculation
        n_samples = 10
        first = self.t_data.get_selected_data_over_time(
            'M',
            snum = 550,
            n_samples = n_samples,
            seed = seed,
        )

        # Make sure we have the right shape
        self.assertEqual( first.shape, (n_samples, n_snapshots) )

        # Our data shouldn't look the same, unless by rare chance.
        second = self.t_data.get_selected_data_over_time(
            'M',
            snum = 550,
            n_samples = n_samples,
            seed = seed,
        )
        assert np.allclose( first, second )

    ########################################################################

    def test_get_selected_data_over_time_particle_ind( self ):
        '''Make sure we can get data based on its classification at a
        particular time.
        '''

        # Mock test data
        n_particles = int( 1e3 )
        n_snapshots = 550
        self.t_data.data['M'] = np.random.randn( n_particles, n_snapshots )
        self.t_data.data['T'] = np.random.randn( n_particles, n_snapshots )
        self.t_data.data['snum'] = np.arange( 600, 600-n_snapshots, -1 )

        # Mask some data
        self.t_data.data_masker.mask_data( 'T', 0., np.inf )

        seed = np.random.randint( 1e7 )

        # Actual calculation
        n_samples = 10
        actual = self.t_data.get_selected_data_over_time(
            'particle_ind',
            snum = 550,
            n_samples = n_samples,
            seed = seed,
        )
        expected = np.array(
            [ np.arange( n_samples ), ]*n_snapshots
        ).transpose()

        npt.assert_allclose( expected, actual )

########################################################################
########################################################################

class TestCalc( unittest.TestCase ):

    def setUp( self ):

        self.t_data = simulation_data.TimeData( **default_kwargs )

        self.t_data.data_attrs = {
            'hubble': 0.70199999999999996,
            'omega_matter': 0.272,
            'omega_lambda': 0.728,
        }

    ########################################################################

    def test_inverse_classification( self ):
        '''When a particle next enters a classification calculate the time it
        will spend as that classification.
        '''

        # Set up test data
        self.t_data.data = {}
        self.t_data.data['is_A'] = np.array([
            [ 1, 1, 1, 0, ],
            [ 1, 0, 0, 0, ],
            [ 1, 0, 1, 0, ],
            [ 0, 1, 1, 0, ],
        ]).astype( bool )

        actual = self.t_data.get_data( 'not_is_A' )
        expected = np.array([
            [ 0, 0, 0, 1, ],
            [ 0, 1, 1, 1, ],
            [ 0, 1, 0, 1, ],
            [ 1, 0, 0, 1, ],
        ]).astype( bool )
        npt.assert_allclose( expected, actual )

        actual = self.t_data.get_data( 'is_not_A' )
        expected = np.array([
            [ 0, 0, 0, 1, ],
            [ 0, 1, 1, 1, ],
            [ 0, 1, 0, 1, ],
            [ 1, 0, 0, 1, ],
        ]).astype( bool )
        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_calc_time_as_classification( self ):
        '''Calculate the time spent as a certain classification.
        '''

        # Set up test data
        self.t_data.data = {}
        self.t_data.data['is_A'] = np.array([
            [ 1, 1, 1, ],
            [ 1, 0, 1, ],
            [ 1, 1, 0, ],
            [ 1, 0, 0, ],
        ]).astype( bool )
        self.t_data.data['dt'] = np.array([
            1.0, 2.0, 3.0,
        ])

        self.t_data.calc_time_as_classification( 'time_as_A' )

        actual = self.t_data.data['time_as_A']
        expected = np.array([
            [ 6., 5., 3., ],
            [ 1., 0., 3., ],
            [ 3., 2., 0., ],
            [ 1., 0., 0., ],
        ])

        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_calc_time_until_not_classification( self ):
        '''Calculate the time until not a certain classification
        '''

        # Set up test data
        self.t_data.data = {}
        self.t_data.data['is_A'] = np.array([
            [ 1, 1, 1, ],
            [ 1, 0, 1, ],
            [ 1, 1, 0, ],
            [ 0, 1, 1, ],
        ]).astype( bool )
        self.t_data.data['dt'] = np.array([
            1.0, 2.0, 3.0,
        ])

        self.t_data.calc_time_until_not_classification( 'time_until_not_A' )

        actual = self.t_data.data['time_until_not_A']
        expected = np.array([
            [ 0., 1., 3., ],
            [ 0., 0., 2., ],
            [ 0., 1., 0., ],
            [ 0., 1., 3., ],
        ])

        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_calc_next_time_as_classification( self ):
        '''When a particle next enters a classification calculate the time it
        will spend as that classification.
        '''

        # Set up test data
        self.t_data.data = {}
        self.t_data.data['is_A'] = np.array([
            [ 1, 1, 1, 0, ],
            [ 1, 0, 0, 0, ],
            [ 1, 0, 1, 0, ],
            [ 0, 1, 1, 0, ],
        ]).astype( bool )
        self.t_data.data['dt'] = np.array([
            1.0, 2.0, 3.0,
        ])

        self.t_data.calc_next_time_as_classification( 'next_time_as_A' )

        actual = self.t_data.data['next_time_as_A']
        expected = np.array([
            [ 6., 6., 6., 6., ],
            [ 1., 1., 1., 1., ],
            [ 1., 1., 3., 3., ],
            [ 0., 5., 5., 5., ],
        ])

        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_calc_radial_velocity( self ):

        # Setup Mock Data
        self.t_data.data = {
            'P': np.zeros( ( 3, 5, 4 ) ),
            'V': np.zeros( ( 3, 5, 4 ) ),
            'Den': np.zeros( ( 5, 4 ) ), # Five particles, 4 snapshots
            'snum': np.array([ 600, 550, 500, 450 ]),
        }
        # Offset the data slightly from the origin
        self.t_data.retrieve_halo_data()
        for i in range( 5 ):
            offset = 10. * i * np.ones( shape=self.t_data.halo_coords.shape )
            self.t_data.data['P'][:,i] = self.t_data.halo_coords + offset

            vel_offset = 1. * i * np.ones( shape=self.t_data.halo_coords.shape )
            self.t_data.data['V'][:,i] = self.t_data.halo_velocity + vel_offset

        # Don't report divide by zero warnings, because I want those to happen
        with np.errstate(divide='ignore',invalid='ignore'):
            self.t_data.calc_radial_velocity()

        assert self.t_data.data['Vr'].shape == ( 5, 4 )

        # The first particle is located at the origin, so it should be NaN
        for i in range( 4 ):
            assert np.isnan( self.t_data.data['Vr'][0,i] )

    ########################################################################

    def test_calc_radial_distance( self ):

        # Change the mock data
        self.t_data.data = {
            'P': np.array( [ [ [1., 2., 3., 4., 5.], ]*4, ]*3 ),
            'V': np.zeros( ( 3, 4, 5 ) ),
        }
        

        self.t_data.centered = True
        self.t_data.vel_centered = True

        expected = np.array( [ [1., 2., 3., 4., 5.], ]*4 )*np.sqrt( 3 )
        actual = self.t_data.get_data( 'R' )
        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_calc_ang_momentum( self ):

        # Mock data
        self.t_data.data = {
            'M': np.random.randn( 4, 5 ),
            'P': np.random.randn( 3, 4, 5 ),
            'V': np.random.randn( 3, 4, 5 ),
        }
        # Mock data for particle index 1 in the 3rd snapshot
        self.t_data.data['M'][1,2] = 2.
        self.t_data.data['P'][:,1,2] = np.array([ 1., 2., 3. ])
        self.t_data.data['V'][:,1,2] = np.array([ 3., 5., 6. ])

        # Make sure we don't do automatic calculations not relevant
        self.t_data.centered = True
        self.t_data.vel_centered = True
        self.t_data.hubble_corrected = True

        self.t_data.calc_ang_momentum()

        actual = self.t_data.get_data( 'L' )

        # Make sure we get the general right shape
        assert self.t_data.data['P'].shape == actual.shape

        # Make sure we get exactly the right answer for a particular random
        # particle at a random snapshot
        expected_explicit = np.array([ -6., 6., -2. ])
        npt.assert_allclose( expected_explicit, actual[:,1,2] )

    ########################################################################

    def test_calc_phi( self ):

        # Mock data
        self.t_data.data = {
            'M': np.random.randn( 4, 5 ),
            'P': np.random.randn( 3, 4, 5 ),
            'V': np.random.randn( 3, 4, 5 ),
        }
        # Mock data for particle index 1 in the 3rd snapshot
        expected_phi = 180. - 15. # in degrees
        phi_r = expected_phi * np.pi / 180. # in radians
        theta = 50. * np.pi / 180.
        self.t_data.data['P'][:,1,2] = np.array([
            5. * np.cos( theta ) * np.sin( phi_r ),
            5. * np.sin( theta ) * np.sin( phi_r ),
            5. * np.cos( phi_r ),
        ])

        # Make sure we don't do automatic calculations not relevant
        self.t_data.centered = True
        self.t_data.vel_centered = True
        self.t_data.hubble_corrected = True

        # Normal vector will just be in the z-direction
        normal_vector = np.array([ 0., 0., 2. ])

        # Actual calculation
        self.t_data.calc_phi( normal_vector=normal_vector )

        # Get the data
        actual = self.t_data.data['Phi']

        # Make sure we get the general right shape
        assert self.t_data.data['M'].shape == actual.shape

        # Check that we get the right value
        npt.assert_allclose( expected_phi, actual[1,2] )

    ########################################################################

    def test_calc_abs_phi( self ):

        # Mock data
        self.t_data.data = {
            'M': np.random.randn( 4, 5 ),
            'Phi': np.random.randn( 4, 5 ),
        }
        # Mock data for particular indices
        self.t_data.data['Phi'][1,2] = 180. - 15.
        self.t_data.data['Phi'][2,2] = 15.
        self.t_data.normal_vector = np.array( [ 1., 2., 3. ] )

        # Actual calculation
        self.t_data.calc_abs_phi( [ 1., 2., 3. ] )

        # Get the data
        actual = self.t_data.data['AbsPhi']

        # Test that we get the right shape
        assert self.t_data.data['M'].shape == actual.shape

        # Check that we get the right value
        npt.assert_allclose( 15., actual[1,2] )
        npt.assert_allclose( 15., actual[2,2] )

    ########################################################################

    # TODO: If I want to do this, that is
    # def test_calc_h( self ):

    #     # Mock data
    #     self.t_data.data = {
    #         'M': np.random.randn( 4, 5 ),
    #         'P': np.random.randn( 3, 4, 5 ),
    #         'V': np.random.randn( 3, 4, 5 ),
    #     }
    #     # Mock data for particle index 1 in the 3rd snapshot
    #     expected_phi = 180. - 15. # in degrees
    #     phi_r = expected_phi * np.pi / 180. # in radians
    #     theta = 50. * np.pi / 180.
    #     self.t_data.data['P'][:,1,2] = np.array([
    #         5. * np.cos( theta ) * np.sin( phi_r ),
    #         5. * np.sin( theta ) * np.sin( phi_r ),
    #         5. * np.cos( phi_r ),
    #     ])
    #     self.t_data.data['P'][:1,2] = np.array([
    #         

    #     # Make sure we don't do automatic calculations not relevant
    #     self.t_data.centered = True
    #     self.t_data.vel_centered = True
    #     self.t_data.hubble_corrected = True

    #     # Normal vector will just be in the z-direction
    #     normal_vector = np.array([ 0., 0., 2. ])

    #     # Actual calculation
    #     self.t_data.calc_phi( normal_vector=normal_vector )

    #     # Get the data
    #     actual = self.t_data.data['Phi']

    #     # Make sure we get the general right shape
    #     assert self.t_data.data['M'].shape == actual.shape

    #     # Check that we get the right value
    #     npt.assert_allclose( expected_phi, actual[1,2] )
