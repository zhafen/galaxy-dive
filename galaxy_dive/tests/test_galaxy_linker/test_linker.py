#!/usr/bin/env python
'''Testing for tracking.py

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import copy
import mock
import numpy as np
import numpy.testing as npt
import pytest
import unittest
import unyt

import galaxy_dive.read_data.ahf as read_ahf
import galaxy_dive.analyze_data.halo_data as analyze_halos
import galaxy_dive.galaxy_linker.linker as general_galaxy_linker

########################################################################
# Useful global test variables
########################################################################

gal_linker_kwargs = {
    'length_scale': 'Rvir',

    'redshift': 0.16946003,
    'snum': 500,
    'hubble': 0.70199999999999996,
    'halo_data_dir': './tests/data/analysis_dir3',
    'mtree_halos_index': 600,
    'main_mt_halo_id': 0,
    'halo_file_tag': 'smooth',

    'galaxy_cut': 0.1,
    'ids_to_return': [ 'halo_id', 'host_halo_id', 'gal_id', 'host_gal_id',
                       'mt_halo_id', 'mt_gal_id', 'd_gal',
                       'd_other_gal_scaled', '0.1_Rvir',  ],
    'minimum_criteria': 'n_star',
    'minimum_value': 0,

    'low_memory_mode': False,
}

########################################################################


class TestGalaxyLinker( unittest.TestCase ):

    def setUp( self ):

        # Get input data
        comoving_halo_coords = np.array([
            [ 29414.96458784, 30856.75007114, 32325.90901812],
            [ 31926.42103071, 51444.46756529, 1970.1967437 ] ])

        self.redshift = gal_linker_kwargs['redshift']
        self.hubble = gal_linker_kwargs['hubble']
        halo_coords = comoving_halo_coords / (1. + self.redshift) / self.hubble

        # Make the necessary kwargs
        self.kwargs = gal_linker_kwargs

        self.galaxy_linker = general_galaxy_linker.GalaxyLinker(
            halo_coords, **self.kwargs )

        # Get the necessary reader.
        self.galaxy_linker.halo_data.data_reader = read_ahf.AHFReader(
            self.kwargs['halo_data_dir'] )

        # Get the full needed ahf info.
        self.galaxy_linker.halo_data.data_reader.get_halos( 500 )

    ########################################################################

    @mock.patch( 'galaxy_dive.analyze_data.halo_data.HaloData.get_data' )
    def test_valid_halo_inds( self, mock_get_halo_data ):

        # Make sure we actually have a minimum
        self.galaxy_linker.minimum_value = 10

        # Mock the halo data
        mock_get_halo_data.side_effect = [ np.array( [ 100, 5, 10, 0 ] ), ]

        actual = self.galaxy_linker.valid_halo_inds

        expected = np.array( [ 0, 2, ] )

        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_dist_to_all_valid_halos( self ):
        '''Test that this works for using r_scale.'''

        self.galaxy_linker.particle_positions = np.array([
            # Right in the middle of mt halo 0 at snap 500
            [ 29414.96458784, 30856.75007114, 32325.90901812],
            # Just outside the scale radius of mt halo 0 at snap 500.
            [ 29414.96458784 + 50., 30856.75007114, 32325.90901812],
            # Just inside the scale radius of mt halo 0 at snap 500.
            [ 29414.96458784, 30856.75007114 - 25., 32325.90901812],
        ])
        self.galaxy_linker.particle_positions *= 1. / (1. + self.redshift) / \
            self.hubble
        self.galaxy_linker.n_particles = 3

        actual = self.galaxy_linker.dist_to_all_valid_halos

        # Build the expected output
        n_halos = self.galaxy_linker.halo_data.data_reader.halos.index.size
        n_particles = self.galaxy_linker.n_particles
        expected_shape = ( n_particles, n_halos )

        npt.assert_allclose( actual[ 0, 0 ], 0., atol=1e-7 )
        npt.assert_allclose( actual[ 1, 0 ], 50. * 1. / (1. + self.redshift) /
                             self.hubble )
        npt.assert_allclose( actual[ 2, 0 ], 25. * 1. / (1. + self.redshift) /
                             self.hubble )

        self.assertEqual( actual.shape, expected_shape )

    ########################################################################

    def test_dist_to_all_valid_halos_single_particle( self ):
        '''Test that we can get the distance to all valid halos,
        formatted correctly, even for a single particle.
        '''

        # Setup test data.
        particle_positions = np.array( [
            [ 29414.96458784, 30856.75007114, 32325.90901812 ]
        ] ) / ( ( 1. + self.redshift) * self.hubble )
        actual = self.galaxy_linker.dist_to_all_valid_halos_fn(
            particle_positions )

        # Build the expected output
        n_halos = self.galaxy_linker.halo_data.data_reader.halos.index.size
        n_particles = 1
        expected_shape = ( n_particles, n_halos )

        npt.assert_allclose( actual[ 0, 0, ], 0., atol=1e-7 )

        self.assertEqual( actual.shape, expected_shape )

    ########################################################################

    def test_find_containing_halos( self ):

        result = self.galaxy_linker.find_containing_halos()

        # If none of the distances are within any of the halos,
        # we have a problem.
        assert result.sum() > 0

    ########################################################################

    def test_find_containing_halos_strict( self ):
        '''Here I'll restrict the fraction to a very small fraction of the
        virial radius, such that the sum of the results should be two.
        '''

        result = self.galaxy_linker.find_containing_halos( 0.0001 )

        # If none of the distances are within any of the halos,
        # we have a problem.
        npt.assert_allclose( 2, result.sum() )

    ########################################################################

    def test_find_containing_halos_r_scale( self ):
        '''Test that this works for using r_scale.'''

        # Set the length scale
        self.galaxy_linker.galaxy_cut = 1.
        self.galaxy_linker.length_scale = 'r_scale'

        r_scale_500 = 21.113602882685832
        self.galaxy_linker.particle_positions = np.array([
            [ 29414.96458784,  30856.75007114,  32325.90901812], # Right in the middle of mt halo 0 at snap 500
            [ 29414.96458784 + r_scale_500*1.01,  30856.75007114,  32325.90901812], # Just outside the scale radius of mt halo 0 at snap 500.
            [ 29414.96458784 + r_scale_500*0.99,  30856.75007114,  32325.90901812], # Just inside the scale radius of mt halo 0 at snap 500.
            ])
        self.galaxy_linker.particle_positions *= 1./(1. + self.redshift)/self.hubble
        self.galaxy_linker.n_particles = 3

        actual = self.galaxy_linker.find_containing_halos( 1. )

        # Build the expected output
        n_halos = self.galaxy_linker.halo_data.data_reader.halos.index.size
        expected = np.zeros( (self.galaxy_linker.particle_positions.shape[0], n_halos) ).astype( bool )
        expected[ 0, 0 ] = True
        expected[ 1, 0 ] = False
        expected[ 2, 0 ] = True

        npt.assert_allclose( actual, expected )

    ########################################################################

    def test_find_containing_halos_nan_particle( self ):
        # Anywhere the particle data has NaN values, we want that to read as False

        self.galaxy_linker.particle_positions = np.array([
            [ 29414.96458784,  30856.75007114,  32325.90901812], # Right in the middle of mt halo 0 at snap 500
            [ np.nan, np.nan, np.nan ], # Invalid values, because a particle with that ID didn't exist
            ])
        self.galaxy_linker.particle_positions *= 1./(1. + self.redshift)/self.hubble
        self.galaxy_linker.n_particles = 2

        actual = self.galaxy_linker.find_containing_halos()

        n_halos = self.galaxy_linker.halo_data.data_reader.halos.index.size
        expected = np.zeros( (self.galaxy_linker.particle_positions.shape[0], n_halos) ).astype( bool )
        expected[ 0, 0 ] = True

        npt.assert_allclose( actual, expected )

    ########################################################################

    def test_find_mt_containing_halos( self ):

        self.galaxy_linker.particle_positions = np.array([
            [ 29414.96458784,  30856.75007114,  32325.90901812], # Right in the middle of mt halo 0 at snap 500
            [ 29467.07226789,  30788.6179313 ,  32371.38749237], # Right in the middle of mt halo 9 at snap 500.
                                                                                                                      # mt halo 9 is 0.5 Rvir_mt_0 (2 Rvir_mt_9) away from the center of mt halo 0
            [ 29073.22333685,  31847.72434505,  32283.53620817], # Right in the middle of mt halo 19 at snap 500.
            ])
        self.galaxy_linker.particle_positions *= 1./(1. + self.redshift)/self.hubble

        actual = self.galaxy_linker.find_mt_containing_halos( 2.5 )

        # Build the expected output
        expected = np.zeros( (self.galaxy_linker.particle_positions.shape[0], 6) ).astype( bool )
        expected[ 0, 0 ] = True
        expected[ 0, -2 ] = True
        expected[ 1, 0 ] = True
        expected[ 1, -2 ] = True
        expected[ 2, -1 ] = True

        npt.assert_allclose( actual, expected )

    ########################################################################

    def test_find_mt_containing_halos_r_scale( self ):
        '''Test that this works for using r_scale.'''

        # Set the length scale
        self.galaxy_linker.galaxy_cut = 1.
        self.galaxy_linker.mt_length_scale = 'r_scale'

        r_scale_500 = 21.113602882685832
        self.galaxy_linker.particle_positions = np.array([
            [ 29414.96458784,  30856.75007114,  32325.90901812], # Right in the middle of mt halo 0 at snap 500
            [ 29414.96458784 + r_scale_500*1.01,  30856.75007114,  32325.90901812], # Just outside the scale radius of mt halo 0 at snap 500.
            [ 29414.96458784 + r_scale_500*0.99,  30856.75007114,  32325.90901812], # Just inside the scale radius of mt halo 0 at snap 500.
                                                                                                                      # (It will be. It currently isn't.)
            ])
        self.galaxy_linker.particle_positions *= 1./(1. + self.redshift)/self.hubble
        self.galaxy_linker.n_particles = 3

        actual = self.galaxy_linker.find_mt_containing_halos( 1. )

        # Build the expected output
        n_halos = len( self.galaxy_linker.halo_data.data_reader.mtree_halos )
        expected = np.zeros( (self.galaxy_linker.particle_positions.shape[0], n_halos) ).astype( bool )
        expected[ 0, 0 ] = True
        expected[ 1, 0 ] = False
        expected[ 2, 0 ] = True

        npt.assert_allclose( actual, expected )

    ########################################################################

    def test_find_mt_containing_halos_nan_particles( self ):
        '''Test that this works for using r_scale.'''

        self.galaxy_linker.particle_positions = np.array([
            [ 29414.96458784,  30856.75007114,  32325.90901812], # Right in the middle of mt halo 0 at snap 500
            [ np.nan, np.nan, np.nan, ], # Just outside the scale radius of mt halo 0 at snap 500.
            ])
        self.galaxy_linker.particle_positions *= 1./(1. + self.redshift)/self.hubble
        self.galaxy_linker.n_particles = 2

        actual = self.galaxy_linker.find_mt_containing_halos( 1. )

        # Build the expected output
        n_halos = len( self.galaxy_linker.halo_data.data_reader.mtree_halos )
        expected = np.zeros( (self.galaxy_linker.particle_positions.shape[0], n_halos) ).astype( bool )
        expected[ 0, 0 ] = True

        npt.assert_allclose( actual, expected )

    ########################################################################

    def test_find_smallest_host_halo( self ):

        self.galaxy_linker.particle_positions = np.array([
            [ 29414.96458784,  30856.75007114,  32325.90901812],
            [ 31926.42103071,  51444.46756529,   1970.1967437 ],
            [ 29467.07226789,  30788.6179313 ,  32371.38749237],
            [ 29459.32290246,  30768.32556725,  32357.26078864], # Halo 3783, host halo 3610
            ])
        self.galaxy_linker.particle_positions *= 1./(1. + self.redshift)/self.hubble

        self.galaxy_linker.n_particles = 4

        expected = np.array( [0, 6962, 7, 3783] )
        actual = self.galaxy_linker.find_halo_id()

        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_find_smallest_host_halo_none( self ):

        self.galaxy_linker.particle_positions = np.array([
            [ 0., 0., 0. ],
            [ 0., 0., 0. ],
            ])

        expected = np.array( [-2, -2] )
        actual = self.galaxy_linker.find_halo_id()

        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_find_host_id( self ):

        self.galaxy_linker.particle_positions = np.array([
            [ 29414.96458784,  30856.75007114,  32325.90901812], # Halo 0, host halo 0
            [ 30068.5541178 ,  32596.72758226,  32928.1115097 ], # Halo 10, host halo 1
            [ 29459.32290246,  30768.32556725,  32357.26078864], # Halo 3783, host halo 3610
            ])
        self.galaxy_linker.particle_positions *= 1./(1. + self.redshift)/self.hubble

        self.galaxy_linker.n_particles = 3

        expected = np.array( [-1, 1, 3610] )
        actual = self.galaxy_linker.find_host_id()

        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_find_host_id_none( self ):

        self.galaxy_linker.particle_positions = np.array([
            [ 0., 0., 0. ],
            [ 0., 0., 0. ],
            ])

        expected = np.array( [-2, -2] )
        actual = self.galaxy_linker.find_host_id()

        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_find_mt_halo_id( self ):

        self.galaxy_linker.particle_positions = np.array([
            [ 29414.96458784,  30856.75007114,  32325.90901812], # Right in the middle of mt halo 0 at snap 500
            [ 29467.07226789,  30788.6179313 ,  32371.38749237], # Right in the middle of mt halo 9 at snap 500.
                                                                                                                      # mt halo 9 is 0.5 Rvir_mt_0 (2 Rvir_mt_9) away from the center of mt halo 0
            [ 29073.22333685,  31847.72434505,  32283.53620817], # Right in the middle of mt halo 19 at snap 500.
            [             0.,              0.,              0.], # The middle of nowhere.
            ])
        self.galaxy_linker.particle_positions *= 1./(1. + self.redshift)/self.hubble
        self.galaxy_linker.n_particles = 4

        actual = self.galaxy_linker.find_halo_id( 2.5, 'mt_halo_id' )

        # Build the expected output
        expected = np.array([ 0, 0, 19, -2 ])

        npt.assert_allclose( actual, expected )

    ########################################################################

    def test_find_mt_halo_id_early_universe( self ):
        '''Test that, when there are no galaxies formed, we return an mt halo value of -2'''

        # Set it to early redshifts
        self.galaxy_linker.snum = 0

        # It doesn't really matter where the particles are, because there shouldn't be any galaxies anyways....
        self.galaxy_linker.particle_positions = np.array([
            [ 29414.96458784,  30856.75007114,  32325.90901812], # Right in the middle of mt halo 0 at snap 500
            [ 29467.07226789,  30788.6179313 ,  32371.38749237], # Right in the middle of mt halo 9 at snap 500.
                                                                                                                      # mt halo 9 is 0.5 Rvir_mt_0 (2 Rvir_mt_9) away from the center of mt halo 0
            [ 29073.22333685,  31847.72434505,  32283.53620817], # Right in the middle of mt halo 19 at snap 500.
            [             0.,              0.,              0.], # The middle of nowhere.
            ])
        self.galaxy_linker.particle_positions *= 1./(1. + 30.)/self.hubble
        self.galaxy_linker.n_particles = 4

        actual = self.galaxy_linker.find_halo_id( 2.5, 'mt_halo_id' )

        # Build the expected output
        expected = np.array([ -2, -2, -2, -2 ])

        npt.assert_allclose( actual, expected )

    ########################################################################

    def test_extract_additional_keys( self ):

        self.galaxy_linker.particle_positions = np.array([
            [ 29414.96458784,  30856.75007114,  32325.90901812],
            [ 31926.42103071,  51444.46756529,   1970.1967437 ],
            [ 29467.07226789,  30788.6179313 ,  32371.38749237],
            [ 29459.32290246,  30768.32556725,  32357.26078864], # Halo 3783, host halo 3610
            ])
        self.galaxy_linker.particle_positions *= 1./(1. + self.redshift)/self.hubble

        self.galaxy_linker.n_particles = 4
        self.galaxy_linker.supplementary_data_keys = [ 'Xc', 'Yc', 'Zc' ]

        expected_id, expected = (
            np.array( [0, 6962, 7, 3783] ),
            {
                'Xc': np.array([ 29414.96458784, 31926.42103071, 29467.07226789, 29459.32290246, ]),
                'Yc': np.array([ 30856.75007114, 51444.46756529, 30788.6179313 , 30768.32556725, ]),
                'Zc': np.array([ 32325.90901812, 1970.1967437, 32371.38749237, 32357.26078864, ]),
            },
        )

        actual_id, actual = self.galaxy_linker.find_halo_id( supplementary_data=True )

        assert len( expected.keys() ) == len( actual.keys() )
        for key in expected.keys():
            npt.assert_allclose( expected[key], actual[key] )
        npt.assert_allclose( expected_id, actual_id )

    ########################################################################

    def test_find_ids( self ):

        particle_positions = np.array([
            [ 29414.96458784,  30856.75007114,  32325.90901812], # Halo 0, host halo 0
            [ 30068.5541178 ,  32596.72758226,  32928.1115097 ], # Halo 10, host halo 1
            [ 29459.32290246,  30768.32556725,  32357.26078864], # Halo 3783, host halo 3610
            ])
        particle_positions *= 1./(1. + self.redshift)/self.hubble

        # Move one just off to the side to help with testing
        particle_positions[0] += 5.

        used_kwargs = copy.deepcopy( self.kwargs )

        used_kwargs['ids_to_return'].append( '0.00000001_Rvir' )
        used_kwargs['ids_with_supplementary_data'] = [ 'gal_id', 'd_gal' ]
        used_kwargs['supplementary_data_keys'] = [ 'Xc', 'Yc', 'Zc' ]

        expected = {
            'd_gal': np.array( [ 5.*np.sqrt( 3 ), 0., 0., ] ),
            'host_halo_id': np.array( [-1, 1, 3610] ),
            'halo_id': np.array( [0, 10, 3783] ),
            'host_gal_id': np.array( [-1, 1, 3610] ),
            'gal_id': np.array( [0, 10, 3783] ),
            'mt_gal_id': np.array( [0, -2, -2] ),
            'mt_halo_id': np.array( [0, 1, 0] ),
            '0.00000001_Rvir': np.array( [-2, 10, 3783] ),
            'gal_id_Xc': np.array([ 29414.96458784, 30068.5541178, 29459.32290246, ]),
            'gal_id_Yc': np.array([ 30856.75007114, 32596.72758226, 30768.32556725, ]),
            'gal_id_Zc': np.array([ 32325.90901812, 32928.1115097, 32357.26078864, ]),
            'd_gal_Xc': np.array([ 29414.96458784, 30068.5541178, 29459.32290246, ]),
            'd_gal_Yc': np.array([ 30856.75007114, 32596.72758226, 30768.32556725, ]),
            'd_gal_Zc': np.array([ 32325.90901812, 32928.1115097, 32357.26078864, ]),
        }

        # Do the actual calculation
        galaxy_linker = general_galaxy_linker.GalaxyLinker(
            particle_positions,
            **used_kwargs
        )
        actual = galaxy_linker.find_ids()

        for key in expected.keys():
            try:
                npt.assert_allclose( expected[key], actual[key], atol=1e-10 )
            except AssertionError:
                raise AssertionError( 'key = {}, expected = {}, actual = {}'.format(
                        key,
                        expected[key],
                        actual[key],
                    )
                )

    ########################################################################

    def test_find_ids_different_mt_scale( self ):

        particle_positions = np.array([
            [ 29414.96458784,  30856.75007114,  32325.90901812], # Halo 0, host halo 0
            [ 30068.5541178 ,  32596.72758226,  32928.1115097 ], # Halo 10, host halo 1
            [ 29459.32290246,  30768.32556725,  32357.26078864], # Halo 3783, host halo 3610
            ])
        # Shift the first particle over slightly to put it outside Rmax, which
        # we'll use as our length scale.
        particle_positions[0][0] += 4.
        particle_positions *= 1./(1. + self.redshift)/self.hubble

        expected = {
            'd_gal': np.array( [ 4., 0., 0., ] )/(1. + self.redshift)/self.hubble,
            'host_gal_id': np.array( [-1, 1, 3610] ),
            'gal_id': np.array( [0, 10, 3783] ),
            'mt_gal_id': np.array( [-2, -2, -2] ),
        }

        kwargs = copy.deepcopy( self.kwargs )
        kwargs['mt_length_scale'] = 'Rmax'
        kwargs['galaxy_cut'] = 0.1
        kwargs['ids_to_return'] = [
            'd_gal',
            'd_other_gal_scaled',
            'host_gal_id',
            'gal_id',
            'mt_gal_id',
        ]

        # Do the actual calculation
        galaxy_linker = general_galaxy_linker.GalaxyLinker(
            particle_positions,
            **kwargs
        )
        actual = galaxy_linker.find_ids()

        for key in expected.keys():
            print(key)
            npt.assert_allclose( expected[key], actual[key], atol=1e-10 )

    ########################################################################

    def test_find_ids_rockstar( self ):
        '''Test that this works for Rockstar as well.'''

        # Update the arguments
        used_kwargs = copy.copy( self.kwargs )
        used_kwargs['ids_to_return'] = [
            'd_gal',
            'gal_id',
        ]
        used_kwargs['halo_data_dir'] = './tests/data/rockstar_dir'
        used_kwargs['halo_finder'] = 'Rockstar'
        used_kwargs['minimum_criteria'] = 'Np'
        used_kwargs['minimum_value'] = 10
        used_kwargs['length_scale'] = 'Rs'
        used_kwargs['halo_length_scale'] = 'R200b'

        particle_positions = np.array([
            [ 29534.48, 29714.94, 32209.94, ], # Halo 6811
            [ 28495.69, 30439.15, 31563.17, ], # Halo 7339
            [ 28838.19, 30631.51, 32055.31, ], # Halo 9498
            ])
        particle_positions *= 1./(1. + self.redshift)/self.hubble

        expected = {
            'd_gal': np.array( [ 0., 0., 0., ] ),
            'gal_id': np.array( [ 6811, 7339, 9498 ] ),
        }

        # Do the actual calculation
        galaxy_linker = general_galaxy_linker.GalaxyLinker(
            particle_positions,
            **used_kwargs
        )
        actual = galaxy_linker.find_ids()

        for key in expected.keys():
            print(key)
            npt.assert_allclose( expected[key], actual[key], atol=1e-10 )

    ########################################################################

    def test_find_ids_snap0( self ):

        particle_positions = np.array([
            [ 29414.96458784,  30856.75007114,  32325.90901812], # Halo 0, host halo 0
            [ 30068.5541178 ,  32596.72758226,  32928.1115097 ], # Halo 10, host halo 1
            [ 29459.32290246,  30768.32556725,  32357.26078864], # Halo 3783, host halo 3610
            ])
        particle_positions *= 1./(1. + 30.)/self.hubble

        expected = {
            'host_halo_id': np.array( [-2, -2, -2] ),
            'halo_id': np.array( [-2, -2, -2] ),
            'host_gal_id': np.array( [-2, -2, -2] ),
            'gal_id': np.array( [-2, -2, -2] ),
            'mt_gal_id': np.array( [-2, -2, -2] ),
            'mt_halo_id': np.array( [-2, -2, -2] ),
        }

        # Setup the input parameters
        snap0_kwargs = copy.deepcopy( self.kwargs )
        snap0_kwargs['snum'] = 0

        # Do the actual calculation
        galaxy_linker = general_galaxy_linker.GalaxyLinker( particle_positions, **snap0_kwargs )
        actual = galaxy_linker.find_ids()

        for key in expected.keys():
            print(key)
            npt.assert_allclose( expected[key], actual[key] )

    ########################################################################

    def test_find_ids_early_universe( self ):

        particle_positions = np.array([
            [ 29414.96458784,  30856.75007114,  32325.90901812], # Halo 0, host halo 0
            [ 30068.5541178 ,  32596.72758226,  32928.1115097 ], # Halo 10, host halo 1
            [ 29459.32290246,  30768.32556725,  32357.26078864], # Halo 3783, host halo 3610
            ])
        particle_positions *= 1./(1. + 28.)/self.hubble

        expected = {
            'host_halo_id': np.array( [-2, -2, -2] ),
            'halo_id': np.array( [-2, -2, -2] ),
            'host_gal_id': np.array( [-2, -2, -2] ),
            'gal_id': np.array( [-2, -2, -2] ),
            'mt_gal_id': np.array( [-2, -2, -2] ),
            'mt_halo_id': np.array( [-2, -2, -2] ),
            '0.1_Rvir': np.array( [-2, -2, -2] ),
        }

        # Setup the input parameters
        snap0_kwargs = copy.deepcopy( self.kwargs )
        snap0_kwargs['snum'] = 1

        # Do the actual calculation
        galaxy_linker = general_galaxy_linker.GalaxyLinker( particle_positions, **snap0_kwargs )
        actual = galaxy_linker.find_ids()

        for key in expected.keys():
            print(key)
            npt.assert_allclose( expected[key], actual[key] )

    ########################################################################

    def test_pass_halo_data( self ):
        '''Test that it still works when we pass in an halo_data. '''

        particle_positions = np.array([
            [ 29414.96458784,  30856.75007114,  32325.90901812], # Halo 0, host halo 0
            [ 30068.5541178 ,  32596.72758226,  32928.1115097 ], # Halo 10, host halo 1
            [ 29459.32290246,  30768.32556725,  32357.26078864], # Halo 3783, host halo 3610
            ])
        particle_positions *= 1./(1. + self.redshift)/self.hubble

        expected = {
            'host_halo_id': np.array( [-1, 1, 3610] ),
            'halo_id': np.array( [0, 10, 3783] ),
            'host_gal_id': np.array( [-1, 1, 3610] ),
            'gal_id': np.array( [0, 10, 3783] ),
            'mt_gal_id': np.array( [0, -2, -2] ),
            'mt_halo_id': np.array( [0, 1, 0] ),
        }

        # Prepare an halo_data to pass along.
        halo_data = analyze_halos.HaloData(
            self.kwargs['halo_data_dir'],
        )

        # Muck it up by making it try to retrieve data
        halo_data.data_reader.get_halos( 600 )
        halo_data.data_reader.get_mtree_halos( 600, tag='smooth' )

        # Do the actual calculation
        galaxy_linker = general_galaxy_linker.GalaxyLinker( particle_positions, halo_data=halo_data, **self.kwargs )
        actual = galaxy_linker.find_ids()

        for key in expected.keys():
            print(key)
            npt.assert_allclose( expected[key], actual[key] )

    ########################################################################

    def test_find_d_gal( self ):
        '''This tests we can find the shortest distance to the nearest galaxy.
        '''

        # Setup the distance so we don't have to calculate it.
        self.galaxy_linker._dist_to_all_valid_halos = np.array([
            [ 0.5, 1.0, 0.5, ],
            [ 15., 5., 3., ],
            [ 0.2, 2.5e-4, 4., ],
        ])

        actual = self.galaxy_linker.find_d_gal()

        expected = np.array([ 0.5, 3., 2.5e-4, ])

        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_find_d_other_gal( self ):
        '''This tests we can find the shortest distance to the nearest galaxy.
        '''

        # Setup the distance so we don't have to calculate it.
        self.galaxy_linker._dist_to_all_valid_halos = np.array([
            [ 0.5, 1.0, 0.75, ],
            [ 15., 5., 3., ],
            [ 0.2, 2.5e-4, 4., ],
        ])

        actual = self.galaxy_linker.find_d_other_gal()

        expected = np.array([ 0.75, 3., 2.5e-4, ])

        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_find_d_other_gal_main_halo_id_not_0( self ):
        '''This tests we can find the shortest distance to the nearest galaxy.
        '''

        self.galaxy_linker.main_mt_halo_id = 1

        # Setup the distance so we don't have to calculate it.
        self.galaxy_linker._dist_to_all_valid_halos = np.array([
            [ 0.5, 1.0, 0.75, ],
            [ 15., 5., 3., ],
            [ 0.2, 2.5e-4, 4., ],
        ])

        actual = self.galaxy_linker.find_d_other_gal()

        expected = np.array([ 0.5, 3., 0.2, ])

        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_find_d_other_gal_early_universe( self ):
        '''This tests we can find the shortest distance to the nearest galaxy.
        '''

        self.galaxy_linker.snum = 1

        # Setup the distance so we don't have to calculate it.
        self.galaxy_linker._dist_to_all_valid_halos = np.array([
            [ 0.5, 1.0, 0.75, ],
            [ 15., 5., 3., ],
            [ 0.2, 2.5e-4, 4., ],
        ])

        actual = self.galaxy_linker.find_d_other_gal()

        expected = np.array([ 0.5, 3., 2.5e-4, ])

        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_find_d_other_gal_scaled( self ):
        '''This tests we can find the shortest distance to the nearest galaxy.
        '''

        # Setup dummy data
        self.galaxy_linker._ahf_halos_length_scale_pkpc = np.array([ 1., 2., 3., 4., 5., ])
        self.galaxy_linker._valid_halo_inds = np.array([ 0, 1, 2, 3, ])
        self.galaxy_linker._dist_to_all_valid_halos = np.array([
            [ 2., 4., 6., 8. ],
            [ 4., 3., 2., 1., ],
            [ 10., 8., 6., 7., ],
        ])

        # Make sure we set the number of particles correctly, to match the number we're using
        self.galaxy_linker.n_particles = 3

        actual = self.galaxy_linker.find_d_other_gal( scaled=True )

        expected = np.array([ 2., 0.25, 2., ])

        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_find_d_other_gal_scaled_early_universe( self ):
        '''This tests we can find the shortest distance to the nearest galaxy.
        '''

        # Setup dummy data
        self.galaxy_linker.snum = 1
        self.galaxy_linker._ahf_halos_length_scale_pkpc = np.array([ 1., 2., 3., 4., 5., ])
        self.galaxy_linker._valid_halo_inds = np.array([ 0, 1, 2, 3, ])
        self.galaxy_linker._dist_to_all_valid_halos = np.array([
            [ 2., 4., 6., 8. ],
            [ 4., 3., 2., 1., ],
            [ 10., 8., 6., 7., ],
        ])

        # Make sure we set the number of particles correctly, to match the number we're using
        self.galaxy_linker.n_particles = 3

        actual = self.galaxy_linker.find_d_other_gal( scaled=True )

        expected = np.array([ 2., 0.25, 2., ])

        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_find_d_other_gal_only_valid_halo_is_main( self ):
        '''Test that things still work when there are other halos, but the
        only valid halo is the main halo.
        '''

        # Setup dummy data
        self.galaxy_linker.minimum_value = 10
        self.galaxy_linker.snum = 10
        self.galaxy_linker.halo_data.data_reader.data_dir = \
            './tests/data/analysis_dir5'
        self.galaxy_linker.halo_data.data_reader.get_halos( 10 )

        # Make sure we set the number of particles correctly, to match the number we're using
        #self.galaxy_linker.n_particles = 3

        actual = self.galaxy_linker.find_d_other_gal()

        expected = np.array([ -2., -2., ])

        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_find_d_other_gal_scaled_no_halos( self ):
        '''This tests we can find the shortest distance to the nearest galaxy.
        '''

        # Setup dummy data
        self.galaxy_linker.snum = 1
        self.galaxy_linker.halo_data.data_reader.data_dir = './tests/data/analysis_dir4'
        self.galaxy_linker.halo_data.data_reader.get_halos( 1 )

        # Make sure we set the number of particles correctly, to match the number we're using
        #self.galaxy_linker.n_particles = 3

        actual = self.galaxy_linker.find_d_other_gal( scaled=True )

        expected = np.array([ -2., -2., ])

        npt.assert_allclose( expected, actual )

    ########################################################################

    @mock.patch( 'galaxy_dive.read_data.ahf.AHFReader.get_halos_add' )
    def test_find_d_other_gal_scaled_no_halos_with_sufficient_mass( self, mock_get_halos_add ):
        '''This tests we can find the shortest distance to the nearest galaxy.
        '''

        # Setup dummy data
        self.galaxy_linker.snum = 12
        self.galaxy_linker.minimum_value = 10
        self.galaxy_linker.halo_data.data_reader.data_dir = './tests/data/analysis_dir4'
        self.galaxy_linker.halo_data.data_reader.get_halos( 12 )

        # Make sure we set the number of particles correctly, to match the number we're using
        #self.galaxy_linker.n_particles = 3

        actual = self.galaxy_linker.find_d_other_gal( scaled=True )

        expected = np.array([ -2., -2., ])

        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_find_d_other_gal_scaled_main_halo_id_not_0( self ):
        '''This tests we can find the shortest distance to the nearest galaxy.
        '''

        # Setup dummy data
        self.galaxy_linker.main_mt_halo_id = 3
        self.galaxy_linker._ahf_halos_length_scale_pkpc = np.array([ 1., 2., 3., 4., 5., ])
        self.galaxy_linker._valid_halo_inds = np.array([ 0, 1, 2, 3, ])
        self.galaxy_linker._dist_to_all_valid_halos = np.array([
            [ 2., 4., 6., 8. ],
            [ 4., 3., 2., 1., ],
            [ 10., 8., 6., 7., ],
        ])

        # Make sure we set the number of particles correctly, to match the number we're using
        self.galaxy_linker.n_particles = 3

        actual = self.galaxy_linker.find_d_other_gal( scaled=True )

        expected = np.array([ 2., 2./3., 2., ])

        npt.assert_allclose( expected, actual )

########################################################################


class TestGalaxyLinkerMinimumStellarMass( unittest.TestCase ):
    '''Test that we're properly applying a minimum stellar mass for a halo to be counted as containing a galaxy.'''

    def setUp( self ):

        gal_linker_kwargs_min_mstar = {
            'length_scale': 'r_scale',
            'minimum_criteria': 'M_star',
            'minimum_value': 1e6,

            'redshift': 6.1627907799999999,
            'snum': 50,
            'hubble': 0.70199999999999996,
            'halo_data_dir': './tests/data/analysis_dir3',
            'mtree_halos_index': 600,
            'halo_file_tag': 'smooth',
            'main_mt_halo_id': 0,
            'galaxy_cut': 0.1,
            'length_scale': 'Rvir',

            'ids_to_return': [ 'halo_id', 'host_halo_id', 'gal_id', 'host_gal_id', 'mt_halo_id', 'mt_gal_id', 'd_gal', 'd_other_gal_scaled', ],
        }

        # Get input data
        comoving_particle_positions = np.array([
            [ 30252.60118534,  29483.51635481,  31011.17715464], # Right in the middle of mt halo 0 (AHF halo id 3) at snum 50.
                                                                                                                      # This halo contains a galaxy with 1e6.7 Msun of stars at this redshift.
            [ 28651.1193359,  29938.7253038,  32168.1380575], # Right in the middle of mt halo 19 (AHF halo id 374) at snum 50
                                                                                                                      # This halo no stars at this redshift.
        ])

        self.redshift = gal_linker_kwargs_min_mstar['redshift']
        self.hubble = gal_linker_kwargs_min_mstar['hubble']
        particle_positions = comoving_particle_positions/(1. + self.redshift)/self.hubble

        # Make the necessary kwargs
        self.kwargs = gal_linker_kwargs_min_mstar

        self.galaxy_linker = general_galaxy_linker.GalaxyLinker( particle_positions, **self.kwargs )

        # Get the necessary reader.
        self.galaxy_linker.halo_data.data_reader = read_ahf.AHFReader( self.kwargs['halo_data_dir'] )

        # Get the full needed ahf info.
        self.galaxy_linker.halo_data.data_reader.get_halos( 50 )

    ########################################################################

    def test_find_containing_halos( self ):

        actual = self.galaxy_linker.find_containing_halos( 1. )

        # Build the expected output
        n_halos = self.galaxy_linker.halo_data.data_reader.halos.index.size
        expected = np.zeros( (self.galaxy_linker.particle_positions.shape[0], n_halos) ).astype( bool )
        expected[ 0, 3 ] = True # Should only be in the galaxy with sufficient stellar mass.

        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_find_mt_containing_halos( self ):

        actual = self.galaxy_linker.find_mt_containing_halos( 1. )

        # Build the expected output
        n_halos = len( self.galaxy_linker.halo_data.data_reader.mtree_halos )
        expected = np.zeros( (self.galaxy_linker.particle_positions.shape[0], n_halos) ).astype( bool )
        expected[ 0, 0 ] = True # Should only be in the galaxy with sufficient stellar gas.

        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_find_ids_custom( self ):

        self.galaxy_linker.ids_to_return = [
            'gal_id',
            'halo_id',
            '0.1_Rvir',
            '1.0_Rvir',
            '5_r_scale',
        ]

        results = self.galaxy_linker.find_ids()

        npt.assert_allclose( results['gal_id'], results['0.1_Rvir'] )
        npt.assert_allclose( results['halo_id'], results['1.0_Rvir'] )

########################################################################
########################################################################

class TestGalaxyLinkerMinimumNumStars( unittest.TestCase ):
    '''Test that we're properly applying a minimum number of stars for a halo to be counted as containing a galaxy.'''

    def setUp( self ):

        gal_linker_kwargs_min_nstar = {
            'minimum_criteria': 'n_star',
            'minimum_value': 10,

            'redshift': 6.1627907799999999,
            'snum': 50,
            'hubble': 0.70199999999999996,
            'halo_data_dir': './tests/data/analysis_dir3',
            'mtree_halos_index': 600,
            'halo_file_tag': 'smooth',
            'main_mt_halo_id': 0,
            'galaxy_cut': 0.1,
            'length_scale': 'Rvir',

            'ids_to_return': [ 'halo_id', 'host_halo_id', 'gal_id', 'host_gal_id', 'mt_halo_id', 'mt_gal_id', 'd_gal', 'd_other_gal_scaled', ],
        }

        # Get input data
        comoving_particle_positions = np.array([
            [ 30252.60118534,  29483.51635481,  31011.17715464], # Right in the middle of mt halo 0 (AHF halo id 3) at snum 50.
                                                                                                                      # This halo contains a galaxy with 1e6.7 Msun of stars at this redshift.
            [ 28651.1193359,  29938.7253038,  32168.1380575], # Right in the middle of mt halo 19 (AHF halo id 374) at snum 50
                                                                                                                      # This halo no stars at this redshift.
        ])

        self.redshift = gal_linker_kwargs_min_nstar['redshift']
        self.hubble = gal_linker_kwargs_min_nstar['hubble']
        particle_positions = comoving_particle_positions/(1. + self.redshift)/self.hubble

        # Make the necessary kwargs
        self.kwargs = gal_linker_kwargs_min_nstar

        self.galaxy_linker = general_galaxy_linker.GalaxyLinker( particle_positions, **self.kwargs )

        # Get the necessary reader.
        self.galaxy_linker.halo_data.data_reader = read_ahf.AHFReader( self.kwargs['halo_data_dir'] )

        # Get the full needed ahf info.
        self.galaxy_linker.halo_data.data_reader.get_halos( 50 )

    ########################################################################

    def test_find_containing_halos( self ):

        actual = self.galaxy_linker.find_containing_halos( 1. )

        # Build the expected output
        n_halos = self.galaxy_linker.halo_data.data_reader.halos.index.size
        expected = np.zeros( (self.galaxy_linker.particle_positions.shape[0], n_halos) ).astype( bool )
        expected[ 0, 3 ] = True # Should only be in the galaxy with sufficient stellar mass.

        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_find_mt_containing_halos( self ):

        actual = self.galaxy_linker.find_mt_containing_halos( 1. )

        # Build the expected output
        n_halos = len( self.galaxy_linker.halo_data.data_reader.mtree_halos )
        expected = np.zeros( (self.galaxy_linker.particle_positions.shape[0], n_halos) ).astype( bool )
        expected[ 0, 0 ] = True # Should only be in the galaxy with sufficient stellar gas.

        npt.assert_allclose( expected, actual )

########################################################################
########################################################################

class TestFindMassRadii( unittest.TestCase ):

    def setUp( self ):

        # Test Data
        self.kwargs = dict( gal_linker_kwargs )
        self.kwargs['particle_masses'] = np.array([ 1., 2., 3., 4., ])
        particle_positions = np.array([
            [ 0., 0., 0., ],
            [ 0., 0., 0., ],
            [ 0., 0., 0., ],
            [ 0., 0., 0., ],
        ]) # These shouldn't ever be used directly, since we're relying on the results of previous functions.

        self.galaxy_linker = general_galaxy_linker.GalaxyLinker( particle_positions, **self.kwargs )

    ########################################################################

    def test_mass_inside_galaxy_cut( self ):

        # Test Data
        self.galaxy_linker._ahf_halos_length_scale_pkpc = np.array([ 200., 10., 100., 50., ])
        self.galaxy_linker._valid_halo_inds = np.array([ 0, 1, 2, ])
        self.galaxy_linker._dist_to_all_valid_halos = np.array([
            [ 0., 10., 500., ],
            [ 15., 5., 485., ],
            [ 10., 0., 490., ],
            [ 500., 490., 0., ],
        ])

        expected = np.array([ 6., 3., 4., ])
        actual = self.galaxy_linker.mass_inside_galaxy_cut

        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_mass_inside_galaxy_cut_no_valid_inds( self ):
        '''Test we still get a reasonable result out, even when there's not a single valid halo.'''

        # Test Data
        self.galaxy_linker._ahf_halos_length_scale_pkpc = np.array([ 200., 10., 100., 50., ])
        self.galaxy_linker._valid_halo_inds = np.array([])
        self.galaxy_linker._dist_to_all_valid_halos = np.array([
            [ 0., 10., 500., ],
            [ 15., 5., 485., ],
            [ 10., 0., 490., ],
            [ 500., 490., 0., ],
        ])

        actual = self.galaxy_linker.mass_inside_galaxy_cut
        expected = np.array([])

        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_mass_inside_galaxy_cut_no_halos( self ):
        '''Test we still get a reasonable result out, even when there aren't any halos formed yet.'''

        # Test Data
        self.galaxy_linker._ahf_halos_length_scale_pkpc = np.array([])
        self.galaxy_linker._valid_halo_inds = np.array([])

        actual = self.galaxy_linker.mass_inside_galaxy_cut
        expected = np.array([])

        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_mass_inside_galaxy_cut_no_inside_cut( self ):
        '''Make sure we give the right results when no particles are inside the cut.'''

        # Test Data
        self.galaxy_linker._ahf_halos_length_scale_pkpc = np.array([ 0.1, 0.1, 0.1, 0.1, ])
        self.galaxy_linker._valid_halo_inds = np.array([ 0, 1, 2, ])
        self.galaxy_linker._dist_to_all_valid_halos = np.array([
            [ 100., 10., 500., ],
            [ 15., 5., 485., ],
            [ 10., 100., 490., ],
            [ 500., 490., 100., ],
        ])

        expected = np.array([ 0., 0., 0., ])
        actual = self.galaxy_linker.mass_inside_galaxy_cut

        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_mass_inside_all_halos( self ):

        # Test Data
        self.galaxy_linker._ahf_halos_length_scale_pkpc = np.array([ 200., 10., 100., 50., np.nan, ])
        self.galaxy_linker._valid_halo_inds = np.array([ 0, 1, 2, 4, ])
        self.galaxy_linker._dist_to_all_valid_halos = np.array([
            [ 0., 10., 500., np.nan, ],
            [ 15., 5., 485., np.nan, ],
            [ 10., 0., 490., np.nan, ],
            [ 500., 490., 0., np.nan, ],
        ])

        expected = np.array([ 6., 3., 4., np.nan, np.nan ])
        actual = self.galaxy_linker.mass_inside_all_halos

        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_mass_inside_all_halos_no_valid_gals( self ):

        # Test Data
        self.galaxy_linker._ahf_halos_length_scale_pkpc = np.array([ 200., 10., 100., 50., np.nan, ])
        self.galaxy_linker._valid_halo_inds = np.array([])

        expected = np.array([ np.nan, np.nan, np.nan, np.nan, np.nan ])
        actual = self.galaxy_linker.mass_inside_all_halos

        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_cumlulative_mass_valid_halos( self ):

        # Test Data
        self.galaxy_linker._dist_to_all_valid_halos = np.array([
            [ 0., 500., ],
            [ 480., 450., ],
            [ 50., 20., ],
            [ 490., 10., ],
        ])

        expected = np.array([
            [ 1., 10., ],
            [ 6., 9., ],
            [ 4., 7., ],
            [ 10., 4., ],
        ])
        actual = self.galaxy_linker.cumulative_mass_valid_halos

        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_get_mass_radius( self ):

        # Test Data
        self.galaxy_linker._ahf_halos_length_scale_pkpc = np.array([ 0., 0., 0. ]) # Values shouldn't matter here
        self.galaxy_linker._valid_halo_inds = np.array([ 0, 1, ])
        self.galaxy_linker._mass_inside_galaxy_cut = np.array([ 10., 19, ])
        self.galaxy_linker._dist_to_all_valid_halos = np.array([
            [ 0., 500., ],
            [ 480., 450., ],
            [ 50., 20., ],
            [ 490., 10., ],
        ])
        self.galaxy_linker._cumulative_mass_valid_halos = np.array([
            [ 1., 10., ],
            [ 6., 9., ],
            [ 4., 7., ],
            [ 10., 4., ],
        ]),

        # Expected result, in comoving coords.
        expected = np.array( [ 50., 450., np.nan ] )*( 1. + 0.16946003 )*0.702
        actual = self.galaxy_linker.get_mass_radius( 0.5 )

        npt.assert_allclose( expected, actual )

########################################################################
########################################################################

class TestSummedQuantityInsideGalaxy( unittest.TestCase ):
    '''Test that we can calculate a more general attribute inside a galaxy.'''

    def setUp( self ):

        # Test Data
        self.kwargs = dict( gal_linker_kwargs )
        self.kwargs['particle_masses'] = np.array([ 2., 2., 1., 3., ])
        particle_positions = np.array([
            [ 0., 0., 0., ],
            [ 0., 0., 0., ],
            [ 0., 0., 0., ],
            [ 0., 0., 0., ],
        ]) # These shouldn't ever be used directly, since we're relying on the results of previous functions.

        self.galaxy_linker = general_galaxy_linker.GalaxyLinker( particle_positions, **self.kwargs )

    ########################################################################

    def test_summed_quantity_inside_galaxy_valid_halos( self ):

        # Test Data
        self.galaxy_linker._ahf_halos_length_scale_pkpc = np.array([ 200., 10., 100., 50., ])
        self.galaxy_linker._valid_halo_inds = np.array([ 0, 1, 2, ])
        self.galaxy_linker._dist_to_all_valid_halos = np.array([
            [ 0., 10., 500., ],
            [ 15., 5., 485., ],
            [ 10., 0., 490., ],
            [ 500., 490., 0., ],
        ])
        particle_quantities = np.array([ 1., 2., 3., 4., ])

        actual = self.galaxy_linker.summed_quantity_inside_galaxy_valid_halos( particle_quantities, np.nan )
        expected = np.array([ 6., 3., 4., ])

        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_summed_quantity_valid_halos_no_inside_cut( self ):
        '''Make sure we give the right results when no particles are inside the cut.'''

        # Test Data
        self.galaxy_linker._ahf_halos_length_scale_pkpc = np.array([ 0.1, 0.1, 0.1, 0.1, ])
        self.galaxy_linker._valid_halo_inds = np.array([ 0, 1, 2, ])
        self.galaxy_linker._dist_to_all_valid_halos = np.array([
            [ 100., 10., 500., ],
            [ 15., 5., 485., ],
            [ 10., 100., 490., ],
            [ 500., 490., 100., ],
        ])
        particle_quantities = np.array([ 1., 2., 3., 4., ])

        expected = np.array([ np.nan, np.nan, np.nan, ])
        actual = self.galaxy_linker.summed_quantity_inside_galaxy_valid_halos( particle_quantities, np.nan )

        npt.assert_allclose( expected, actual )

    ########################################################################

    @mock.patch( 'galaxy_dive.galaxy_linker.linker.GalaxyLinker.dist_to_all_valid_halos_fn' )
    def test_summed_quantity_inside_galaxy_low_memory_mode( self, mock_dist_all_valid ):
        '''Test that we can get the summed quantity inside the galaxy, but reduce the memory consumption
        (at the cost of speed) by doing getting the sum for less galaxies at a given time.
        '''

        # Test Data
        self.galaxy_linker._ahf_halos_length_scale_pkpc = np.array([ 200., 10., 100., 50., ])
        self.galaxy_linker._valid_halo_inds = np.array([ 0, 1, 2, ])
        mock_dist_all_valid.side_effect = [
            np.array( [ [ 0., 10., 500., ], ] ),
            np.array( [ [ 15., 5., 485., ], ] ),
            np.array( [ [ 10., 0., 490., ], ] ),
            np.array( [ [ 500., 490., 0., ], ] ),
            np.array( [] ),
            np.array( [] ),
            np.array( [] ),
            np.array( [] ),
            np.array( [] ),
            np.array( [] ),
        ]
        particle_quantities = np.array([ 1., 2., 3., 4., ])

        # Change parameters of galaxy finder to run low memory node.
        self.galaxy_linker.low_memory_mode = True

        actual = self.galaxy_linker.summed_quantity_inside_galaxy_valid_halos( particle_quantities )
        expected = np.array([ 6., 3., 4., ])

        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_summed_quantity_inside_galaxy( self ):

        # Test Data
        self.galaxy_linker._ahf_halos_length_scale_pkpc = np.array([ 200., 10., 100., 50., np.nan, ])
        self.galaxy_linker._valid_halo_inds = np.array([ 0, 1, 2, 4, ])
        self.galaxy_linker._dist_to_all_valid_halos = np.array([
            [ 0., 10., 500., np.nan, ],
            [ 15., 5., 485., np.nan, ],
            [ 10., 0., 490., np.nan, ],
            [ 500., 490., 0., np.nan, ],
        ])
        particle_quantities = np.array([ 1., 2., 3., 4., ])

        expected = np.array([ 6., 3., 4., np.nan, np.nan ])
        actual = self.galaxy_linker.summed_quantity_inside_galaxy(
            particle_quantities,
            np.nan,
        )

        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_weighted_summed_quantity_inside_galaxy( self ):

        # Test Data
        self.galaxy_linker._ahf_halos_length_scale_pkpc = np.array([ 200., 10., 100., 50., np.nan, ])
        self.galaxy_linker._valid_halo_inds = np.array([ 0, 1, 2, 4, ])
        self.galaxy_linker._dist_to_all_valid_halos = np.array([
            [ 0., 10., 500., np.nan, ],
            [ 15., 5., 485., np.nan, ],
            [ 10., 0., 490., np.nan, ],
            [ 500., 490., 0., np.nan, ],
        ])
        particle_quantities = np.array([ 1., 2., 3., 4., ])
        particle_weights = np.array([ 4., 3., 2., 1., ])

        actual = self.galaxy_linker.weighted_summed_quantity_inside_galaxy( particle_quantities, particle_weights, np.nan )
        expected = np.array([ 16./9., 3., 4., np.nan, np.nan, ])

        npt.assert_allclose( expected, actual )






