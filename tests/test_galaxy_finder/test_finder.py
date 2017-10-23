#!/usr/bin/env python
'''Testing for tracking.py

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import copy
import h5py
import mock
import numpy as np
import numpy.testing as npt
import os
import pdb
import pytest
import unittest

import galaxy_diver.read_data.ahf as read_ahf
import galaxy_diver.galaxy_finder.finder as general_galaxy_finder

########################################################################
# Useful global test variables
########################################################################

gal_finder_kwargs = {
  'length_scale' : 'Rvir',

  'redshift' : 0.16946003,
  'snum' : 500,
  'hubble' : 0.70199999999999996,
  'ahf_data_dir' : './tests/data/analysis_dir3',
  'mtree_halos_index' : 600,
  'main_mt_halo_id' : 0,
  'halo_file_tag' : 'smooth',

  'galaxy_cut' : 0.1,
  'ids_to_return' : [ 'halo_id', 'host_halo_id', 'gal_id', 'host_gal_id', 'mt_halo_id', 'mt_gal_id', 'd_gal', 'd_other_gal_scaled', ],
  'minimum_criteria' : 'n_star',
  'minimum_value' : 0,
}

########################################################################

# Decorator for skipping slow tests
slow = pytest.mark.skipif(
    not pytest.config.getoption("--runslow"),
    reason="need --runslow option to run"
)

########################################################################

class TestGalaxyFinder( unittest.TestCase ):

  def setUp( self ):

    # Get input data
    comoving_halo_coords = np.array([ [ 29414.96458784,  30856.75007114,  32325.90901812],
                                      [ 31926.42103071,  51444.46756529,   1970.1967437 ] ])

    self.redshift = gal_finder_kwargs['redshift']
    self.hubble = gal_finder_kwargs['hubble']
    halo_coords = comoving_halo_coords/(1. + self.redshift)/self.hubble

    # Make the necessary kwargs
    self.kwargs = gal_finder_kwargs

    self.galaxy_finder = general_galaxy_finder.GalaxyFinder( halo_coords, **self.kwargs ) 

    # Get the necessary reader.
    self.galaxy_finder.ahf_reader = read_ahf.AHFReader( self.kwargs['ahf_data_dir'] )
    
    # Get the full needed ahf info.
    self.galaxy_finder.ahf_reader.get_halos( 500 )

  ########################################################################

  @mock.patch( 'galaxy_diver.read_data.ahf.AHFReader.get_halos_add' )
  def test_valid_halo_inds( self, mock_get_halos_add ):

    # We don't actually want get_halos_add to do anything.
    mock_get_halos_add.side_effects = [ None, ]

    # Make sure we actually have a minimum
    self.galaxy_finder.minimum_value = 10

    # Modify the AHF halos data for easy replacement
    self.galaxy_finder.ahf_reader.ahf_halos = {}
    self.galaxy_finder.ahf_reader.ahf_halos['n_star'] = np.array( [ 100, 5, 10, 0 ] )

    actual = self.galaxy_finder.valid_halo_inds

    expected = np.array( [ 0, 2, ] )

    npt.assert_allclose( expected, actual )
    
  ########################################################################

  def test_dist_to_all_valid_halos( self ):
    '''Test that this works for using r_scale.'''

    self.galaxy_finder.particle_positions = np.array([
      [ 29414.96458784,  30856.75007114,  32325.90901812], # Right in the middle of mt halo 0 at snap 500
      [ 29414.96458784 + 50.,  30856.75007114,  32325.90901812], # Just outside the scale radius of mt halo 0 at snap 500.
      [ 29414.96458784,  30856.75007114 - 25.,  32325.90901812], # Just inside the scale radius of mt halo 0 at snap 500.
      ])
    self.galaxy_finder.particle_positions *= 1./(1. + self.redshift)/self.hubble
    self.galaxy_finder.n_particles = 3

    actual = self.galaxy_finder.dist_to_all_valid_halos

    # Build the expected output
    n_halos = self.galaxy_finder.ahf_reader.ahf_halos.index.size
    n_particles = self.galaxy_finder.n_particles
    expected_shape = ( n_particles, n_halos )

    npt.assert_allclose( actual[ 0, 0 ], 0., atol=1e-7 )
    npt.assert_allclose( actual[ 1, 0 ], 50.*1./(1. + self.redshift)/self.hubble )
    npt.assert_allclose( actual[ 2, 0 ], 25.*1./(1. + self.redshift)/self.hubble )

    self.assertEqual( actual.shape, expected_shape )

  ########################################################################

  def test_dist_to_all_valid_halos_single_particle( self ):
    '''Test that we can get the distance to all valid halos, formatted correctly, even for a single particle.
    '''

    assert False

  ########################################################################

  def test_find_containing_halos( self ):

    result = self.galaxy_finder.find_containing_halos()

    # If none of the distances are within any of the halos, we have a problem.
    assert result.sum() > 0

  ########################################################################

  def test_find_containing_halos_strict( self ):
    '''Here I'll restrict the fraction to a very small fraction of the virial radius,
    such that the sum of the results should be two.
    '''

    result = self.galaxy_finder.find_containing_halos( 0.0001 )

    # If none of the distances are within any of the halos, we have a problem.
    npt.assert_allclose( 2, result.sum() )

  ########################################################################

  def test_find_containing_halos_r_scale( self ):
    '''Test that this works for using r_scale.'''

    # Set the length scale
    self.galaxy_finder.galaxy_cut = 1.
    self.galaxy_finder.length_scale = 'r_scale'

    r_scale_500 = 21.113602882685832
    self.galaxy_finder.particle_positions = np.array([
      [ 29414.96458784,  30856.75007114,  32325.90901812], # Right in the middle of mt halo 0 at snap 500
      [ 29414.96458784 + r_scale_500*1.01,  30856.75007114,  32325.90901812], # Just outside the scale radius of mt halo 0 at snap 500.
      [ 29414.96458784 + r_scale_500*0.99,  30856.75007114,  32325.90901812], # Just inside the scale radius of mt halo 0 at snap 500.
      ])
    self.galaxy_finder.particle_positions *= 1./(1. + self.redshift)/self.hubble
    self.galaxy_finder.n_particles = 3

    actual = self.galaxy_finder.find_containing_halos( 1. )

    # Build the expected output
    n_halos = self.galaxy_finder.ahf_reader.ahf_halos.index.size
    expected = np.zeros( (self.galaxy_finder.particle_positions.shape[0], n_halos) ).astype( bool )
    expected[ 0, 0 ] = True
    expected[ 1, 0 ] = False
    expected[ 2, 0 ] = True

    npt.assert_allclose( actual, expected )

  ########################################################################

  def test_find_containing_halos_nan_particle( self ):
    # Anywhere the particle data has NaN values, we want that to read as False

    self.galaxy_finder.particle_positions = np.array([
      [ 29414.96458784,  30856.75007114,  32325.90901812], # Right in the middle of mt halo 0 at snap 500
      [ np.nan, np.nan, np.nan ], # Invalid values, because a particle with that ID didn't exist
      ])
    self.galaxy_finder.particle_positions *= 1./(1. + self.redshift)/self.hubble
    self.galaxy_finder.n_particles = 2
      
    actual = self.galaxy_finder.find_containing_halos()

    n_halos = self.galaxy_finder.ahf_reader.ahf_halos.index.size
    expected = np.zeros( (self.galaxy_finder.particle_positions.shape[0], n_halos) ).astype( bool )
    expected[ 0, 0 ] = True

    npt.assert_allclose( actual, expected )

  ########################################################################

  def test_find_mt_containing_halos( self ):
    
    self.galaxy_finder.particle_positions = np.array([
      [ 29414.96458784,  30856.75007114,  32325.90901812], # Right in the middle of mt halo 0 at snap 500
      [ 29467.07226789,  30788.6179313 ,  32371.38749237], # Right in the middle of mt halo 9 at snap 500.
                                                           # mt halo 9 is 0.5 Rvir_mt_0 (2 Rvir_mt_9) away from the center of mt halo 0
      [ 29073.22333685,  31847.72434505,  32283.53620817], # Right in the middle of mt halo 19 at snap 500.
      ])
    self.galaxy_finder.particle_positions *= 1./(1. + self.redshift)/self.hubble

    actual = self.galaxy_finder.find_mt_containing_halos( 2.5 )

    # Build the expected output
    expected = np.zeros( (self.galaxy_finder.particle_positions.shape[0], 6) ).astype( bool )
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
    self.galaxy_finder.galaxy_cut = 1.
    self.galaxy_finder.length_scale = 'r_scale'

    r_scale_500 = 21.113602882685832
    self.galaxy_finder.particle_positions = np.array([
      [ 29414.96458784,  30856.75007114,  32325.90901812], # Right in the middle of mt halo 0 at snap 500
      [ 29414.96458784 + r_scale_500*1.01,  30856.75007114,  32325.90901812], # Just outside the scale radius of mt halo 0 at snap 500.
      [ 29414.96458784 + r_scale_500*0.99,  30856.75007114,  32325.90901812], # Just inside the scale radius of mt halo 0 at snap 500.
                                                           # (It will be. It currently isn't.)
      ])
    self.galaxy_finder.particle_positions *= 1./(1. + self.redshift)/self.hubble
    self.galaxy_finder.n_particles = 3

    actual = self.galaxy_finder.find_mt_containing_halos( 1. )

    # Build the expected output
    n_halos = len( self.galaxy_finder.ahf_reader.mtree_halos )
    expected = np.zeros( (self.galaxy_finder.particle_positions.shape[0], n_halos) ).astype( bool )
    expected[ 0, 0 ] = True
    expected[ 1, 0 ] = False
    expected[ 2, 0 ] = True

    npt.assert_allclose( actual, expected )

  ########################################################################

  def test_find_mt_containing_halos_nan_particles( self ):
    '''Test that this works for using r_scale.'''

    self.galaxy_finder.particle_positions = np.array([
      [ 29414.96458784,  30856.75007114,  32325.90901812], # Right in the middle of mt halo 0 at snap 500
      [ np.nan, np.nan, np.nan, ], # Just outside the scale radius of mt halo 0 at snap 500.
      ])
    self.galaxy_finder.particle_positions *= 1./(1. + self.redshift)/self.hubble
    self.galaxy_finder.n_particles = 2

    actual = self.galaxy_finder.find_mt_containing_halos( 1. )

    # Build the expected output
    n_halos = len( self.galaxy_finder.ahf_reader.mtree_halos )
    expected = np.zeros( (self.galaxy_finder.particle_positions.shape[0], n_halos) ).astype( bool )
    expected[ 0, 0 ] = True

    npt.assert_allclose( actual, expected )

  ########################################################################

  def test_find_smallest_host_halo( self ):

    self.galaxy_finder.particle_positions = np.array([
      [ 29414.96458784,  30856.75007114,  32325.90901812],
      [ 31926.42103071,  51444.46756529,   1970.1967437 ],
      [ 29467.07226789,  30788.6179313 ,  32371.38749237],
      [ 29459.32290246,  30768.32556725,  32357.26078864], # Halo 3783, host halo 3610
      ])
    self.galaxy_finder.particle_positions *= 1./(1. + self.redshift)/self.hubble

    self.galaxy_finder.n_particles = 4

    expected = np.array( [0, 6962, 7, 3783] )
    actual = self.galaxy_finder.find_halo_id()

    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_find_smallest_host_halo_none( self ):

    self.galaxy_finder.particle_positions = np.array([
      [ 0., 0., 0. ],
      [ 0., 0., 0. ],
      ])

    expected = np.array( [-2, -2] )
    actual = self.galaxy_finder.find_halo_id()

    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_find_host_id( self ):

    self.galaxy_finder.particle_positions = np.array([
      [ 29414.96458784,  30856.75007114,  32325.90901812], # Halo 0, host halo 0
      [ 30068.5541178 ,  32596.72758226,  32928.1115097 ], # Halo 10, host halo 1
      [ 29459.32290246,  30768.32556725,  32357.26078864], # Halo 3783, host halo 3610
      ])
    self.galaxy_finder.particle_positions *= 1./(1. + self.redshift)/self.hubble

    self.galaxy_finder.n_particles = 3

    expected = np.array( [-1, 1, 3610] )
    actual = self.galaxy_finder.find_host_id()

    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_find_host_id_none( self ):

    self.galaxy_finder.particle_positions = np.array([
      [ 0., 0., 0. ],
      [ 0., 0., 0. ],
      ])

    expected = np.array( [-2, -2] )
    actual = self.galaxy_finder.find_host_id()

    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_find_mt_halo_id( self ):
    
    self.galaxy_finder.particle_positions = np.array([
      [ 29414.96458784,  30856.75007114,  32325.90901812], # Right in the middle of mt halo 0 at snap 500
      [ 29467.07226789,  30788.6179313 ,  32371.38749237], # Right in the middle of mt halo 9 at snap 500.
                                                           # mt halo 9 is 0.5 Rvir_mt_0 (2 Rvir_mt_9) away from the center of mt halo 0
      [ 29073.22333685,  31847.72434505,  32283.53620817], # Right in the middle of mt halo 19 at snap 500.
      [             0.,              0.,              0.], # The middle of nowhere.
      ])
    self.galaxy_finder.particle_positions *= 1./(1. + self.redshift)/self.hubble
    self.galaxy_finder.n_particles = 4

    actual = self.galaxy_finder.find_halo_id( 2.5, 'mt_halo_id' )

    # Build the expected output
    expected = np.array([ 0, 0, 19, -2 ])

    npt.assert_allclose( actual, expected )

  ########################################################################

  def test_find_mt_halo_id_early_universe( self ):
    '''Test that, when there are no galaxies formed, we return an mt halo value of -2'''
    
    # Set it to early redshifts
    self.galaxy_finder.snum = 0

    # It doesn't really matter where the particles are, because there shouldn't be any galaxies anyways....
    self.galaxy_finder.particle_positions = np.array([
      [ 29414.96458784,  30856.75007114,  32325.90901812], # Right in the middle of mt halo 0 at snap 500
      [ 29467.07226789,  30788.6179313 ,  32371.38749237], # Right in the middle of mt halo 9 at snap 500.
                                                           # mt halo 9 is 0.5 Rvir_mt_0 (2 Rvir_mt_9) away from the center of mt halo 0
      [ 29073.22333685,  31847.72434505,  32283.53620817], # Right in the middle of mt halo 19 at snap 500.
      [             0.,              0.,              0.], # The middle of nowhere.
      ])
    self.galaxy_finder.particle_positions *= 1./(1. + 30.)/self.hubble
    self.galaxy_finder.n_particles = 4

    actual = self.galaxy_finder.find_halo_id( 2.5, 'mt_halo_id' )

    # Build the expected output
    expected = np.array([ -2, -2, -2, -2 ])

    npt.assert_allclose( actual, expected )

  ########################################################################

  def test_find_ids( self ):

    particle_positions = np.array([
      [ 29414.96458784,  30856.75007114,  32325.90901812], # Halo 0, host halo 0
      [ 30068.5541178 ,  32596.72758226,  32928.1115097 ], # Halo 10, host halo 1
      [ 29459.32290246,  30768.32556725,  32357.26078864], # Halo 3783, host halo 3610
      ])
    particle_positions *= 1./(1. + self.redshift)/self.hubble

    expected = {
      'd_gal' : np.array( [ 0., 0., 0., ] ),
      'host_halo_id' : np.array( [-1, 1, 3610] ),
      'halo_id' : np.array( [0, 10, 3783] ),
      'host_gal_id' : np.array( [-1, 1, 3610] ),
      'gal_id' : np.array( [0, 10, 3783] ),
      'mt_gal_id' : np.array( [0, -2, -2] ),
      'mt_halo_id' : np.array( [0, 1, 0] ),
    }

    # Do the actual calculation
    galaxy_finder = general_galaxy_finder.GalaxyFinder( particle_positions, **self.kwargs )
    actual = galaxy_finder.find_ids()

    for key in expected.keys():
      print key
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
      'host_halo_id' : np.array( [-2, -2, -2] ),
      'halo_id' : np.array( [-2, -2, -2] ),
      'host_gal_id' : np.array( [-2, -2, -2] ),
      'gal_id' : np.array( [-2, -2, -2] ),
      'mt_gal_id' : np.array( [-2, -2, -2] ),
      'mt_halo_id' : np.array( [-2, -2, -2] ),
    }

    # Setup the input parameters
    snap0_kwargs = copy.deepcopy( self.kwargs )
    snap0_kwargs['snum'] = 0

    # Do the actual calculation
    galaxy_finder = general_galaxy_finder.GalaxyFinder( particle_positions, **snap0_kwargs )
    actual = galaxy_finder.find_ids()

    for key in expected.keys():
      print key
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
      'host_halo_id' : np.array( [-2, -2, -2] ),
      'halo_id' : np.array( [-2, -2, -2] ),
      'host_gal_id' : np.array( [-2, -2, -2] ),
      'gal_id' : np.array( [-2, -2, -2] ),
      'mt_gal_id' : np.array( [-2, -2, -2] ),
      'mt_halo_id' : np.array( [-2, -2, -2] ),
    }

    # Setup the input parameters
    snap0_kwargs = copy.deepcopy( self.kwargs )
    snap0_kwargs['snum'] = 1

    # Do the actual calculation
    galaxy_finder = general_galaxy_finder.GalaxyFinder( particle_positions, **snap0_kwargs )
    actual = galaxy_finder.find_ids()

    for key in expected.keys():
      print key
      npt.assert_allclose( expected[key], actual[key] )

  ########################################################################

  def test_pass_ahf_reader( self ):
    '''Test that it still works when we pass in an ahf_reader. '''

    particle_positions = np.array([
      [ 29414.96458784,  30856.75007114,  32325.90901812], # Halo 0, host halo 0
      [ 30068.5541178 ,  32596.72758226,  32928.1115097 ], # Halo 10, host halo 1
      [ 29459.32290246,  30768.32556725,  32357.26078864], # Halo 3783, host halo 3610
      ])
    particle_positions *= 1./(1. + self.redshift)/self.hubble

    expected = {
      'host_halo_id' : np.array( [-1, 1, 3610] ),
      'halo_id' : np.array( [0, 10, 3783] ),
      'host_gal_id' : np.array( [-1, 1, 3610] ),
      'gal_id' : np.array( [0, 10, 3783] ),
      'mt_gal_id' : np.array( [0, -2, -2] ),
      'mt_halo_id' : np.array( [0, 1, 0] ),
    }

    # Prepare an ahf_reader to pass along.
    ahf_reader = read_ahf.AHFReader( self.kwargs['ahf_data_dir'] )

    # Muck it up by making it try to retrieve data
    ahf_reader.get_halos( 600 )
    ahf_reader.get_mtree_halos( 600, tag='smooth' )

    # Do the actual calculation
    galaxy_finder = general_galaxy_finder.GalaxyFinder( particle_positions, ahf_reader=ahf_reader, **self.kwargs )
    actual = galaxy_finder.find_ids()

    for key in expected.keys():
      print key
      npt.assert_allclose( expected[key], actual[key] )

  ########################################################################

  def test_find_d_gal( self ):
    '''This tests we can find the shortest distance to the nearest galaxy.
    '''

    # Setup the distance so we don't have to calculate it.
    self.galaxy_finder._dist_to_all_valid_halos = np.array([
      [ 0.5, 1.0, 0.5, ],
      [ 15., 5., 3., ],
      [ 0.2, 2.5e-4, 4., ],
    ])

    actual = self.galaxy_finder.find_d_gal()

    expected = np.array([ 0.5, 3., 2.5e-4, ])

    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_find_d_other_gal( self ):
    '''This tests we can find the shortest distance to the nearest galaxy.
    '''

    # Setup the distance so we don't have to calculate it.
    self.galaxy_finder._dist_to_all_valid_halos = np.array([
      [ 0.5, 1.0, 0.75, ],
      [ 15., 5., 3., ],
      [ 0.2, 2.5e-4, 4., ],
    ])

    actual = self.galaxy_finder.find_d_other_gal()

    expected = np.array([ 0.75, 3., 2.5e-4, ])

    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_find_d_other_gal_main_halo_id_not_0( self ):
    '''This tests we can find the shortest distance to the nearest galaxy.
    '''

    self.galaxy_finder.main_mt_halo_id = 1

    # Setup the distance so we don't have to calculate it.
    self.galaxy_finder._dist_to_all_valid_halos = np.array([
      [ 0.5, 1.0, 0.75, ],
      [ 15., 5., 3., ],
      [ 0.2, 2.5e-4, 4., ],
    ])

    actual = self.galaxy_finder.find_d_other_gal()

    expected = np.array([ 0.5, 3., 0.2, ])

    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_find_d_other_gal_early_universe( self ):
    '''This tests we can find the shortest distance to the nearest galaxy.
    '''

    self.galaxy_finder.snum = 1

    # Setup the distance so we don't have to calculate it.
    self.galaxy_finder._dist_to_all_valid_halos = np.array([
      [ 0.5, 1.0, 0.75, ],
      [ 15., 5., 3., ],
      [ 0.2, 2.5e-4, 4., ],
    ])

    actual = self.galaxy_finder.find_d_other_gal()

    expected = np.array([ 0.5, 3., 2.5e-4, ])

    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_find_d_other_gal_scaled( self ):
    '''This tests we can find the shortest distance to the nearest galaxy.
    '''

    # Setup dummy data
    self.galaxy_finder._ahf_halos_length_scale_pkpc = np.array([ 1., 2., 3., 4., 5., ])
    self.galaxy_finder._valid_halo_inds = np.array([ 0, 1, 2, 3, ])
    self.galaxy_finder._dist_to_all_valid_halos = np.array([
      [ 2., 4., 6., 8. ],
      [ 4., 3., 2., 1., ],
      [ 10., 8., 6., 7., ],
    ])

    # Make sure we set the number of particles correctly, to match the number we're using
    self.galaxy_finder.n_particles = 3

    actual = self.galaxy_finder.find_d_other_gal( scaled=True )

    expected = np.array([ 2., 0.25, 2., ])

    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_find_d_other_gal_scaled_early_universe( self ):
    '''This tests we can find the shortest distance to the nearest galaxy.
    '''

    # Setup dummy data
    self.galaxy_finder.snum = 1
    self.galaxy_finder._ahf_halos_length_scale_pkpc = np.array([ 1., 2., 3., 4., 5., ])
    self.galaxy_finder._valid_halo_inds = np.array([ 0, 1, 2, 3, ])
    self.galaxy_finder._dist_to_all_valid_halos = np.array([
      [ 2., 4., 6., 8. ],
      [ 4., 3., 2., 1., ],
      [ 10., 8., 6., 7., ],
    ])

    # Make sure we set the number of particles correctly, to match the number we're using
    self.galaxy_finder.n_particles = 3

    actual = self.galaxy_finder.find_d_other_gal( scaled=True )

    expected = np.array([ 2., 0.25, 2., ])

    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_find_d_other_gal_scaled_no_halos( self ):
    '''This tests we can find the shortest distance to the nearest galaxy.
    '''

    # Setup dummy data
    self.galaxy_finder.snum = 1
    self.galaxy_finder.ahf_reader.sdir = './tests/data/analysis_dir4'
    self.galaxy_finder.ahf_reader.get_halos( 1 )

    # Make sure we set the number of particles correctly, to match the number we're using
    #self.galaxy_finder.n_particles = 3

    actual = self.galaxy_finder.find_d_other_gal( scaled=True )

    expected = np.array([ -2., -2., ])

    npt.assert_allclose( expected, actual )

  ########################################################################

  @mock.patch( 'galaxy_diver.read_data.ahf.AHFReader.get_halos_add' )
  def test_find_d_other_gal_scaled_no_halos_with_sufficient_mass( self, mock_get_halos_add ):
    '''This tests we can find the shortest distance to the nearest galaxy.
    '''

    # Setup dummy data
    self.galaxy_finder.snum = 12
    self.galaxy_finder.minimum_value = 10
    self.galaxy_finder.ahf_reader.sdir = './tests/data/analysis_dir4'
    self.galaxy_finder.ahf_reader.get_halos( 12 )

    # Make sure we set the number of particles correctly, to match the number we're using
    #self.galaxy_finder.n_particles = 3

    actual = self.galaxy_finder.find_d_other_gal( scaled=True )

    expected = np.array([ -2., -2., ])

    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_find_d_other_gal_scaled_main_halo_id_not_0( self ):
    '''This tests we can find the shortest distance to the nearest galaxy.
    '''

    # Setup dummy data
    self.galaxy_finder.main_mt_halo_id = 3
    self.galaxy_finder._ahf_halos_length_scale_pkpc = np.array([ 1., 2., 3., 4., 5., ])
    self.galaxy_finder._valid_halo_inds = np.array([ 0, 1, 2, 3, ])
    self.galaxy_finder._dist_to_all_valid_halos = np.array([
      [ 2., 4., 6., 8. ],
      [ 4., 3., 2., 1., ],
      [ 10., 8., 6., 7., ],
    ])

    # Make sure we set the number of particles correctly, to match the number we're using
    self.galaxy_finder.n_particles = 3

    actual = self.galaxy_finder.find_d_other_gal( scaled=True )

    expected = np.array([ 2., 2./3., 2., ])

    npt.assert_allclose( expected, actual )

########################################################################

class TestGalaxyFinderMinimumStellarMass( unittest.TestCase ):
  '''Test that we're properly applying a minimum stellar mass for a halo to be counted as containing a galaxy.'''

  def setUp( self ):

    gal_finder_kwargs_min_mstar = {
      'length_scale' : 'r_scale',
      'minimum_criteria' : 'M_star',
      'minimum_value' : 1e6,

      'redshift' : 6.1627907799999999,
      'snum' : 50,
      'hubble' : 0.70199999999999996,
      'ahf_data_dir' : './tests/data/analysis_dir3',
      'mtree_halos_index' : 600,
      'halo_file_tag' : 'smooth',
      'main_mt_halo_id' : 0,
      'galaxy_cut' : 0.1,
      'length_scale' : 'Rvir',

      'ids_to_return' : [ 'halo_id', 'host_halo_id', 'gal_id', 'host_gal_id', 'mt_halo_id', 'mt_gal_id', 'd_gal', 'd_other_gal_scaled', ],
    }

    # Get input data
    comoving_particle_positions = np.array([
      [ 30252.60118534,  29483.51635481,  31011.17715464], # Right in the middle of mt halo 0 (AHF halo id 3) at snum 50.
                                                           # This halo contains a galaxy with 1e6.7 Msun of stars at this redshift.
      [ 28651.1193359,  29938.7253038,  32168.1380575], # Right in the middle of mt halo 19 (AHF halo id 374) at snum 50
                                                           # This halo no stars at this redshift.
    ])

    self.redshift = gal_finder_kwargs_min_mstar['redshift']
    self.hubble = gal_finder_kwargs_min_mstar['hubble']
    particle_positions = comoving_particle_positions/(1. + self.redshift)/self.hubble

    # Make the necessary kwargs
    self.kwargs = gal_finder_kwargs_min_mstar

    self.galaxy_finder = general_galaxy_finder.GalaxyFinder( particle_positions, **self.kwargs ) 

    # Get the necessary reader.
    self.galaxy_finder.ahf_reader = read_ahf.AHFReader( self.kwargs['ahf_data_dir'] )
    
    # Get the full needed ahf info.
    self.galaxy_finder.ahf_reader.get_halos( 50 )

  ########################################################################

  def test_find_containing_halos( self ):

    actual = self.galaxy_finder.find_containing_halos( 1. )

    # Build the expected output
    n_halos = self.galaxy_finder.ahf_reader.ahf_halos.index.size
    expected = np.zeros( (self.galaxy_finder.particle_positions.shape[0], n_halos) ).astype( bool )
    expected[ 0, 3 ] = True # Should only be in the galaxy with sufficient stellar mass.

    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_find_mt_containing_halos( self ):

    actual = self.galaxy_finder.find_mt_containing_halos( 1. )

    # Build the expected output
    n_halos = len( self.galaxy_finder.ahf_reader.mtree_halos )
    expected = np.zeros( (self.galaxy_finder.particle_positions.shape[0], n_halos) ).astype( bool )
    expected[ 0, 0 ] = True # Should only be in the galaxy with sufficient stellar gas.

    npt.assert_allclose( expected, actual )

########################################################################
########################################################################

class TestGalaxyFinderMinimumNumStars( unittest.TestCase ):
  '''Test that we're properly applying a minimum number of stars for a halo to be counted as containing a galaxy.'''

  def setUp( self ):

    gal_finder_kwargs_min_nstar = {
      'length_scale' : 'r_scale',
      'minimum_criteria' : 'n_star',
      'minimum_value' : 10,

      'redshift' : 6.1627907799999999,
      'snum' : 50,
      'hubble' : 0.70199999999999996,
      'ahf_data_dir' : './tests/data/analysis_dir3',
      'mtree_halos_index' : 600,
      'halo_file_tag' : 'smooth',
      'main_mt_halo_id' : 0,
      'galaxy_cut' : 0.1,
      'length_scale' : 'Rvir',

      'ids_to_return' : [ 'halo_id', 'host_halo_id', 'gal_id', 'host_gal_id', 'mt_halo_id', 'mt_gal_id', 'd_gal', 'd_other_gal_scaled', ],
    }

    # Get input data
    comoving_particle_positions = np.array([
      [ 30252.60118534,  29483.51635481,  31011.17715464], # Right in the middle of mt halo 0 (AHF halo id 3) at snum 50.
                                                           # This halo contains a galaxy with 1e6.7 Msun of stars at this redshift.
      [ 28651.1193359,  29938.7253038,  32168.1380575], # Right in the middle of mt halo 19 (AHF halo id 374) at snum 50
                                                           # This halo no stars at this redshift.
    ])

    self.redshift = gal_finder_kwargs_min_nstar['redshift']
    self.hubble = gal_finder_kwargs_min_nstar['hubble']
    particle_positions = comoving_particle_positions/(1. + self.redshift)/self.hubble

    # Make the necessary kwargs
    self.kwargs = gal_finder_kwargs_min_nstar

    self.galaxy_finder = general_galaxy_finder.GalaxyFinder( particle_positions, **self.kwargs ) 

    # Get the necessary reader.
    self.galaxy_finder.ahf_reader = read_ahf.AHFReader( self.kwargs['ahf_data_dir'] )
    
    # Get the full needed ahf info.
    self.galaxy_finder.ahf_reader.get_halos( 50 )

  ########################################################################

  def test_find_containing_halos( self ):

    actual = self.galaxy_finder.find_containing_halos( 1. )

    # Build the expected output
    n_halos = self.galaxy_finder.ahf_reader.ahf_halos.index.size
    expected = np.zeros( (self.galaxy_finder.particle_positions.shape[0], n_halos) ).astype( bool )
    expected[ 0, 3 ] = True # Should only be in the galaxy with sufficient stellar mass.

    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_find_mt_containing_halos( self ):

    actual = self.galaxy_finder.find_mt_containing_halos( 1. )

    # Build the expected output
    n_halos = len( self.galaxy_finder.ahf_reader.mtree_halos )
    expected = np.zeros( (self.galaxy_finder.particle_positions.shape[0], n_halos) ).astype( bool )
    expected[ 0, 0 ] = True # Should only be in the galaxy with sufficient stellar gas.

    npt.assert_allclose( expected, actual )

########################################################################
########################################################################

class TestFindMassRadii( unittest.TestCase ):

  def setUp( self ):

    # Test Data
    self.kwargs = dict( gal_finder_kwargs )
    self.kwargs['particle_masses'] = np.array([ 1., 2., 3., 4., ])
    particle_positions = np.array([
      [ 0., 0., 0., ],
      [ 0., 0., 0., ],
      [ 0., 0., 0., ],
      [ 0., 0., 0., ],
    ]) # These shouldn't ever be used directly, since we're relying on the results of previous functions.

    self.galaxy_finder = general_galaxy_finder.GalaxyFinder( particle_positions, **self.kwargs ) 

  ########################################################################

  def test_mass_inside_galaxy_cut( self ):

    # Test Data
    self.galaxy_finder._ahf_halos_length_scale_pkpc = np.array([ 200., 10., 100., 50., ])
    self.galaxy_finder._valid_halo_inds = np.array([ 0, 1, 2, ])
    self.galaxy_finder._dist_to_all_valid_halos = np.array([ 
      [ 0., 10., 500., ],
      [ 15., 5., 485., ],
      [ 10., 0., 490., ],
      [ 500., 490., 0., ],
    ])

    expected = np.array([ 6., 3., 4., ])
    actual = self.galaxy_finder.mass_inside_galaxy_cut
    
    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_mass_inside_galaxy_cut_no_valid_inds( self ):
    '''Test we still get a reasonable result out, even when there's not a single valid halo.'''

    # Test Data
    self.galaxy_finder._ahf_halos_length_scale_pkpc = np.array([ 200., 10., 100., 50., ])
    self.galaxy_finder._valid_halo_inds = np.array([])
    self.galaxy_finder._dist_to_all_valid_halos = np.array([ 
      [ 0., 10., 500., ],
      [ 15., 5., 485., ],
      [ 10., 0., 490., ],
      [ 500., 490., 0., ],
    ])

    actual = self.galaxy_finder.mass_inside_galaxy_cut
    expected = np.array([])
    
    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_mass_inside_galaxy_cut_no_halos( self ):
    '''Test we still get a reasonable result out, even when there aren't any halos formed yet.'''

    # Test Data
    self.galaxy_finder._ahf_halos_length_scale_pkpc = np.array([])
    self.galaxy_finder._valid_halo_inds = np.array([])

    actual = self.galaxy_finder.mass_inside_galaxy_cut
    expected = np.array([])
    
    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_mass_inside_galaxy_cut_no_inside_cut( self ):
    '''Make sure we give the right results when no particles are inside the cut.'''

    # Test Data
    self.galaxy_finder._ahf_halos_length_scale_pkpc = np.array([ 0.1, 0.1, 0.1, 0.1, ])
    self.galaxy_finder._valid_halo_inds = np.array([ 0, 1, 2, ])
    self.galaxy_finder._dist_to_all_valid_halos = np.array([ 
      [ 100., 10., 500., ],
      [ 15., 5., 485., ],
      [ 10., 100., 490., ],
      [ 500., 490., 100., ],
    ])

    expected = np.array([ 0., 0., 0., ])
    actual = self.galaxy_finder.mass_inside_galaxy_cut
    
    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_mass_inside_all_halos( self ):

    # Test Data
    self.galaxy_finder._ahf_halos_length_scale_pkpc = np.array([ 200., 10., 100., 50., np.nan, ])
    self.galaxy_finder._valid_halo_inds = np.array([ 0, 1, 2, 4, ])
    self.galaxy_finder._dist_to_all_valid_halos = np.array([ 
      [ 0., 10., 500., np.nan, ],
      [ 15., 5., 485., np.nan, ],
      [ 10., 0., 490., np.nan, ],
      [ 500., 490., 0., np.nan, ],
    ])

    expected = np.array([ 6., 3., 4., np.nan, np.nan ])
    actual = self.galaxy_finder.mass_inside_all_halos
    
    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_mass_inside_all_halos_no_valid_gals( self ):

    # Test Data
    self.galaxy_finder._ahf_halos_length_scale_pkpc = np.array([ 200., 10., 100., 50., np.nan, ])
    self.galaxy_finder._valid_halo_inds = np.array([])

    expected = np.array([ np.nan, np.nan, np.nan, np.nan, np.nan ])
    actual = self.galaxy_finder.mass_inside_all_halos
    
    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_cumlulative_mass_valid_halos( self ):

    # Test Data
    self.galaxy_finder._dist_to_all_valid_halos = np.array([
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
    actual = self.galaxy_finder.cumulative_mass_valid_halos

    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_get_mass_radius( self ):

    # Test Data
    self.galaxy_finder._ahf_halos_length_scale_pkpc = np.array([ 0., 0., 0. ]) # Values shouldn't matter here
    self.galaxy_finder._valid_halo_inds = np.array([ 0, 1, ])
    self.galaxy_finder._mass_inside_galaxy_cut = np.array([ 10., 19, ])
    self.galaxy_finder._dist_to_all_valid_halos = np.array([
      [ 0., 500., ],
      [ 480., 450., ],
      [ 50., 20., ],
      [ 490., 10., ],
    ])
    self.galaxy_finder._cumulative_mass_valid_halos = np.array([
      [ 1., 10., ],
      [ 6., 9., ],
      [ 4., 7., ],
      [ 10., 4., ],
    ]),

    # Expected result, in comoving coords.
    expected = np.array( [ 50., 450., np.nan ] )*( 1. + 0.16946003 )*0.702
    actual = self.galaxy_finder.get_mass_radius( 0.5 )

    npt.assert_allclose( expected, actual )
    
########################################################################
########################################################################

class TestSummedQuantityInsideGalaxy( unittest.TestCase ):
  '''Test that we can calculate a more general attribute inside a galaxy.'''

  def setUp( self ):

    # Test Data
    self.kwargs = dict( gal_finder_kwargs )
    self.kwargs['particle_masses'] = np.array([ 2., 2., 1., 3., ])
    particle_positions = np.array([
      [ 0., 0., 0., ],
      [ 0., 0., 0., ],
      [ 0., 0., 0., ],
      [ 0., 0., 0., ],
    ]) # These shouldn't ever be used directly, since we're relying on the results of previous functions.

    self.galaxy_finder = general_galaxy_finder.GalaxyFinder( particle_positions, **self.kwargs ) 

  ########################################################################

  def test_summed_quantity_inside_galaxy_valid_halos( self ):

    # Test Data
    self.galaxy_finder._ahf_halos_length_scale_pkpc = np.array([ 200., 10., 100., 50., ])
    self.galaxy_finder._valid_halo_inds = np.array([ 0, 1, 2, ])
    self.galaxy_finder._dist_to_all_valid_halos = np.array([ 
      [ 0., 10., 500., ],
      [ 15., 5., 485., ],
      [ 10., 0., 490., ],
      [ 500., 490., 0., ],
    ])
    particle_quantities = np.array([ 1., 2., 3., 4., ])

    actual = self.galaxy_finder.summed_quantity_inside_galaxy_valid_halos( particle_quantities, np.nan )
    expected = np.array([ 6., 3., 4., ])
    
    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_summed_quantity_valid_halos_no_inside_cut( self ):
    '''Make sure we give the right results when no particles are inside the cut.'''

    # Test Data
    self.galaxy_finder._ahf_halos_length_scale_pkpc = np.array([ 0.1, 0.1, 0.1, 0.1, ])
    self.galaxy_finder._valid_halo_inds = np.array([ 0, 1, 2, ])
    self.galaxy_finder._dist_to_all_valid_halos = np.array([ 
      [ 100., 10., 500., ],
      [ 15., 5., 485., ],
      [ 10., 100., 490., ],
      [ 500., 490., 100., ],
    ])
    particle_quantities = np.array([ 1., 2., 3., 4., ])

    expected = np.array([ np.nan, np.nan, np.nan, ])
    actual = self.galaxy_finder.summed_quantity_inside_galaxy_valid_halos( particle_quantities, np.nan )
    
    npt.assert_allclose( expected, actual )

  ########################################################################

  @mock.patch( 'galaxy_diver.galaxy_finder.finder.GalaxyFinder.dist_to_all_valid_halos_fn' )
  def test_summed_quantity_inside_galaxy_low_memory_mode( self, mock_dist_all_valid ):
    '''Test that we can get the summed quantity inside the galaxy, but reduce the memory consumption
    (at the cost of speed) by doing getting the sum for less galaxies at a given time.
    '''

    # Test Data
    self.galaxy_finder._ahf_halos_length_scale_pkpc = np.array([ 200., 10., 100., 50., ])
    self.galaxy_finder._valid_halo_inds = np.array([ 0, 1, 2, ])
    mock_dist_all_valid.side_effect = [ 
      np.array( [ 0., 10., 500., ] ),
      np.array( [ 15., 5., 485., ] ),
      np.array( [ 10., 0., 490., ] ),
      np.array( [ 500., 490., 0., ] ),
    ]
    particle_quantities = np.array([ 1., 2., 3., 4., ])

    # Change parameters of galaxy finder to run low memory node.
    self.galaxy_finder.low_memory_mode = True

    actual = self.galaxy_finder.summed_quantity_inside_galaxy_valid_halos( particle_quantities )
    expected = np.array([ 6., 3., 4., ])
    
    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_summed_quantity_inside_galaxy( self ):

    # Test Data
    self.galaxy_finder._ahf_halos_length_scale_pkpc = np.array([ 200., 10., 100., 50., np.nan, ])
    self.galaxy_finder._valid_halo_inds = np.array([ 0, 1, 2, 4, ])
    self.galaxy_finder._dist_to_all_valid_halos = np.array([ 
      [ 0., 10., 500., np.nan, ],
      [ 15., 5., 485., np.nan, ],
      [ 10., 0., 490., np.nan, ],
      [ 500., 490., 0., np.nan, ],
    ])
    particle_quantities = np.array([ 1., 2., 3., 4., ])

    expected = np.array([ 6., 3., 4., np.nan, np.nan ])
    actual = self.galaxy_finder.summed_quantity_inside_galaxy( 
      particle_quantities,
      np.nan,
    )
    
    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_weighted_summed_quantity_inside_galaxy( self ):

    # Test Data
    self.galaxy_finder._ahf_halos_length_scale_pkpc = np.array([ 200., 10., 100., 50., np.nan, ])
    self.galaxy_finder._valid_halo_inds = np.array([ 0, 1, 2, 4, ])
    self.galaxy_finder._dist_to_all_valid_halos = np.array([ 
      [ 0., 10., 500., np.nan, ],
      [ 15., 5., 485., np.nan, ],
      [ 10., 0., 490., np.nan, ],
      [ 500., 490., 0., np.nan, ],
    ])
    particle_quantities = np.array([ 1., 2., 3., 4., ])
    particle_weights = np.array([ 4., 3., 2., 1., ])

    actual = self.galaxy_finder.weighted_summed_quantity_inside_galaxy( particle_quantities, particle_weights, np.nan )
    expected = np.array([ 16./9., 3., 4., np.nan, np.nan, ])
    
    npt.assert_allclose( expected, actual )






