#!/usr/bin/env python
'''Testing for read_ahf.py

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import glob
from mock import call, patch
import numpy as np
import numpy.testing as npt
import os
import pdb
import pytest
import unittest

import galaxy_diver.analyze_data.ahf as analyze_ahf
import galaxy_diver.utils.utilities as utilities

sdir = './tests/data/analysis_dir'
sdir2 = './tests/data/analysis_dir2'
data_sdir = './tests/data/sdir'
data_sdir2 = './tests/data/sdir2'

########################################################################

# Decorator for skipping slow tests
slow = pytest.mark.skipif(
    not pytest.config.getoption("--runslow"),
    reason="need --runslow option to run"
)

########################################################################
########################################################################

class TestAHFUpdater( unittest.TestCase ):

  def setUp( self ):

    self.ahf_updater = analyze_ahf.AHFUpdater( sdir )

    # Remove smooth halo files that are generated
    halo_filepaths = glob.glob( './tests/data/ahf_test_data/halo_*_smooth.dat' )
    for halo_filepath in halo_filepaths:
      if os.path.isfile( halo_filepath ):
        os.system( 'rm {}'.format( halo_filepath ) )

  ########################################################################

  def tearDown( self ):

    # Remove extra files that are generated
    halo_filepaths = glob.glob( './tests/data/ahf_test_data/halo_*_test.dat' )
    for halo_filepath in halo_filepaths:
      if os.path.isfile( halo_filepath ):
        os.system( 'rm {}'.format( halo_filepath ) )

  ########################################################################

  def test_save_and_load_mtree_halo_files( self ):

    self.ahf_updater.get_mtree_halos( 'snum' )
    expected = self.ahf_updater.mtree_halos[10]['ID']
    expected_detail = self.ahf_updater.mtree_halos[0]['ID'][500]

    # Save
    save_tag = 'test'
    self.ahf_updater.save_mtree_halos( save_tag )

    # Load
    new_ahf_reader = analyze_ahf.AHFUpdater( sdir )
    new_ahf_reader.get_mtree_halos( 'snum', tag=save_tag )
    actual = new_ahf_reader.mtree_halos[10]['ID']
    actual_detail = new_ahf_reader.mtree_halos[0]['ID'][500]

    npt.assert_allclose( expected, actual )
    npt.assert_allclose( expected_detail, actual_detail )

  ########################################################################

  def test_get_accurate_redshift( self ):

    expected = 0.00015930

    self.ahf_updater.get_mtree_halos( 'snum' )

    self.ahf_updater.get_accurate_redshift( './tests/data/sdir' )

    actual = self.ahf_updater.mtree_halos[0]['redshift'][599]

    npt.assert_allclose( expected, actual, )

  ########################################################################

  def test_smooth_mtree_halos( self ):
    
    self.ahf_updater.get_mtree_halos( 'snum' )

    self.ahf_updater.smooth_mtree_halos( data_sdir )

    # Test that the redshift worked.
    redshift_expected_598 = 0.00031860
    redshift_actual = self.ahf_updater.mtree_halos[0]['redshift'][598]
    npt.assert_allclose( redshift_expected_598, redshift_actual )

    # Test that r_vir worked (that we never make a dip down in the early snapshot)
    r_vir_min = self.ahf_updater.mtree_halos[0]['Rvir'][210].min()
    assert r_vir_min > 30.

    # Test that m_vir worked (that we never make a dip down in the early snapshot)
    m_vir_min = self.ahf_updater.mtree_halos[0]['Mvir'][210].min()
    assert m_vir_min > 1e11

  ########################################################################

  def test_smooth_mtree_halos_fire1( self ):
    
    # Get the right directory
    self.ahf_updater.sdir = sdir2
    self.ahf_updater.get_mtree_halos( 440 )

    self.ahf_updater.smooth_mtree_halos( data_sdir2 )

    # Test that the redshift worked.
    redshift_expected_439 = 0.0049998743750157
    redshift_actual = self.ahf_updater.mtree_halos[0]['redshift'][439]
    npt.assert_allclose( redshift_expected_439, redshift_actual )

    # Test that r_vir worked (that we never make a dip down in the early snapshot)
    r_vir_min = self.ahf_updater.mtree_halos[0]['Rvir'][326].min()
    assert r_vir_min > 100.

    # Test that m_vir worked (that we never make a dip down in the early snapshot)
    m_vir_min = self.ahf_updater.mtree_halos[0]['Mvir'][326].min()
    assert m_vir_min > 1e11

  ########################################################################

  @slow
  def test_save_smooth_mtree_halos( self ):

    # Get the results
    self.ahf_updater.save_smooth_mtree_halos( data_sdir, 'snum', )

    # Load the saved files
    self.ahf_updater.get_mtree_halos( 'snum', 'smooth' )

    # Test that the redshift worked.
    redshift_expected_598 = 0.00031860
    redshift_actual = self.ahf_updater.mtree_halos[0]['redshift'][598]
    npt.assert_allclose( redshift_expected_598, redshift_actual )

    # Test that r_vir worked (that we never make a dip down in the early snapshot)
    r_vir_min = self.ahf_updater.mtree_halos[0]['Rvir'][210].min()
    assert r_vir_min > 30.

    # Test that m_vir worked (that we never make a dip down in the early snapshot)
    m_vir_min = self.ahf_updater.mtree_halos[0]['Mvir'][210].min()
    assert m_vir_min > 1e11

  ########################################################################

  @slow
  def test_save_smooth_mtree_halos_different_snum( self ):
  
    # Pass it the right directory
    self.ahf_updater.sdir = sdir2

    # Get the results
    self.ahf_updater.save_smooth_mtree_halos( data_sdir2, 440, False )

    # Load the saved files
    self.ahf_updater.get_mtree_halos( 440, 'smooth' )

    # Test that the redshift worked.
    redshift_expected_439 = 0.0049998743750157
    redshift_actual = self.ahf_updater.mtree_halos[0]['redshift'][439]
    npt.assert_allclose( redshift_expected_439, redshift_actual )

    # Test that r_vir worked (that we never make a dip down in the early snapshot)
    r_vir_min = self.ahf_updater.mtree_halos[0]['Rvir'][210].min()
    assert r_vir_min > 30.

    # Test that m_vir worked (that we never make a dip down in the early snapshot)
    m_vir_min = self.ahf_updater.mtree_halos[0]['Mvir'][210].min()
    assert m_vir_min > 1e11

    # Make sure there aren't NaN's
    assert self.ahf_updater.mtree_halos[0]['Rvir'][440] > 0

  ########################################################################

  @patch( 'galaxy_diver.read_data.ahf.AHFReader.get_halos_add', )
  def test_save_custom_mtree_halos( self, mock_get_halos_add ):

    # Run the test
    halo_ids = np.array( [ 0, 3, 5, ] )
    self.ahf_updater.save_custom_mtree_halos(
      snums=[600,550,500],
      halo_ids=halo_ids,
      metafile_dir=data_sdir,
    )

    # Load in new file
    self.ahf_updater.get_mtree_halos( tag='custom' )
    mtree_halo = self.ahf_updater.mtree_halos[0]

    # Compare IDs
    npt.assert_allclose( halo_ids, mtree_halo['ID'] )

    # Compare number of substructures (nice easy integers to compare)
    expected = np.array([ 792, 48, 55 ])
    actual = mtree_halo['numSubStruct']
    npt.assert_allclose( expected, actual )

    # Compare the snapshot number
    expected = np.array([ 600, 550, 500, ])
    actual = mtree_halo.index
    npt.assert_allclose( expected, actual )

    # Compare the redshift
    expected = np.array([ 0., 0.069847, 0.169460, ])
    actual = mtree_halo['redshift']
    npt.assert_allclose( expected, actual, atol=1e-3 )

  ########################################################################

  def test_save_custom_mtree_halos_including_ahf_halos_add( self ):

    # Run the test
    halo_ids = np.array( [ 3, ] )
    self.ahf_updater.save_custom_mtree_halos( snums=[600,], halo_ids=halo_ids, metafile_dir=data_sdir, )

    # Load in new file
    self.ahf_updater.get_mtree_halos( tag='custom' )
    mtree_halo = self.ahf_updater.mtree_halos[0]
    
    expected = np.array([ 13.753927, ])
    actual = mtree_halo['cAnalytic']
    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_save_custom_mtree_halos_default_snums( self ):
  
    # Run the test
    halo_ids = np.array( [ 3, ] )
    self.ahf_updater.save_custom_mtree_halos( snums=600, halo_ids=halo_ids, metafile_dir=data_sdir, )

    # Load in new file
    self.ahf_updater.get_mtree_halos( tag='custom' )
    mtree_halo = self.ahf_updater.mtree_halos[0]
    
    expected = np.array([ 600 ])
    actual = mtree_halo.index
    npt.assert_allclose( expected, actual )

  ########################################################################

  @slow
  def test_get_analytic_concentration_mtree_halos( self ):

    # Load the saved files
    self.ahf_updater.get_mtree_halos( 'snum', )
    self.ahf_updater.get_analytic_concentration( data_sdir )

    c_vir_z0_expected = 10.66567139
    c_vir_z0_actual = self.ahf_updater.mtree_halos[0]['cAnalytic'][600]

    npt.assert_allclose( c_vir_z0_expected, c_vir_z0_actual )

  ########################################################################

  @slow
  def test_save_and_load_ahf_halos_add( self ):

    # Save halos_add
    self.ahf_updater.save_ahf_halos_add( 600, data_sdir )

    # Load halos_add
    self.ahf_updater.get_halos_add( 600 )

    # Check that we calculated the concentration correctly.
    c_vir_z0_expected = 10.66567139
    c_vir_z0_actual = self.ahf_updater.ahf_halos_add['cAnalytic'][0]
    npt.assert_allclose( c_vir_z0_expected, c_vir_z0_actual )

  ########################################################################

  @patch( 'galaxy_diver.analyze_data.ahf.AHFUpdater.save_ahf_halos_add' )
  def test_save_multiple_ahf_halos_adds( self, mock_save_ahf_halos_add ):

    self.ahf_updater.save_multiple_ahf_halos_adds( data_sdir, 500, 600, 50 )

    calls = [
      call(500, data_sdir),
      call(550, data_sdir),
      call(600, data_sdir),
      ]
    mock_save_ahf_halos_add.assert_has_calls( calls, any_order=True )

########################################################################
########################################################################

class TestMassRadii( unittest.TestCase ):

  def setUp( self ):

    self.ahf_updater = analyze_ahf.AHFUpdater( sdir )

    snum = 600
    self.ahf_updater.get_halos( snum )

  ########################################################################

  @patch( 'galaxy_diver.galaxy_finder.finder.GalaxyFinder.get_mass_radius', )
  @patch( 'galaxy_diver.galaxy_finder.finder.GalaxyFinder.__init__', )
  def test_get_mass_radii( self, mock_init, mock_get_mass_radius ):

    mock_init.side_effect = [ None, ]
    mock_get_mass_radius.side_effect = [ np.arange(10)*10., np.linspace( 50., 200., 10 ), ]

    mass_fractions = [ 0.5, 0.99, ]

    expected = [ np.arange(10)*10., np.linspace( 50., 200., 10 ), ]
    data_dir = os.path.join( data_sdir, 'output' )
    actual = self.ahf_updater.get_mass_radii( mass_fractions, data_dir, 0.15, 'R_vir', )

    npt.assert_allclose( expected, actual )

  ########################################################################
  
  def test_get_mass_radii_from_subsampled_data( self,):
    '''Now test, using actual simulation data (even if it's subsampled).'''

    mass_fractions = [ 0.5,  ]
    data_dir = os.path.join( data_sdir, 'output' )

    actual = self.ahf_updater.get_mass_radii( mass_fractions, data_dir, 0.15, 'R_vir', )
    expected = np.array( [ np.nan, ]*9 )

    npt.assert_allclose( expected, actual[0] )

  ########################################################################

  @patch( 'galaxy_diver.analyze_data.ahf.AHFUpdater.get_analytic_concentration' )
  def test_save_ahf_halos_add_including_masses( self, mock_get_analytic_concentration ):

    mock_get_analytic_concentration.side_effect = [ np.zeros( 9 ), ]

    # Save halos_add
    sim_data_dir = os.path.join( data_sdir, 'output' )

    self.ahf_updater.save_ahf_halos_add( 600, data_sdir, [ 0.5, 0.99 ], sim_data_dir, 0.15, 'R_vir', )

    # Load halos_add
    self.ahf_updater.get_halos_add( 600 )

    # Make sure the columns exist
    assert 'Rmass0.5' in self.ahf_updater.ahf_halos.columns
    assert 'Rmass0.99' in self.ahf_updater.ahf_halos.columns




    







