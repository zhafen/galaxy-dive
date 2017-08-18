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

import galaxy_diver.read_data.ahf as read_ahf
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

class TestAHFReader( unittest.TestCase ):

  def setUp( self ):

    self.ahf_reader = read_ahf.AHFReader( sdir )

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

  def test_get_halos( self ):

    self.ahf_reader.get_halos( 500 )

    expected = 789
    actual = self.ahf_reader.ahf_halos['numSubStruct'][0]
    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_get_mtree_idx( self ):

    self.ahf_reader.get_mtree_idx( 500 )

    expected = 10
    actual = self.ahf_reader.ahf_mtree_idx['HaloID(1)'][10]
    npt.assert_allclose( expected, actual )

    expected = 11
    actual = self.ahf_reader.ahf_mtree_idx['HaloID(2)'][10]
    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_get_mtree_halo_files( self ):

    self.ahf_reader.get_mtree_halos( 'snum' )

    # Halo mass at z=0 for mtree_halo_id = 0
    expected = 7.5329e+11
    actual = self.ahf_reader.mtree_halos[0]['Mvir'][600]
    npt.assert_allclose( expected, actual )

    # ID at an early snapshot (snap 10) for halo file 0
    expected = 10
    actual = self.ahf_reader.mtree_halos[0]['ID'][10]
    npt.assert_allclose( expected, actual )

    # ID at snapshot 30 for halo file 2
    expected = 60
    actual = self.ahf_reader.mtree_halos[2]['ID'][30]
    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_get_mtree_halo_files_explicit_snapshot_number( self ):

    # Set the directory to the correct one for FIRE-1 data
    self.ahf_reader.sdir = sdir2

    self.ahf_reader.get_mtree_halos( 440 )

    # Halo mass at z=0 for mtree_halo_id = 0
    expected = 7.95197e+11
    actual = self.ahf_reader.mtree_halos[0]['Mvir'][440]
    npt.assert_allclose( expected, actual )

    # ID at an early snapshot (snap 30) for halo file 0
    expected = 13
    actual = self.ahf_reader.mtree_halos[0]['ID'][30]
    npt.assert_allclose( expected, actual )

    # ID at snapshot 30 for halo file 2
    expected = 15
    actual = self.ahf_reader.mtree_halos[2]['ID'][30]
    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_get_mtree_halo_quantity( self ):

    self.ahf_reader.get_mtree_halos( 'snum' )

    actual = self.ahf_reader.get_mtree_halo_quantity( 'ID', 600, 'snum' )
    expected = np.array([ 0, 1, 2, 10 ])

    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_get_mtree_halo_quantity_low_redshift( self ):

    self.ahf_reader.get_mtree_halos( 'snum' )

    actual = self.ahf_reader.get_mtree_halo_quantity( 'Mvir', 5, 'snum' )
    expected = np.array([ 0., 5.731000e+07, 0., 0. ])

    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_mtree_halo_id_matches( self ):
    '''Test that the ID in the mtree_halo_id is exactly what we expect it to be.'''

    self.ahf_reader.get_mtree_halos( 'snum' )
    halo_id = self.ahf_reader.mtree_halos[10]['ID'][500]

    # First make sure we have the right ID
    assert halo_id == 11 # just looked this up manually.

    # Now make sure we have the right x position, as a check
    expected = 28213.25906375 # Halo 11 X position at snum 500

    self.ahf_reader.get_halos( 500 )
    actual = self.ahf_reader.ahf_halos['Xc'][ halo_id ]

    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_get_pos_or_vel( self ):

    # Snap 550, mt halo 0, position
    expected = np.array([ 29372.26565053,  30929.16894187,  32415.81701217])

    # Get the values
    self.ahf_reader.get_mtree_halos( 'snum' )
    actual = self.ahf_reader.get_pos_or_vel( 'pos', 0, 550 )
    
    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_get_pos_or_vel_ahf_halos( self ):

    # Snap 550, halo 2, position
    expected = np.array([ 28687.97011391,  32183.71865825,  32460.26607195 ])

    # Get the values
    actual = self.ahf_reader.get_pos_or_vel( 'pos', 2, 550, 'ahf_halos' )
    
    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_save_and_load_mtree_halo_files( self ):

    self.ahf_reader.get_mtree_halos( 'snum' )
    expected = self.ahf_reader.mtree_halos[10]['ID']
    expected_detail = self.ahf_reader.mtree_halos[0]['ID'][500]

    # Save
    save_tag = 'test'
    self.ahf_reader.save_mtree_halos( save_tag )

    # Load
    new_ahf_reader = read_ahf.AHFReader( sdir )
    new_ahf_reader.get_mtree_halos( 'snum', tag=save_tag )
    actual = new_ahf_reader.mtree_halos[10]['ID']
    actual_detail = new_ahf_reader.mtree_halos[0]['ID'][500]

    npt.assert_allclose( expected, actual )
    npt.assert_allclose( expected_detail, actual_detail )

  ######################################################################

  def test_get_mtree_halo_files_multiple_files( self ):
    '''Take care of the issue where if halo_00000.dat and halo_00000_test.dat are in the same directory,
    then the loading function breaks.
    '''

    halo_0_filepath = './tests/data/ahf_test_data/halo_00000.dat'
    halo_0_filepath_test = './tests/data/ahf_test_data/halo_00000_test.dat'
    os.system( 'cp {} {}'.format( halo_0_filepath, halo_0_filepath_test ) )

    self.ahf_reader.get_mtree_halos( 'snum' )

  ########################################################################

  def test_get_accurate_redshift( self ):

    expected = 0.00015930

    self.ahf_reader.get_mtree_halos( 'snum' )

    self.ahf_reader.get_accurate_redshift( './tests/data/sdir' )

    actual = self.ahf_reader.mtree_halos[0]['redshift'][599]

    npt.assert_allclose( expected, actual, )

  ########################################################################

  def test_smooth_mtree_halos( self ):
    
    self.ahf_reader.get_mtree_halos( 'snum' )

    self.ahf_reader.smooth_mtree_halos( data_sdir )

    # Test that the redshift worked.
    redshift_expected_598 = 0.00031860
    redshift_actual = self.ahf_reader.mtree_halos[0]['redshift'][598]
    npt.assert_allclose( redshift_expected_598, redshift_actual )

    # Test that r_vir worked (that we never make a dip down in the early snapshot)
    r_vir_min = self.ahf_reader.mtree_halos[0]['Rvir'][210].min()
    assert r_vir_min > 30.

    # Test that m_vir worked (that we never make a dip down in the early snapshot)
    m_vir_min = self.ahf_reader.mtree_halos[0]['Mvir'][210].min()
    assert m_vir_min > 1e11

  ########################################################################

  def test_smooth_mtree_halos_fire1( self ):
    
    # Get the right directory
    self.ahf_reader.sdir = sdir2
    self.ahf_reader.get_mtree_halos( 440 )

    self.ahf_reader.smooth_mtree_halos( data_sdir2 )

    # Test that the redshift worked.
    redshift_expected_439 = 0.0049998743750157
    redshift_actual = self.ahf_reader.mtree_halos[0]['redshift'][439]
    npt.assert_allclose( redshift_expected_439, redshift_actual )

    # Test that r_vir worked (that we never make a dip down in the early snapshot)
    r_vir_min = self.ahf_reader.mtree_halos[0]['Rvir'][326].min()
    assert r_vir_min > 100.

    # Test that m_vir worked (that we never make a dip down in the early snapshot)
    m_vir_min = self.ahf_reader.mtree_halos[0]['Mvir'][326].min()
    assert m_vir_min > 1e11

  ########################################################################

  @slow
  def test_save_smooth_mtree_halos( self ):

    # Get the results
    self.ahf_reader.save_smooth_mtree_halos( data_sdir, 'snum', )

    # Load the saved files
    self.ahf_reader.get_mtree_halos( 'snum', 'smooth' )

    # Test that the redshift worked.
    redshift_expected_598 = 0.00031860
    redshift_actual = self.ahf_reader.mtree_halos[0]['redshift'][598]
    npt.assert_allclose( redshift_expected_598, redshift_actual )

    # Test that r_vir worked (that we never make a dip down in the early snapshot)
    r_vir_min = self.ahf_reader.mtree_halos[0]['Rvir'][210].min()
    assert r_vir_min > 30.

    # Test that m_vir worked (that we never make a dip down in the early snapshot)
    m_vir_min = self.ahf_reader.mtree_halos[0]['Mvir'][210].min()
    assert m_vir_min > 1e11

  ########################################################################

  def test_save_smooth_mtree_halos_different_snum( self ):
  
    # Pass it the right directory
    self.ahf_reader.sdir = sdir2

    # Get the results
    self.ahf_reader.save_smooth_mtree_halos( data_sdir2, 440, False )

    # Load the saved files
    self.ahf_reader.get_mtree_halos( 440, 'smooth' )

    # Test that the redshift worked.
    redshift_expected_439 = 0.0049998743750157
    redshift_actual = self.ahf_reader.mtree_halos[0]['redshift'][439]
    npt.assert_allclose( redshift_expected_439, redshift_actual )

    # Test that r_vir worked (that we never make a dip down in the early snapshot)
    r_vir_min = self.ahf_reader.mtree_halos[0]['Rvir'][210].min()
    assert r_vir_min > 30.

    # Test that m_vir worked (that we never make a dip down in the early snapshot)
    m_vir_min = self.ahf_reader.mtree_halos[0]['Mvir'][210].min()
    assert m_vir_min > 1e11

    # Make sure there aren't NaN's
    assert self.ahf_reader.mtree_halos[0]['Rvir'][440] > 0

  ########################################################################

  @slow
  def test_get_analytic_concentration_mtree_halos( self ):

    # Load the saved files
    self.ahf_reader.get_mtree_halos( 'snum', )
    self.ahf_reader.get_analytic_concentration( data_sdir )

    c_vir_z0_expected = 10.66567139
    c_vir_z0_actual = self.ahf_reader.mtree_halos[0]['cAnalytic'][600]

    npt.assert_allclose( c_vir_z0_expected, c_vir_z0_actual )

  ########################################################################

  @slow
  def test_save_and_load_ahf_halos_add( self ):

    # Save halos_add
    self.ahf_reader.save_ahf_halos_add( 600, data_sdir )

    # Load halos_add
    self.ahf_reader.get_halos_add( 600 )

    # Check that we calculated the concentration correctly.
    c_vir_z0_expected = 10.66567139
    c_vir_z0_actual = self.ahf_reader.ahf_halos_add['cAnalytic'][0]
    npt.assert_allclose( c_vir_z0_expected, c_vir_z0_actual )

  ########################################################################

  @patch( 'galaxy_diver.read_data.ahf.AHFReader.save_ahf_halos_add' )
  def test_save_multiple_ahf_halos_adds( self, mock_save_ahf_halos_add ):

    self.ahf_reader.save_multiple_ahf_halos_adds( data_sdir, 500, 600, 50 )

    calls = [
      call(500, data_sdir),
      call(550, data_sdir),
      call(600, data_sdir),
      ]
    mock_save_ahf_halos_add.assert_has_calls( calls, any_order=True )

########################################################################
########################################################################

class TestUtilities( unittest.TestCase ):

  def setUp( self ):

    self.ahf_reader = read_ahf.AHFReader( sdir )

  ########################################################################

  def test_check_files_exist( self ):

    assert self.ahf_reader.check_files_exist( 500, 600, 50 )

  ########################################################################

  def test_check_files_exist_missing_files( self ):

    with utilities.captured_output() as ( out, err ):
      assert not self.ahf_reader.check_files_exist( 400, 600, 50 )

    # Make sure we print information about missing snapshots
    output = out.getvalue().strip()
    self.assertEqual( output, 'Missing snums:\n400, 450,' )


