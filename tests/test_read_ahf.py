#!/usr/bin/env python
'''Testing for tracking.py

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import glob
import numpy as np
import numpy.testing as npt
import os
import pdb
import unittest

from galaxy_diver import read_ahf

sdir = './tests/test_data/ahf_test_data'

########################################################################

class TestAHFReader( unittest.TestCase ):

  def setUp( self ):

    self.ahf_reader = read_ahf.AHFReader( sdir )

    # Remove smooth halo files that are generated
    halo_filepaths = glob.glob( './tests/test_data/ahf_test_data/halo_*_smooth.dat' )
    for halo_filepath in halo_filepaths:
      if os.path.isfile( halo_filepath ):
        os.system( 'rm {}'.format( halo_filepath ) )

  ########################################################################

  def tearDown( self ):

    # Remove extra files that are generated
    halo_filepaths = glob.glob( './tests/test_data/ahf_test_data/halo_*_test.dat' )
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

  def test_get_mtree_halo_quantity( self ):

    self.ahf_reader.get_mtree_halos( 'snum' )

    actual = self.ahf_reader.get_mtree_halo_quantity( 'ID', 600, 'snum' )
    expected = np.arange( 0, 20 )

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

    halo_0_filepath = './tests/test_data/ahf_test_data/halo_00000.dat'
    halo_0_filepath_test = './tests/test_data/ahf_test_data/halo_00000_test.dat'
    os.system( 'cp {} {}'.format( halo_0_filepath, halo_0_filepath_test ) )

    self.ahf_reader.get_mtree_halos( 'snum' )

  ########################################################################

  def test_smooth_mtree_halos( self ):
    
    self.ahf_reader.get_mtree_halos( 'snum' )

    self.ahf_reader.smooth_mtree_halos()

    # Test that r_vir worked
    r_vir_expected_600 = 209.83000000000001
    r_vir_actual = self.ahf_reader.mtree_halos[0]['Rvir'][600]
    npt.assert_allclose( r_vir_expected_600, r_vir_actual )

    # Test that m_vir worked
    m_vir_expected_600 = 753290000000.0
    m_vir_actual = self.ahf_reader.mtree_halos[0]['Mvir'][600]
    npt.assert_allclose( m_vir_expected_600, m_vir_actual )

  ########################################################################

  def test_save_smooth_mtree_halos( self ):

    # Get the results
    self.ahf_reader.save_smooth_mtree_halos( 'snum' )

    # Load the saved files
    self.ahf_reader.get_mtree_halos( 'snum', 'smooth' )

    # Test that r_vir worked
    r_vir_expected_600 = 209.83000000000001
    r_vir_actual = self.ahf_reader.mtree_halos[0]['Rvir'][600]
    npt.assert_allclose( r_vir_expected_600, r_vir_actual )

    # Test that m_vir worked
    m_vir_expected_600 = 753290000000.0
    m_vir_actual = self.ahf_reader.mtree_halos[0]['Mvir'][600]
    npt.assert_allclose( m_vir_expected_600, m_vir_actual )
