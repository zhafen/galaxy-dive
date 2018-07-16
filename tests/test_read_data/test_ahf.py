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
    actual = self.ahf_reader.halos['numSubStruct'][0]
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
    self.ahf_reader.data_dir = sdir2

    self.ahf_reader.get_mtree_halos( index=440 )

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
    actual = self.ahf_reader.halos['Xc'][ halo_id ]

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

  def test_get_pos_or_vel_halos( self ):

    # Snap 550, halo 2, position
    expected = np.array([ 28687.97011391,  32183.71865825,  32460.26607195 ])

    # Get the values
    actual = self.ahf_reader.get_pos_or_vel( 'pos', 2, 550, 'halos' )
    
    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_get_mtree_halo_files_multiple_files( self ):
    '''Take care of the issue where if halo_00000.dat and halo_00000_test.dat are in the same directory,
    then the loading function breaks.
    '''

    halo_0_filepath = './tests/data/ahf_test_data/halo_00000.dat'
    halo_0_filepath_test = './tests/data/ahf_test_data/halo_00000_test.dat'
    os.system( 'cp {} {}'.format( halo_0_filepath, halo_0_filepath_test ) )

    self.ahf_reader.get_mtree_halos( 'snum' )

  ########################################################################

  def test_get_mtree_halo_files_default_index( self ):
    '''Test that, if the data contains a 'snum' column, the default is to use it as an index.'''

    self.ahf_reader.get_mtree_halos( tag='smooth' )

    expected = np.arange( 600, 6, -1 )
    actual = self.ahf_reader.mtree_halos[0].index
    npt.assert_allclose( expected, actual )

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
    self.assertEqual( output, 'Missing snums:\n400,\n450,' )


