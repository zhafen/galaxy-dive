#!/usr/bin/env python
'''Testing for read_data.snapshot.py

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import h5py
from mock import call, patch
import numpy as np
import numpy.testing as npt
import os
import pdb
import pytest
import unittest

import galaxy_dive.read_data.snapshot as read_snapshot

########################################################################

# Decorator for skipping slow tests
slow = pytest.mark.skipif(
    not pytest.config.getoption("--runslow"),
    reason="need --runslow option to run"
)

########################################################################

class TestGetSnapshotFilepath( unittest.TestCase ):

  def test_get_single_file( self ):

    sdir = './tests/data/sdir4/output'
    snum = 600

    actual = read_snapshot.get_snapshot_filepaths( sdir, snum )[0]
    expected = './tests/data/sdir4/output/snapshot_600.hdf5'

    assert expected == actual

  ########################################################################

  def test_get_multiple_files( self ):

    sdir = './tests/data/sdir/output'
    snum = 600

    actual = read_snapshot.get_snapshot_filepaths( sdir, snum )
    expected = [ './tests/data/sdir/output/snapdir_600/snapshot_600.0.hdf5',
                 './tests/data/sdir/output/snapdir_600/snapshot_600.1.hdf5',
                 './tests/data/sdir/output/snapdir_600/snapshot_600.2.hdf5',
                 './tests/data/sdir/output/snapdir_600/snapshot_600.3.hdf5' ]

    # I don't care what order, just give me the names!
    assert set( expected ) == set( actual )

  ########################################################################

  def test_fails_for_no_snapshot( self ):

    sdir = './tests/data/sdir4/output'
    snum = 100

    self.assertRaises( NameError, read_snapshot.get_snapshot_filepaths, sdir, snum )

########################################################################
########################################################################

class TestLoadSnapshot( unittest.TestCase ):

  def test_reads_potential( self ):

    sdir = './tests/data/sdir/output'
    snum = 600

    data = read_snapshot.readsnap( sdir=sdir, snum=snum, ptype=0 )

    expected_potential = 146365.1875
    npt.assert_allclose( expected_potential, data['potential'][-3] )

########################################################################
########################################################################

# TODO: If I start making my own readsnap routine again, these tests need to be done.
#class TestReadSnapshotFiles( unittest.TestCase ):
#
#  def test_single_file( self ):
#
#    actuals = read_snapshot.read_snapshot_files( [ './tests/data/sdir4/output/snapshot_600.hdf5', ] )
#    actual_keys = [
#      'P',
#      'Den',
#      'U',
#      'M',
#      'Z',
#      'XHI',
#      'ChildID',
#      'IDGen',
#      'ID',
#      'V',
#      ]
#      
#    expecteds = h5py.File( './tests/data/sdir4/output/snapshot_600.hdf5' )
#    expected_keys = [
#      u'Coordinates',
#      u'Density',
#      u'InternalEnergy',
#      u'Masses',
#      u'Metallicity',
#      u'NeutralHydrogenAbundance',
#      u'ParticleChildIDsNumber',
#      u'ParticleIDGenerationNumber',
#      u'ParticleIDs',
#      u'Velocities',
#      ]
#
#    for expected_key, actual_key in zip( expected_keys, actual_keys ):
#
#      expected = exepecteds[expected_key]
#      actual = actuals[actual_keys]
#
#      npt.assert_allclose( expected, actual )
#
#      assert expected.dtype == actual.dtype
#
#  ########################################################################
#
#  def test_different_ptypes( self ):
#
#    assert False, "Need to do."
#
#  ########################################################################














