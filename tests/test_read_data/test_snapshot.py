#!/usr/bin/env python
'''Testing for read_data.snapshot.py

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

from mock import call, patch
import numpy as np
import numpy.testing as npt
import os
import pdb
import pytest
import unittest

import galaxy_diver.read_data.snapshot as read_snapshot

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

    actual = read_snapshot.get_snapshot_filepath( sdir, snum )
    expected = './tests/data/sdir4/output/snapshot_600.hdf5'

    assert expected == actual

  ########################################################################

  def test_get_multiple_files( self ):

    sdir = './tests/data/sdir/output'
    snum = 600

    actual = read_snapshot.get_snapshot_filepath( sdir, snum )
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

    self.assertRaises( NameError, read_snapshot.get_snapshot_filepath, sdir, snum )

  ########################################################################

