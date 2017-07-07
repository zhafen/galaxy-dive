#!/usr/bin/env python
'''Testing for read_metafile.py

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import numpy as np
import numpy.testing as npt
import unittest

import galaxy_diver.read_data.metafile as read_metafile

sdir = './tests/test_data/test_sdir'
sdir2 = './tests/test_data/test_sdir2'

########################################################################

class TestMetafileReader( unittest.TestCase ):

  def setUp( self ):

    self.metafile_reader = read_metafile.MetafileReader( sdir )

  ########################################################################

  def test_get_snapshot_times( self ):

    expected = 13.55350759 # Time in Gyr for snapshot 580

    self.metafile_reader.get_snapshot_times()

    actual = self.metafile_reader.snapshot_times['time[Gyr]'][580]

    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_get_snapshot_times_old_filename( self ):

    expected = 0.0049998745585804194

    # Give it the right directory
    self.metafile_reader.sdir = sdir2

    self.metafile_reader.get_snapshot_times()

    actual = self.metafile_reader.snapshot_times['redshift'][439]

    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_get_used_parameters( self ):

    OmegaBaryon_expected = 0.0455

    self.metafile_reader.get_used_parameters()

    OmegaBaryon_actual = float( self.metafile_reader.used_parameters['OmegaBaryon'] )

    npt.assert_allclose( OmegaBaryon_expected, OmegaBaryon_actual )
