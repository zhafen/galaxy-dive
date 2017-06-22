#!/usr/bin/env python
'''Testing for read_metafile.py

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import numpy as np
import numpy.testing as npt
import unittest

from galaxy_diver import read_metafile

sdir = './tests/test_data/test_sdir'

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
