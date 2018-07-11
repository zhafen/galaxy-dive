#!/usr/bin/env python
'''Testing for read_rockstar.py

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

import galaxy_diver.read_data.rockstar as read_rockstar
import galaxy_diver.utils.utilities as utilities

########################################################################

# Decorator for skipping slow tests
slow = pytest.mark.skipif(
    not pytest.config.getoption("--runslow"),
    reason="need --runslow option to run"
)

########################################################################
########################################################################

class TestRockstarReader( unittest.TestCase ):

  def setUp( self ):

    self.rockstar_reader = read_rockstar.RockstarReader(
        './tests/data/rockstar_test_data',
    )

  ########################################################################

  def test_get_halos( self ):

    self.rockstar_reader.get_halos( 600 )

    expected = 51
    actual = self.rockstar_reader.halos['Np'][6723]
    npt.assert_allclose( expected, actual )

