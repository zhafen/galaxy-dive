#!/usr/bin/env python
'''Testing for trends.cgm.py

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

import galaxy_dive.trends.cgm as cgm_trends

########################################################################
########################################################################

class TestCGMTrends( unittest.TestCase ):

  ########################################################################

  def test_cooling_time( self ):
    '''Just test that we're reading the data correctly.
    '''

    actual = cgm_trends.cooling_time(
        0.70057054,
        1.45,
        'm12m',
        'core',
    )
    # From manually looking at the data
    expected = 0.0006051895628244133

    npt.assert_allclose( expected, actual )
        
    

