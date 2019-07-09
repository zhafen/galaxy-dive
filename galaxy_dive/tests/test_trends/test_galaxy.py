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

import galaxy_dive.trends.galaxy as gal_trends

########################################################################
########################################################################

class TestGalaxyTrends( unittest.TestCase ):

    def test_dynamical_friction_time( self ):

        result = gal_trends.dynamical_friction_time(
            energy,
            ang_mom,
            mass,
        )

        assert result.unyt.dimensions == unyt.dimensions.time
