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
import unyt

import galaxy_dive.trends.galaxy as gal_trends

########################################################################
########################################################################

class TestGalaxyTrends( unittest.TestCase ):

    def test_dynamical_friction_time( self ):

        result = gal_trends.dynamical_friction_time(
            ang_mom = 100. * unyt.kpc * 200. * unyt.km / unyt.s,
            r_c = 100. * unyt.kpc,
            v_c = 100. * unyt.km / unyt.s,
            mass = 1e6 * unyt.msun,
            sigma = 50. * unyt.km / unyt.s,
            m_enc = 1e11 * unyt.msun,
        )

        result.to( 'Gyr' )

