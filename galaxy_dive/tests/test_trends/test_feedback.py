#!/usr/bin/env python
'''Testing for trends.feedback.py

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

import galaxy_dive.trends.feedback as feedback

########################################################################
########################################################################

class TestSpecificEnergyRate( unittest.TestCase ):

    def test_all_run( self ):

        feedback_sources = [
            'all',
            'mechanical',
            'radiation',
            'heating and ionizing radiation',
            'SNe',
            'SNe Ia',
            'SNe II',
            'stellar wind',
            'bolometric radiation',
            'mid/far IR radiation',
            'optical/near IR radiation',
            'photo-electric FUV radiation',
            'ionizing radiation',
            'NUV radiation',
        ]

        age = np.linspace( 0, 100, 10 )

        for fb_source in feedback_sources:
            feedback.feedback_specific_energy_rate( age, fb_source )
            
########################################################################

class TestEnergyRate( unittest.TestCase ):

    def test_no_nans( self ):

        time = np.array([0.0005, 0.0015, 0.0025]) * 1e3
        sfh = np.array([0., 0., 0.])

        Edot = feedback.feedback_energy_rate_given_sfh(
            time,
            sfh,
            'mechanical',
        )

        assert np.isnan( Edot ).sum() == 0 
