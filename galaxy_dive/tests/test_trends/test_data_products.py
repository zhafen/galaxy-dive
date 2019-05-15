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

import galaxy_dive.trends.data_products as data_products

########################################################################
########################################################################

# Decorator for skipping tests unique to stampede
stampede = pytest.mark.skipif(
    not pytest.config.getoption("--runstampede"),
    reason="need --runstampede option to run"
)

@stampede
class TestDataProducts( unittest.TestCase ):

    def test_tidal_tensor_data_grudic( self ):
        '''Just test that we're reading the data correctly.
        '''

        actual = data_products.tidal_tensor_data_grudic(
            100,
            ids = np.array([ 48850256, 9550185, 1111111111111 ]),
        )
        # From manually looking at the data
        expected = np.array([
            [ 1.22668942e+03,  7.75438811e+01,  6.02076014e+02, -3.91015235e+01, -1.17870247e+02,  7.95706269e+01,  8.74140108e+01, 1.31098915e-01,  6.59692747e+03 ],
            [ 1.00920133e+01,  4.79382207e+01,  2.02634658e+01, 6.64207692e+00,  2.51895674e+01, -1.50385106e+01,  3.50740113e+01, 1.26965516e+00,  5.80723533e+01 ],
            [ np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, ],
        ])

        npt.assert_allclose( expected, actual.values )
            
    ########################################################################

    def test_tidal_tensor_data_grudic_no_matching_ids( self ):
        '''Test what happens when no IDs match.
        '''

        actual = data_products.tidal_tensor_data_grudic(
            100,
            ids = np.array([ 1111111111111, 1111111111112, 1111111111113 ]),
        )
        # From manually looking at the data
        expected = np.array([
            [ np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, ],
            [ np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, ],
            [ np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, ],
        ])

        npt.assert_allclose( expected, actual.values )
            
    ########################################################################

    def test_tidal_tensor_data_grudic_no_data( self ):
        '''Test what happens when a file doesn't exist
        '''

        actual = data_products.tidal_tensor_data_grudic(
            10000,
            ids = np.array([ 48850256, 9550185, 1111111111111 ]),
        )
        # From manually looking at the data
        expected = np.array([
            [ np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, ],
            [ np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, ],
            [ np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, ],
        ])

        npt.assert_allclose( expected, actual.values )
            
