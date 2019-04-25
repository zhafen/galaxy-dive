#!/usr/bin/env python
'''Testing for analyze_data.ahf.py

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import glob
from mock import call, patch, PropertyMock
import numpy as np
import numpy.testing as npt
import os
import pdb
import pytest
import unittest

import galaxy_dive.read_data.ahf as read_ahf
import galaxy_dive.analyze_data.halo_data as halo_data
import galaxy_dive.analyze_data.particle_data as particle_data
import galaxy_dive.galaxy_linker.linker as gal_linker
import galaxy_dive.utils.utilities as utilities

sdir = './tests/data/analysis_dir'
sdir2 = './tests/data/analysis_dir2'
data_sdir = './tests/data/sdir'
data_sdir2 = './tests/data/sdir2'

########################################################################

# Decorator for skipping slow tests
slow = pytest.mark.skipif(
        not pytest.config.getoption("--runslow"),
        reason="need --runslow option to run"
)

class SaneEqualityArray(np.ndarray):
    '''Numpy array subclass that allows you to test if two arrays are equal.'''

    def __eq__(self, other):
            return (isinstance(other, np.ndarray) and self.shape == other.shape and np.allclose(self, other))

def sane_eq_array(list_in):
    '''Wrapper for SaneEqualityArray, that takes in a list.'''

    arr = np.array(list_in)

    return arr.view(SaneEqualityArray)

########################################################################
########################################################################

class TestHaloData( unittest.TestCase ):

    def setUp( self ):

        self.halo_data = halo_data.HaloData(
            sdir,
            mt_kwargs = { 'tag' : 'smooth' },
        )

    ########################################################################

    def test_mtree_halos( self ):
        '''Test we can load mtree_halos.'''

        self.halo_data.mtree_halos

    ########################################################################

    def test_get_mt_data( self ):
        '''Test that we can get data from the merger tree.'''

        result = self.halo_data.get_mt_data( 'Rvir' )

        actual = result[0]
        expected = 188.14
        npt.assert_allclose( expected, actual )

        actual = result[-1]
        expected = 12.95
        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_get_mt_data_including_a( self ):
        '''Test that we can get data from the merger tree, including multiplying by a to some power, for unit conversion'''

        result = self.halo_data.get_mt_data( 'Rvir', a_power=1., )

        actual = result[0]
        expected = 188.14
        npt.assert_allclose( expected, actual )

        actual = result[-1]
        expected = 12.95/(1. + 13.206278 )
        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_get_profile_data( self ):

        # We should get the virial mass when at the virial radius
        actual = self.halo_data.get_profile_data(
            'M_in_r',
            600,
            r = self.halo_data.get_mt_data(
                'Rvir',
                snums = [ 600 ],
            ),
        )
        expected = self.halo_data.get_mt_data( 'Mvir', snums=[ 600 ], )
        npt.assert_allclose( expected, actual, rtol=0.1 )

        # We shouldn't get anything reasonable when too small or too large
        actual = self.halo_data.get_profile_data(
            'M_in_r',
            600,
            r = np.array([ 0., np.inf ]),
        )
        expected = np.full( 2, np.nan )
        npt.assert_allclose( expected, actual, )

        # And this shouldn't fail for the highest halo stored too
        actual = self.halo_data.get_profile_data(
            'M_in_r',
            600,
            mt_halo_id = 3,
            r = 50.,
        )

    ########################################################################

    def test_get_enclosed_mass( self ):

        # Somehow the files got mixed up, so we're changing the analysis dir
        # to one with matching files
        self.halo_data.data_dir = './tests/data/analysis_dir6'
        self.halo_data.data_reader = read_ahf.AHFReader( self.halo_data.data_dir )

        # Setup inputs
        radius = self.halo_data.get_mt_data(
            'Rvir',
            snums = [ 550 ],
            a_power = 1.,
        ) / .702
        positions = np.array([
            [ 1., 0, 0, ],
            [ 1./np.sqrt(3), 1./np.sqrt(3), 1./np.sqrt(3) ],
            [ 0., 0., 1., ],
            [ 0., 0., 0., ],
        ]) * radius
        # Position of halo ID at this redshift
        positions += np.array([39184.23739265, 41173.71952049, 43151.66589124])

        actual = self.halo_data.get_enclosed_mass(
            positions = positions,
            snum = 550,
            hubble_param = .702,
        )
        m_vir = self.halo_data.get_mt_data( 'Mvir', snums=[ 550 ], )[0] / .702
        expected = np.array([ m_vir, m_vir, m_vir, np.nan ])
        npt.assert_allclose( expected, actual, rtol=0.1 )

########################################################################
########################################################################

class TestKeyParser( unittest.TestCase ):

    def setUp( self ):

        self.halo_data = halo_data.HaloData( sdir )

    ########################################################################

    def test_radius_key( self ):
        '''Test that we can get a radius key, given a length scale and a multiplier.
        '''

        actual = self.halo_data.key_parser.get_radius_key( 1.5, 'Rvir' )
        expected = '1.5Rvir'

        self.assertEqual( expected, actual )

    ########################################################################

    def test_radius_label_multiplier_1( self ):
        '''Test that we can get a radius key, given a length scale and a multiplier.
        In this case, test we get a clean result out when multiplier = 1.
        '''

        actual = self.halo_data.key_parser.get_radius_key( 1., 'Rvir' )
        expected = 'Rvir'

        self.assertEqual( expected, actual )














