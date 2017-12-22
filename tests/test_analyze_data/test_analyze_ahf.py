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

import galaxy_diver.read_data.ahf as read_ahf
import galaxy_diver.analyze_data.ahf as analyze_ahf
import galaxy_diver.analyze_data.particle_data as particle_data
import galaxy_diver.galaxy_finder.finder as gal_finder
import galaxy_diver.utils.utilities as utilities

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

    self.halo_data = analyze_ahf.HaloData( sdir, tag='smooth' )

  ########################################################################

  def test_mt_halos( self ):
    '''Test we can load mt_halos.'''

    self.halo_data.mt_halos

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
########################################################################

class TestKeyParser( unittest.TestCase ):

  def setUp( self ):

    self.ahf_data = analyze_ahf.HaloData( sdir )

  ########################################################################

  def test_radius_key( self ):
    '''Test that we can get a radius key, given a length scale and a multiplier.
    '''

    actual = self.ahf_data.key_parser.get_radius_key( 1.5, 'Rvir' )
    expected = '1.5Rvir'

    self.assertEqual( expected, actual )

  ########################################################################

  def test_radius_label_multiplier_1( self ):
    '''Test that we can get a radius key, given a length scale and a multiplier.
    In this case, test we get a clean result out when multiplier = 1.
    '''

    actual = self.ahf_data.key_parser.get_radius_key( 1., 'Rvir' )
    expected = 'Rvir'

    self.assertEqual( expected, actual )






    







