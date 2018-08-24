'''Testing for mp_utils.py
'''

from mock import patch
import multiprocessing as mp
import numpy as np
import numpy.testing as npt
import os
import pdb
import subprocess
import unittest

import galaxy_dive.utils.mp_utils as mp_utils

########################################################################
########################################################################

class TestApplyAmongProcessors( unittest.TestCase ):

  def test_default( self ):

    d = {
      'a' : mp.Array( 'f', np.zeros( 10 ) ),
      'b' : mp.Array( 'f', np.zeros( 10 ) ),
    }
    all_args = [ ( i, d ) for i in range( 10 ) ]

    def f( i, d ):

      num = float( i + 1 )

      d['a'][i] = num**2.
      d['b'][i] = num*2.

    mp_utils.apply_among_processors( f, all_args, 2 )

    expected_d = {
      'a' : np.arange( 1, 11, dtype=float)**2.,
      'b' : np.arange( 1, 11, dtype=float)*2.,
    }

    for key in expected_d.keys():
      expected = expected_d[key]
      actual = d[key]

      npt.assert_allclose( expected, actual )
    
########################################################################
########################################################################

class TestMPQueueToList( unittest.TestCase ):

  def test_default( self ):

    q = mp.Queue()
    [ q.put( i ) for i in range( 4 ) ]
    
    actual = mp_utils.mp_queue_to_list( q, 2 )
    expected = range( 4 )

    npt.assert_allclose( sorted( expected ), sorted( actual ) )

  ########################################################################

  def test_long_odd_list( self ):

    q = mp.Queue()
    [ q.put( i ) for i in range( 7 ) ]
    
    actual = mp_utils.mp_queue_to_list( q, 2 )
    expected = range( 7 )

    npt.assert_allclose( sorted( expected ), sorted( actual ) )

  ########################################################################

  def test_more_processors( self ):

    q = mp.Queue()
    [ q.put( i ) for i in range( 2 ) ]
    
    actual = mp_utils.mp_queue_to_list( q, 4 )
    expected = range( 2 )

    npt.assert_allclose( sorted( expected ), sorted( actual ) )

  ########################################################################

  def test_single_object( self ):

    q = mp.Queue()
    [ q.put( i ) for i in range( 1 ) ]
    
    actual = mp_utils.mp_queue_to_list( q, 2 )
    expected = range( 1 )

    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_queue_of_sets( self ):

    q = mp.Queue()

    ls = [ range( i, i + 4 ) for i in range( 3 ) ]
    sets = [ set( l ) for l in ls ]
    [ q.put( s ) for s in sets ]
    
    actual = mp_utils.mp_queue_to_list( q, 2 )
    expected = sets

    actual_sorted = sorted( [ list( s ) for s in actual ] )
    expected_sorted = sorted( [ list( s ) for s in expected ] )

    npt.assert_allclose( expected_sorted, actual_sorted )
