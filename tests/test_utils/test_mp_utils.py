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

import galaxy_diver.utils.mp_utils as mp_utils

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
    
