'''Testing for data_management.py
'''

from mock import patch
import numpy as np
import numpy.testing as npt
import os
import unittest

import galaxy_diver.data_management.trove_management as trove_management

########################################################################

data_dir = './tests/data/trove_test_dir'
file_format = 'test_{}_{}.hdf5'
file_format2 = 'test_{}_{}_{}.hdf5'
args_a = [ 'a', 'b', ]
args_b = [ 1, 2, 3, ]
args_c = [ 6, ]

########################################################################
########################################################################

class TestTroveManagerInit( unittest.TestCase ):

    def test_init( self ):
        '''Test that we can even initialize.'''

        trove_manager = trove_management.TroveManager(
            data_dir,
            file_format,
            args_a, 
            args_b,
        )

        self.assertEqual( data_dir, trove_manager.data_dir )

        self.assertEqual( file_format, trove_manager.file_format )

        self.assertEqual( args_a, trove_manager.args[0] )
        self.assertEqual( args_b, trove_manager.args[1] )

########################################################################
########################################################################

class TestTroveManager( unittest.TestCase ):

    def setUp( self ):

        self.trove_manager = trove_management.TroveManager(
            data_dir,
            file_format,
            args_a, 
            args_b,
            args_c,
        )

    ########################################################################

    def test_get_combinations( self ):

        actual = self.trove_manager.combinations

        expected = [
            ( 'a', 1, 6, ),
            ( 'a', 2, 6, ),
            ( 'a', 3, 6, ),
            ( 'b', 1, 6, ),
            ( 'b', 2, 6, ),
            ( 'b', 3, 6, ),
        ]

        self.assertEqual( expected, actual )
        

        

