'''Testing for data_management.py
'''

from mock import patch
import numpy as np
import numpy.testing as npt
import os
import unittest

import galaxy_diver.data_management.trove_management as trove_management

########################################################################

data_dirs = [ './tests/data/trove_test_dir' ]
file_format = 'test_{}_{}.dat'
file_format2 = 'test_{}_{}_{}.dat'
args_a = [ 'a', 'b', ]
args_b = [ 1, 2, 3, ]
args_c = [ 6, ]

########################################################################
########################################################################

class TestTroveManagerInit( unittest.TestCase ):

    def test_init( self ):
        '''Test that we can even initialize.'''

        trove_manager = trove_management.TroveManager(
            file_format,
            data_dirs,
            args_a, 
            args_b,
        )

        self.assertEqual( data_dirs, trove_manager.args[0] )

        self.assertEqual( file_format, trove_manager.file_format )

        self.assertEqual( args_a, trove_manager.args[1] )
        self.assertEqual( args_b, trove_manager.args[2] )

########################################################################
########################################################################

class TestTroveManager( unittest.TestCase ):

    def setUp( self ):

        self.trove_manager = trove_management.TroveManager(
            file_format2,
            data_dirs,
            args_a, 
            args_b,
            args_c,
        )

    ########################################################################

    def test_combinations( self ):

        actual = self.trove_manager.combinations

        expected = [
            ( './tests/data/trove_test_dir', 'a', 1, 6, ),
            ( './tests/data/trove_test_dir', 'a', 2, 6, ),
            ( './tests/data/trove_test_dir', 'a', 3, 6, ),
            ( './tests/data/trove_test_dir', 'b', 1, 6, ),
            ( './tests/data/trove_test_dir', 'b', 2, 6, ),
            ( './tests/data/trove_test_dir', 'b', 3, 6, ),
        ]

        self.assertEqual( expected, actual )

    ########################################################################

    def test_data_files( self ):

        actual = self.trove_manager.data_files

        expected = [
            './tests/data/trove_test_dir/test_a_1_6.dat',
            './tests/data/trove_test_dir/test_a_2_6.dat',
            './tests/data/trove_test_dir/test_a_3_6.dat',
            './tests/data/trove_test_dir/test_b_1_6.dat',
            './tests/data/trove_test_dir/test_b_2_6.dat',
            './tests/data/trove_test_dir/test_b_3_6.dat',
        ]

        self.assertEqual( expected, actual )

    ########################################################################
    def test_get_incomplete_combinations( self ):

        actual = self.trove_manager.get_incomplete_combinations()

        expected = [
            ( './tests/data/trove_test_dir', 'a', 2, 6, ),
            ( './tests/data/trove_test_dir', 'b', 1, 6, ),
            ( './tests/data/trove_test_dir', 'b', 3, 6, ),
        ]
        
        self.assertEqual( expected, actual )

    ########################################################################

    def test_get_incomplete_data_files( self ):

        actual = self.trove_manager.get_incomplete_data_files()

        expected = [
            './tests/data/trove_test_dir/test_a_2_6.dat',
            './tests/data/trove_test_dir/test_b_1_6.dat',
            './tests/data/trove_test_dir/test_b_3_6.dat',
        ]
        
        self.assertEqual( expected, actual )

    ########################################################################

    def test_get_next_args_to_use( self ):

        actual = self.trove_manager.get_next_args_to_use()

        expected = ( './tests/data/trove_test_dir', 'a', 2, 6, )

        self.assertEqual( expected, actual )
