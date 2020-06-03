'''Testing for data_management.py
'''

from mock import patch
import numpy as np
import numpy.testing as npt
import os
import unittest

import galaxy_dive.data_management.trove_management as trove_management

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

    def test_combinations_chosen_order( self ):

        self.trove_manager.combinations

        self.trove_manager.set_order( [ 0, 1, 2, 3 ] )

        actual = self.trove_manager.combinations

        expected = [
            ( './tests/data/trove_test_dir', 'a', 1, 6, ),
            ( './tests/data/trove_test_dir', 'b', 1, 6, ),
            ( './tests/data/trove_test_dir', 'a', 2, 6, ),
            ( './tests/data/trove_test_dir', 'b', 2, 6, ),
            ( './tests/data/trove_test_dir', 'a', 3, 6, ),
            ( './tests/data/trove_test_dir', 'b', 3, 6, ),
        ]

        self.assertEqual( expected, actual )

        self.trove_manager.set_order( [ 3, 2, 0, 1 ] )

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

        tm = trove_management.TroveManager(
            file_format2,
            [ 'a', 'b', ],
            [ 0, 1, ],
            [ 10, 20, 30, ],
        )

        def get_file( self, *args ):
            return self.file_format.format( *args )
        tm.get_file = get_file

        tm.set_order( [ 2, 0, 1,] )

        actual = tm.combinations

        expected = [
            ( 'a', 0, 10 ),
            ( 'a', 1, 10 ),
            ( 'a', 0, 20 ),
            ( 'a', 1, 20 ),
            ( 'a', 0, 30 ),
            ( 'a', 1, 30 ),
            ( 'b', 0, 10 ),
            ( 'b', 1, 10 ),
            ( 'b', 0, 20 ),
            ( 'b', 1, 20 ),
            ( 'b', 0, 30 ),
            ( 'b', 1, 30 ),
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

    ########################################################################

    def test_get_next_args_to_use_skipped_args( self ):

        self.trove_manager.combinations_to_skip.append(
            ( './tests/data/trove_test_dir', 'a', 2, 6, )
        )

        actual = self.trove_manager.get_next_args_to_use()

        expected = ( './tests/data/trove_test_dir', 'b', 1, 6, )

        self.assertEqual( expected, actual )
