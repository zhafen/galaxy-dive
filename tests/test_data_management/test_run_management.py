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
parameters_a = [ 'a', 'b', ]
parameters_b = [ 1, 2, 3, ]
parameters_c = [ 6, 5, 4, ]

########################################################################
########################################################################

class TestTroveManagerInit( unittest.TestCase ):

    def test_init( self ):
        '''Test that we can even initialize.'''

        trove_manager = trove_management.TroveManager(
            data_dir,
            file_format,
            parameters_a, 
            parameters_b,
        )

        self.assertEqual( data_dir, trove_manager.data_dir )

        self.assertEqual( file_format, trove_manager.file_format )

        self.assertEqual( parameters_a, trove_manager.args[0] )
        self.assertEqual( parameters_b, trove_manager.args[1] )

        

        

