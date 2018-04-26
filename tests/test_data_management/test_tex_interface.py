'''Testing for data_management.py
'''

from mock import patch
import numpy as np
import numpy.testing as npt
import os
import unittest

import galaxy_diver.data_management.tex_interface as tex_interface

########################################################################

filename = './tests/data/tex_test_dir/analysis_output.tex'

########################################################################
########################################################################

class TestTeXVariableFile( unittest.TestCase ):

    def setUp( self ):

        self.tex_vfile = tex_interface.TeXVariableFile( filename )

    ########################################################################
    
    def test_data_dict( self ):

        expected = {
            'a' : '1',
            'b' : '-100',
        }

        actual = self.tex_vfile.data_dict

        #DEBUG
        import pdb; pdb.set_trace()

        self.assertEqual( expected, actual )
