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

class TestReadTeXVariableFile( unittest.TestCase ):

    def setUp( self ):

        self.tex_vfile = tex_interface.TeXVariableFile( filename )

    ########################################################################
    
    def test_data_dict( self ):

        expected = {
            'a' : '1',
            'b' : '-100',
        }

        actual = self.tex_vfile.data_dict

        self.assertEqual( expected, actual )

########################################################################
########################################################################


class TestTeXVariableFileReset( unittest.TestCase ):

    def setUp( self ):

        self.tex_vfile = tex_interface.TeXVariableFile( filename )

    def tearDown( self ):

        os.remove( filename )

        with open( filename, 'w' ) as f:
            f.write( '\\newcommand{\\a}{1}\n\\newcommand{\\b}{-100}\n' )

    ########################################################################

    def test_write_data( self ):

        self.tex_vfile._data_dict = {
            'a' : '3',
            'b' : '2',
            'c' : '1',
        }

        self.tex_vfile.write()

        with open( filename, 'r' ) as f:
            actual = f.read()

        expected  = '\\newcommand{\\a}{3}\n\\newcommand{\\c}{1}\n\\newcommand{\\b}{2}\n'

        self.assertEqual( expected, actual )

    ########################################################################

    def test_save_variable( self ):

        self.tex_vfile.save_variable( 'c', '9000' )

        with open( filename, 'r' ) as f:
            actual = f.read()

        expected  = '\\newcommand{\\a}{1}\n\\newcommand{\\c}{9000}\n\\newcommand{\\b}{-100}\n'

        self.assertEqual( expected, actual )

        
