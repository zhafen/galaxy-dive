'''Testing for utilities.py
'''

from mock import patch
import numpy as np
import numpy.testing as npt
import os
import pdb
import shutil
import subprocess
import unittest

import galaxy_dive.utils.utilities as utilities

########################################################################
########################################################################

class TestDictFromDefaultsAndVariations( unittest.TestCase ):

    def test_default( self ):

        defaults = { 'best cat' : 'Melvulu', }
        variations = {
            'person a' : { 'other best cat' : 'Chellbrat', },
            'person b' : {
                'best cat' : 'A Normal Melville Cat',
                'other best cat' : 'Chellcat',
            },
        }

        actual = utilities.dict_from_defaults_and_variations( defaults, variations )
        expected = {
            'person a' : {
                'best cat' : 'Melvulu',
                'other best cat' : 'Chellbrat',
            },
            'person b' : {
                'best cat' : 'A Normal Melville Cat',
                'other best cat' : 'Chellcat',
            },
        }

        for key in expected.keys():
            assert expected[key] == actual[key]

########################################################################
########################################################################

class TestGetCodeVersion( unittest.TestCase ):

    def test_default( self ):

        result = utilities.get_code_version( self )

        assert result is not None

    ########################################################################

    def test_works_in_otherdir( self ):

        cwd = os.getcwd()

        # Change directories
        os.chdir( '../' )

        result = utilities.get_code_version( self )

        assert result is not None

        # Change back
        os.chdir( cwd )

    ########################################################################

    def test_works_for_modules( self ):

        actual = utilities.get_code_version( utilities, instance_type='module' )

        expected = subprocess.check_output( [ 'git', 'describe', '--always'] )

        assert actual == expected

########################################################################
########################################################################

class TestGetInstanceSourceDir( unittest.TestCase ):

    def test_default( self ):

        actual = utilities.get_instance_source_dir( self )

        expected = os.path.join( os.getcwd(), 'tests/test_utils' )

        assert actual == expected

    ########################################################################

    def test_works_for_modules( self ):

        actual = utilities.get_instance_source_dir( utilities, 'module' )

        expected = os.path.join( os.getcwd(), 'utils' )

        assert actual == expected

########################################################################
########################################################################

class TestChunkList( unittest.TestCase ):

    def test_default( self ):

        n = 2
        l = range( 10 )
        expected = ( range(0,5), range(5,10) )
        actual = utilities.chunk_list( l, n )
        for expected_, actual_ in zip( expected, actual ):
            assert expected_ == actual_

        n = 3
        l = range( 10 )
        expected = ( range(0,4), range(4,7), range(7,10) )
        actual = utilities.chunk_list( l, n )
        for expected_, actual_ in zip( expected, actual ):
            assert expected_ == actual_

        n = 3
        l = range( 11 )
        expected = ( range(0,4), range(4,8), range(8,11) )
        actual = utilities.chunk_list( l, n )
        for expected_, actual_ in zip( expected, actual ):
            assert expected_ == actual_

########################################################################
########################################################################

class TestArraySetConversion( unittest.TestCase ):

    def test_arrays_to_set( self ):

        expected = set( [ ( 1, 1 ), ( 2, 2 ), ( 0, 0 ) ] )

        actual = utilities.arrays_to_set( np.arange( 3 ), np.arange( 3 ) )

        self.assertEqual( expected, actual )

    ########################################################################

    def test_arrays_to_set_long( self ):

        expected = set( [ ( 1, 1, 1 ), ( 2, 2, 2 ), ( 0, 0, 0 ) ] )

        actual = utilities.arrays_to_set( np.arange( 3 ), np.arange( 3 ), np.arange(3) )

        self.assertEqual( expected, actual )

    ########################################################################

    def test_set_to_arrays( self ):

        expected = np.array([
            [ 1, 4, 7, ],
            [ 2, 5, 8, ],
            [ 3, 6, 9, ],
        ])

        actual = utilities.set_to_arrays( set( [ ( 1, 2, 3, ), ( 4, 5, 6 ), ( 7, 8, 9 ), ] ) )

        npt.assert_allclose( expected[:,0], actual[:,-1] )

        # To allow for full comparison. We don't really care what order they're in, as long as they're matched.
        actual.sort()

        npt.assert_allclose( expected, actual )

########################################################################
########################################################################

class TestStoreParameters( unittest.TestCase ):

    def test_basic( self ):

        class Object( object ):

            @utilities.store_parameters
            def __init__( self, a, b, c=3 ):
                pass

        o = Object( 1, 2, )

        self.assertEqual( 1, o.a )
        self.assertEqual( 2, o.b )
        self.assertEqual( 3, o.c )
        self.assertEqual( [ 'a', 'b', 'c' ], o.stored_parameters )

    ########################################################################

    def test_kwargs_input( self ):

        class Object( object ):

            @utilities.store_parameters
            def __init__( self, a, b, c=3, **kwargs ):
                pass

        o = Object( **{ 'a' : 1, 'b' : 2, 'c' : 4, 'd' : 5 } )

        self.assertEqual( 1, o.a )
        self.assertEqual( 2, o.b )
        self.assertEqual( 4, o.c )
        self.assertEqual( {'d' : 5}, o.kwargs )
        self.assertEqual(
            sorted( [ 'a', 'b', 'c', 'kwargs', ] ),
            sorted( o.stored_parameters )
        )

    ########################################################################

    def test_args_and_kwargs_input( self ):

        class Object( object ):

            @utilities.store_parameters
            def __init__( self, *args, **kwargs ):
                pass

        args = ( 'dog', )
        kwargs = { 'a' : 1, 'b' : 2, 'c' : 4, 'd' : 5 }
        o = Object( *args, **kwargs )

        self.assertEqual( ('dog',), o.args )
        self.assertEqual( { 'a' : 1, 'b' : 2, 'c' : 4, 'd' : 5 }, o.kwargs )
        self.assertEqual( [ 'args', 'kwargs', ], o.stored_parameters )

########################################################################
########################################################################

class TestGenerateDerivedDataDoc( unittest.TestCase ):

    def setUp( self ):

        self.output_filepath = './tests/data/derived_data.rst'

        if os.path.isfile( self.output_filepath ):
            os.remove( self.output_filepath )

    ########################################################################

    def test_gen_derived_data_doc( self ):

        object_import_path = 'galaxy_dive.tests.test_utils.test_utilities.GenDerivedDataObject'

        # Call the function
        utilities.gen_derived_data_doc(
            object_import_path,
            self.output_filepath,
        )
                
        # Build Expected
        # Get the header
        with open ( './utils/derived_data_header.rst', 'r') as myfile:
            expected_header = myfile.readlines()
        # Get the generated data
        expected_generated = [
            '\n',
            '.. autoclass:: {}\n'.format( object_import_path ),
            '    :show-inheritance:\n',
            '\n',
            '    .. automethod:: calc_A\n',
            '    .. automethod:: calc_B\n',
        ]
        expected = expected_header + expected_generated

        with open ( self.output_filepath, 'r') as myfile:
            actual = myfile.readlines()

        self.assertEqual( expected, actual )

# Test Object for TestGenerateDerivedDataDoc
class GenDerivedDataObject( object ):

    def method_a( self, *args, **kwargs ):
        '''Method a.'''
        pass

    def calc_A( self ):
        '''Calculate A''' 
        pass

    def calc_B( self ):
        '''Calculate A''' 
        pass
