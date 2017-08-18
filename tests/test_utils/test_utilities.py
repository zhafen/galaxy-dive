'''Testing for utilities.py
'''

from mock import patch
import numpy as np
import numpy.testing as npt
import os
import pdb
import subprocess
import unittest

import galaxy_diver.utils.utilities as utilities

########################################################################
########################################################################

class TestSmartDictStartup( unittest.TestCase ):

  def test_default( self ):

    d = { 'a' : 1, 'b' : 2 }

    smart_dict = utilities.SmartDict( d )

    self.assertEqual( smart_dict['b'], 2 )
    self.assertEqual( len( smart_dict ), 2 )

  def test_from_defaults_and_variations( self ):

    class TestClassA( object ):
      def __init__( self, a, b ):
        self.a = a
        self.b = b
      def return_contents( self ):
        return self.a, self.b

    defaults = { 'a' : 1, 'b' : 1 }
    variations = {
      1 : {},
      2 : { 'b' : 2 },
    }

    result = utilities.SmartDict.from_class_and_args( TestClassA, variations, defaults, )

    assert isinstance( result, utilities.SmartDict )

    expected = { 1 : ( 1, 1, ), 2 : ( 1, 2, ), }
    actual = result.return_contents()
    assert expected == actual

########################################################################

class TestSmartDict( unittest.TestCase ):

  def test_nested( self ):

    class TestClassA( object ):
      def __init__( self, key ):
        self.foo = 1234
        self.key = key
    class TestClassB( object ):
      def __init__( self, key ):
        self.test_class_a = TestClassA( key )
        self.key = key

    d = {}
    expected = {}
    expected2 = {}
    for i in range( 3 ):
      d[i] = TestClassB( i )
      expected[i] = 1234
      expected2[i] = i

    smart_d = utilities.SmartDict( d )

    actual = smart_d.test_class_a.foo
    self.assertEqual( expected, actual )

    actual = smart_d.key
    self.assertEqual( expected2, actual )

    actual = smart_d.test_class_a.key
    self.assertEqual( expected2, actual )

  ########################################################################

  def test_nested_method( self ):

    class TestClassA( object ):
      def foo( self, x ):
        return x**2
    class TestClassB( object ):
      def __init__( self ):
        self.test_class_a = TestClassA()

    d = {}
    expected = {}
    for i in range( 3 ):
      d[i] = TestClassB()
      expected[i] = 4

    smart_d = utilities.SmartDict( d )

    actual = smart_d.test_class_a.foo( 2 )

    self.assertEqual( expected, actual )

  ########################################################################

  def test_call_custom_kwargs( self ):

    class TestClassA( object ):
      def foo( self, x ):
        return x**2

    d = utilities.SmartDict( { 1 : TestClassA(), 2 : TestClassA(), } )

    kwargs = { 1 : { 'x' : 10}, 2 : { 'x' : 100}, } 
    actual = d.foo.call_custom_kwargs( kwargs )
    expected = { 1 : 100, 2 : 10000, }
    self.assertEqual( expected, actual )

  ########################################################################

  def test_multiply( self ):

    d = utilities.SmartDict( { 1 : 1, 2 : 2 } )

    expected = { 1 : 2, 2 : 4, }

    actual = d*2
    self.assertEqual( expected, actual )

    actual = 2*d
    self.assertEqual( expected, actual )

  ########################################################################

  def test_multiply_smart_dict( self ):

    d1 = utilities.SmartDict( { 1 : 1, 2 : 2 } )
    d2 = utilities.SmartDict( { 1 : 2, 2 : 3 } )

    expected = { 1 : 2, 2 : 6, }

    actual = d1*d2
    self.assertEqual( expected, actual )

    actual = d2*d1
    self.assertEqual( expected, actual )

  ########################################################################

  def test_divide( self ):

    d = utilities.SmartDict( { 1 : 2, 2 : 4 } )
    expected = { 1 : 1, 2 : 2, }
    actual = d/2
    self.assertEqual( expected, actual )

    d = utilities.SmartDict( { 1 : 2, 2 : 4 } )
    expected = { 1 : 2, 2 : 1, }
    actual = 4/d
    self.assertEqual( expected, actual )

  ########################################################################

  def test_divide_smart_dict( self ):

    d1 = utilities.SmartDict( { 1 : 9, 2 : 4 } )
    d2 = utilities.SmartDict( { 1 : 3, 2 : 2 } )

    expected = { 1 : 3, 2 : 2, }

    actual = d1/d2
    self.assertEqual( expected, actual )

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

    expected = os.path.join( os.getcwd(), 'galaxy_diver/utils' )

    assert actual == expected

########################################################################
########################################################################

class TestChunkList( unittest.TestCase ):

  def test_default( self ):

    n = 2
    l = range( 10 )
    expected = ( [ 0, 1, 2, 3, 4 ], [ 5, 6, 7, 8, 9 ] )
    actual = utilities.chunk_list( l, n )
    for expected_, actual_ in zip( expected, actual ):
      assert expected_ == actual_

    n = 3
    l = range( 10 )
    expected = ( [ 0, 1, 2, 3 ], [ 4, 5, 6 ], [ 7, 8, 9 ] )
    actual = utilities.chunk_list( l, n )
    for expected_, actual_ in zip( expected, actual ):
      assert expected_ == actual_

    n = 3
    l = range( 11 )
    expected = ( [ 0, 1, 2, 3 ], [ 4, 5, 6, 7 ], [ 8, 9, 10 ] )
    actual = utilities.chunk_list( l, n )
    for expected_, actual_ in zip( expected, actual ):
      assert expected_ == actual_
