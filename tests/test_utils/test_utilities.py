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

########################################################################

class TestSmartDict( unittest.TestCase ):

  def test_nested( self ):

    class TestClassA( object ):
      def __init__( self ):
        self.foo = 1234
    class TestClassB( object ):
      def __init__( self ):
        self.test_class_a = TestClassA()

    d = {}
    expected = {}
    for i in range( 3 ):
      d[i] = TestClassB()
      expected[i] = 1234

    smart_d = utilities.SmartDict( d )

    actual = smart_d.test_class_a.foo

    self.assertEqual( expected, actual )

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
