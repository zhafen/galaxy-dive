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

