'''Testing for utilities.py
'''

from mock import patch
import numpy as np
import numpy.testing as npt
import os
import pdb
import unittest

import galaxy_diver.utils.utilities as utilities

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

class TestGetInstanceSourceDir( unittest.TestCase ):

  def test_default( self ):

    actual = utilities.get_instance_source_dir( self )

    expected = os.path.join( os.getcwd(), 'tests/test_utils' )

    assert actual == expected
