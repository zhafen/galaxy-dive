'''Testing for particle_data.py
'''

import copy
from mock import patch, sentinel
import numpy as np
import numpy.testing as npt
import pdb
import unittest

import galaxy_dive.analyze_data.simulation_data as simulation_data
import galaxy_dive.read_data.snapshot as read_snapshot

########################################################################

default_kwargs = {
    'data_dir' : './tests/data/sdir',
    'ahf_data_dir' : './tests/data/analysis_dir',
    'snum' : 500,
    'ahf_index' : 600,
    'centered' : True,
    'vel_centered' : True,
}

########################################################################

class TestGetData( unittest.TestCase ):

    def setUp( self ):

        self.g_data = simulation_data.SimulationData( **default_kwargs )

        self.g_data.data_attrs = {
            'hubble' : 0.70199999999999996,
            'omega_matter' : 0.272,
            'omega_lambda' : 0.728,
            'redshift' : 0.16946,
        }

        # Setup some necessary data
        self.g_data.data = {
            'P' : np.random.rand( 3, 4 ),
            'V' : np.random.rand( 3, 4 ),
        }

    ########################################################################

    def test_get_data_slice( self ):

        actual = self.g_data.get_data( 'Rx', 2 )
        expected = self.g_data.data['P'][0][2]

        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_get_data_slice_multidim( self ):

        # Setup some necessary data
        self.g_data.data = {
            'P' : np.random.rand( 3, 4, 3 ),
        }

        actual = self.g_data.get_data( 'Rx', (slice(None),2), )
        expected = self.g_data.data['P'][0][:,2]

        npt.assert_allclose( expected, actual )

########################################################################

