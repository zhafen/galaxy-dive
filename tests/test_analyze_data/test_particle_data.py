'''Testing for particle_data.py
'''

import h5py
from mock import patch
import numpy as np
import numpy.testing as npt
import os
import pdb
import unittest

import galaxy_diver.analyze_data.particle_data as particle_data
import galaxy_diver.read_data.snapshot as read_snapshot

########################################################################

class SaneEqualityArray(np.ndarray):
    '''Numpy array subclass that allows you to test if two arrays are equal.'''

    def __eq__(self, other):
            return (isinstance(other, np.ndarray) and self.shape == other.shape and np.allclose(self, other))

def sane_eq_array(list_in):
    '''Wrapper for SaneEqualityArray, that takes in a list.'''

    arr = np.array(list_in)

    return arr.view(SaneEqualityArray)

########################################################################

class TestParticleData(unittest.TestCase):

    def setUp(self):

        self.test_class = particle_data.ParticleData

        self.kwargs = {
            'sdir' : './tests/data/sdir3/output',
            'snum' : 600,
            'ptype' : 0,
            'ahf_index' : 600,
            }

        self.p_data = self.test_class( **self.kwargs )

    ########################################################################

    def test_init(self):

        assert self.p_data.data['P'] is not None

    ########################################################################

    def test_gets_new_ids( self ):

        self.kwargs['load_additional_ids'] = True
        self.p_data = self.test_class( **self.kwargs )

        P = read_snapshot.readsnap( self.kwargs['sdir'], self.kwargs['snum'], self.kwargs['ptype'], True )

        npt.assert_allclose( P['id'], self.p_data.data['ID'] )
        npt.assert_allclose( P['child_id'], self.p_data.data['ChildID'] )
        npt.assert_allclose( P['id_gen'], self.p_data.data['IDGen'] )

    ########################################################################

    def test_find_duplicate_ids( self ):

        actual = self.p_data.find_duplicate_ids()

        expected = np.array( [ 36091289, ] )

        npt.assert_allclose( actual, expected )

########################################################################


