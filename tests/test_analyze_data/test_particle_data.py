'''Testing for particle_data.py
'''

from mock import patch
import numpy as np
import numpy.testing as npt
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

    self.data_p = {
      'sdir' : './tests/test_data/test_sdir3/output',
      'snum' : 600,
      'ptype' : 0,
      }

  ########################################################################

  def test_init(self):

    instance = self.test_class(self.data_p)

    assert instance.data['P'] is not None

  ########################################################################

  def test_gets_new_ids( self ):

    self.data_p['load_additional_ids'] = True

    instance = self.test_class(self.data_p)

    P = read_snapshot.readsnap( self.data_p['sdir'], self.data_p['snum'], self.data_p['ptype'], True )

    npt.assert_allclose( P['id'], instance.data['ID'] )
    npt.assert_allclose( P['child_id'], instance.data['ChildID'] )
    npt.assert_allclose( P['id_gen'], instance.data['IDGen'] )


    

