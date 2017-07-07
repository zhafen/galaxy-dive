'''Testing for generic_data.py
'''

from mock import patch
import numpy as np
import numpy.testing as npt
import pdb
import unittest

import galaxy_diver.analyze_data.generic_data as generic_data
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

class TestGriddedData(unittest.TestCase):

  def setUp(self):

    self.test_class = generic_data.GriddedData

    self.data_p = {
      'sdir' : './tests/test_data/test_analysis_dir',
      'snum' : 600,
      'Nx' : 10,
      'gridsize' : 'vary',
      'ionized' : 'test',
      }

  ########################################################################

  def test_init(self):

    instance = self.test_class(self.data_p)

    actual = instance.data['Den'][...][0,0,0]
    expected = 8.9849813e-08
    npt.assert_allclose(actual, expected)

  ########################################################################

  def test_load_ion_grid(self):

    self.data_p['ion_grid'] = True

    instance = self.test_class(self.data_p)

    assert instance.is_ion_grid

    actual = instance.data['H_p0_number_density'][...][0,0,0]
    expected = 1.0840375843397891e-11
    npt.assert_allclose(actual, expected)

  ########################################################################

  def test_calc_ion_column_densties(self):

    self.data_p['ion_grid'] = True

    instance = self.test_class(self.data_p)

    hi_col_den = instance.calc_column_density( 'H_p0_number_density', 2 )
    
    # Assert that we conserve the key data, with an outside source
    actual = hi_col_den.sum()/instance.data_attrs['cell_length']/3.086e21
    expected = 0.00013289821045735719
    npt.assert_allclose(actual, expected)

  ########################################################################

  def test_calc_rx_and_ry_for_ions(self):

    self.data_p['ion_grid'] = True

    instance = self.test_class(self.data_p)

    positions = instance.get_data('P')

    actual = positions.transpose()[0,0,0]
    dist = -instance.data_attrs['gridsize']/2.
    expected = np.array( [dist, dist, dist] )
    npt.assert_allclose(actual, expected)

    actual = np.array([ positions[0][1, 0, 0], positions[1][1, 0, 0], positions[2][1, 0, 0], ])
    dist = -instance.data_attrs['gridsize']/2.
    expected = np.array( [dist + 78.974358452690979, dist, dist] )
    npt.assert_allclose(actual, expected)

  ########################################################################

  def test_calc_rx_and_ry_for_columns(self):

    self.data_p['ion_grid'] = True

    instance = self.test_class(self.data_p)

    rx_face = instance.get_data('Rx_face_xy')
    ry_face = instance.get_data('Ry_face_xy')

    actual = np.array([ rx_face[1, 0], ry_face[1, 0], ])
    dist = -instance.data_attrs['gridsize']/2.
    expected = np.array( [ dist + 78.974358452690979, dist, ] )
    npt.assert_allclose(actual, expected)

    ry_face = instance.get_data('Ry_face_yz')
    rz_face = instance.get_data('Rz_face_yz')

    actual = np.array([ ry_face[1, 2], rz_face[1, 2], ])
    dist = -instance.data_attrs['gridsize']/2.
    expected = np.array( [ dist + 78.974358452690979, dist + 78.974358452690979*2., ] )
    npt.assert_allclose(actual, expected)

########################################################################

class TestParticleData(unittest.TestCase):

  def setUp(self):

    self.test_class = generic_data.ParticleData

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


    

