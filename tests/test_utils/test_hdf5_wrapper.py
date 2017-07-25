'''Testing for hdf5_wrapper.py
'''

import h5py
from mock import patch
import numpy as np
import numpy.testing as npt
import os
import pdb
import unittest

import galaxy_diver.utils.hdf5_wrapper as hdf5_wrapper

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

class TestHDF5Wrapper(unittest.TestCase):

  def setUp(self):

    self.filename = './tests/test.h5'
    self.copy_filename = './tests/test_copy.h5'

    self.test_object = hdf5_wrapper.HDF5Wrapper

    self.test_instance = self.test_object(self.filename)

    self.test_data = {'a': np.array([1., 2., 3., 4.]),
                      'b' : np.array([6., 4., 2., 8.]),
                      'c' : np.array([9., -1., -3., 4.]),
                     }

    self.test_data_complex = {'a': np.array([1., 2., 3., 4.]),
                      'b' : np.array([ [6., 7., 8.], [4., 3., 2.], [-2., -3., -4.], [-8., -7., -6.], ]),
                      'c' : np.array([9., -1., -3., 4.]),
                      'd' : np.array(['dog', 'cat', 'rabbit', 'mouse']),
                     }

  def tearDown(self):

    if os.path.isfile(self.filename):
      os.system( 'rm {}'.format(self.filename) )
    if os.path.isfile(self.copy_filename):
      os.system( 'rm {}'.format(self.copy_filename) )
    if os.path.isdir( './tests/data/output_data' ):
      os.system( 'rm -r ./tests/data/output_data' )

  ########################################################################

  def test_can_create_class(self):

    if self.test_instance is None:
      assert False

  ########################################################################

  def test_fragile_makes_hdf5_file_blank(self):
    '''Part of the testing suite that will fail if anything else does.'''

    self.test_instance.clear_hdf5_file()

    # Make a file
    f = h5py.File(self.filename, 'a')
    try:
      f.create_dataset('a', self.test_data['a'])
    except RuntimeError:
      pass
    f.close()

    self.test_instance.clear_hdf5_file()

    f = h5py.File(self.filename, 'r')
    self.assertEqual(len(f.keys()), 0)
    f.close()

  ########################################################################

  def test_dataset_from_empty(self):
    '''Test that we can make a dataset from nothing.'''

    self.test_instance.clear_hdf5_file()
    self.test_instance.save_data(self.test_data)

    f = h5py.File(self.filename, 'r')
    npt.assert_allclose(self.test_data['a'], f['a'])
    f.close()


  ########################################################################

  def test_sort_dictionary(self):

    actual = self.test_instance.sort_dictionary(self.test_data, 'c')
    
    expected = {'a': np.array([3., 2., 4., 1.]),
                'b' : np.array([2., 4., 8., 6.]),
                'c' : np.array([-3., -1., 4., 9.]),
               }

    for key in expected:
      npt.assert_allclose(actual[key], expected[key])

  ########################################################################

  def test_insert_data(self):
    '''Test that we can add to an existing dataset.'''

    # Make the original dataset
    self.test_instance.clear_hdf5_file()
    self.test_instance.save_data(self.test_data)

    data_to_insert = {'a' : 2.5, 'b': 4., 'c': 6.}
    index_key = 'a'

    self.test_instance.insert_data(data_to_insert, index_key)

    f = h5py.File(self.filename, 'r')
    expected_data = {'a': np.array([1., 2., 2.5,  3., 4.]),
                      'b' : np.array([6., 4., 4., 2., 8.]),
                      'c' : np.array([9., -1., 6., -3., 4.]),
                     }

    for key in expected_data:
      npt.assert_allclose(expected_data[key], f[key])
    f.close()

  ########################################################################

  def test_insert_data_complex(self):
    '''Test that we can add to an existing dataset, for a slightly more complicated dataset.'''

    # Make the original dataset
    self.test_instance.clear_hdf5_file()
    self.test_instance.save_data(self.test_data_complex)

    data_to_insert = {'a' : 2.5, 'b': np.array([1., -1., 5.]), 'c': 6., 'd': 'grimalkin'}
    index_key = 'a'

    self.test_instance.insert_data(data_to_insert, index_key)

    f = h5py.File(self.filename, 'r')
    expected_data = {'a': np.array([1., 2., 2.5,  3., 4.]),
                      'b' : np.array([ [6., 7., 8.], [4., 3., 2.], [1., -1., 5.], [-2., -3., -4.], [-8., -7., -6.], ]),
                      'c' : np.array([9., -1., 6., -3., 4.]),
                      'd' : np.array(['dog', 'cat', 'grimalkin', 'rabbit', 'mouse']),
                     }

    for key in expected_data:
      if key != 'd':
        npt.assert_allclose(expected_data[key], f[key])

    # Do the testing for when it's strings instead.
    for i, animal in enumerate(expected_data['d']):
      self.assertEqual(animal, f['d'][i])

    f.close()

  ########################################################################

  def test_insert_duplicate_data(self):
    '''Test that adding existing values to a dataset doesn't do anything..'''

    # Make the original dataset
    self.test_instance.clear_hdf5_file()
    self.test_instance.save_data(self.test_data)

    data_to_insert = {'a' : 3., 'b': 2., 'c': -3.}
    index_key = 'a'

    self.test_instance.insert_data(data_to_insert, index_key)

    f = h5py.File(self.filename, 'r')
    expected_data = {'a': np.array([1., 2., 3., 4.]),
                      'b' : np.array([6., 4., 2., 8.]),
                      'c' : np.array([9., -1., -3., 4.]),
                     }

    for key in expected_data:
      npt.assert_allclose(expected_data[key], f[key])
    f.close()

  ########################################################################

  def test_insert_data_empty_dataset(self):
    '''Test that we can use insert on an empty dataset.'''

    self.test_instance.clear_hdf5_file()

    data_to_insert = {'a' : 2.5, 'b': 4., 'c': 6.}
    index_key = 'a'

    self.test_instance.insert_data(data_to_insert, index_key)

    f = h5py.File(self.filename, 'r')
    expected_data = {'a': np.array([ 2.5,]),
                      'b' : np.array([ 4.,]),
                      'c' : np.array([ 6.,]),
                     }

    for key in expected_data:
      npt.assert_allclose(expected_data[key], f[key])
    f.close()

  ########################################################################

  def test_insert_data_no_file(self):
    '''Test that we can use insert on a file that doesn't exist yet.'''

    # Delete the file
    os.system( 'rm {}'.format(self.filename) )

    data_to_insert = {'a' : 2.5, 'b': 4., 'c': 6.}
    index_key = 'a'

    self.test_instance.insert_data(data_to_insert, index_key)

    f = h5py.File(self.filename, 'r')
    expected_data = {'a': np.array([ 2.5,]),
                      'b' : np.array([ 4.,]),
                      'c' : np.array([ 6.,]),
                     }

    for key in expected_data:
      npt.assert_allclose(expected_data[key], f[key])
    f.close()

  ########################################################################

  def test_insert_data_only_one_row_exists(self):

    # Make the original dataset
    self.test_instance.clear_hdf5_file()
    test_data = {'a': 2.,
                      'b' : 4.,
                      'c' : -1.,
                     }

    self.test_instance.save_data(test_data)

    data_to_insert = {'a' : 1., 'b': 6., 'c': 9.}
    index_key = 'a'

    self.test_instance.insert_data(data_to_insert, index_key)

    f = h5py.File(self.filename, 'r')
    expected_data = {'a': np.array([1., 2.,]),
                      'b' : np.array([6., 4.,]),
                      'c' : np.array([9., -1.,]),
                     }

    for key in expected_data:
      npt.assert_allclose(expected_data[key], f[key])
    f.close()

  ########################################################################

  def test_copy_data(self):

    # Make the original dataset
    self.test_instance.clear_hdf5_file()
    self.test_instance.save_data(self.test_data)

    self.test_instance.copy_hdf5_file(self.copy_filename)

    assert os.path.isfile(self.copy_filename)

    os.system( 'rm {}'.format(self.copy_filename) )

  ########################################################################

  @patch('numpy.random.randint')
  def test_subsample(self, mock_randint):

    # Make a side effect for the random integer
    def side_effect(low, high, number):
      return np.array([3, 5, 7, 4])
    mock_randint.side_effect = side_effect

    subsample_data = {
      'a' : np.array( [ [1., 2., 3.], [4., 5., 6.], [6., 7., 8.] ] ),
      'b' : np.array( [ [3., -9., 7.], [2., 1., 3.], [7., 10., 2.] ] ),
    }

    # Make the original dataset
    self.test_instance.clear_hdf5_file()
    self.test_instance.save_data(subsample_data)

    result = self.test_instance.subsample_hdf5_file(2,)

    mock_randint.assert_called_once_with(0, 9, 4)

    expected = {
      'a' : np.array( [ [4., 6.], [7., 5.] ] ),
      'b' : np.array( [ [2., 3.], [10., 1.] ] ),
      }

    for key in result:
      npt.assert_allclose(expected[key], result[key])

  ########################################################################

  @patch('numpy.random.randint')
  def test_copy_data_subsample(self, mock_randint):

    # Make a side effect for the random integer
    def side_effect(low, high, number):
      return np.array([3, 5, 7, 4])
    mock_randint.side_effect = side_effect

    subsample_data = {
      'a' : np.array( [ [1., 2., 3.], [4., 5., 6.], [6., 7., 8.] ] ),
      'b' : np.array( [ [3., -9., 7.], [2., 1., 3.], [7., 10., 2.] ] ),
    }

    # Make the original dataset
    self.test_instance.clear_hdf5_file()
    self.test_instance.save_data(subsample_data)

    # Make the copy
    subsamples = 2
    self.test_instance.copy_hdf5_file(self.copy_filename, subsamples=subsamples)

    expected = {
      'a' : np.array( [ [4., 6.], [7., 5.] ] ),
      'b' : np.array( [ [2., 3.], [10., 1.] ] ),
      }

    
    g = h5py.File(self.copy_filename, 'r')
    for key in g.keys():
      npt.assert_allclose(expected[key], g[key])

    os.system( 'rm {}'.format(self.copy_filename) )

  ########################################################################

  def test_save_complicated_structure(self):

    group_1 = {
      'a' : np.array( [ [1., 2., 3.], [4., 5., 6.], [6., 7., 8.] ] ),
      'b' : np.array( [ [3., -9., 7.], [2., 1., 3.], [7., 10., 2.] ] ),
    }
    group_2 = {
      'a' : np.array( [ [1., 2., 3.], [4., 5., 6.], [6., 7., 8.] ] ),
      'b' : np.array( [ [3., -9., 7.], [2., 1., 3.], [7., 10., 2.] ] ),
    }
    group_3 = {},
    data = {
      'Group1' : group_1,
      'Group2' : group_2,
      'Group3' : group_3,
    }

    # Make the original dataset
    self.test_instance.clear_hdf5_file()
    self.test_instance.save_data(data)

    f = h5py.File(self.filename, 'r')
    for group in ['Group1', 'Group2']:
      for key in f[group].keys():
        npt.assert_allclose( f[group][key][...], data[group][key] )

  ########################################################################

  @patch('numpy.random.randint')
  def test_copy_data_subsample_complicated_structure(self, mock_randint):

    # Make a side effect for the random integer
    def side_effect(low, high, number):
      return np.array([3, 5, 7, 4])
    mock_randint.side_effect = side_effect

    group_1 = {
      'a' : np.array( [ [1., 2., 3.], [4., 5., 6.], [6., 7., 8.] ] ),
      'b' : np.array( [ [3., -9., 7.], [2., 1., 3.], [7., 10., 2.] ] ),
    }
    group_2 = {
      'a' : np.array( [ [1., 2., 3.], [4., 5., 6.], [6., 7., 8.] ] ),
      'b' : np.array( [ [3., -9., 7.], [2., 1., 3.], [7., 10., 2.] ] ),
    }
    group_3 = {},
    subsample_data = {
      'Group1' : group_1,
      'Group2' : group_2,
      'Group3' : group_3,
    }

    # Make the original dataset
    self.test_instance.clear_hdf5_file()
    self.test_instance.save_data(subsample_data)

    # Make the copy
    subsamples = 2
    self.test_instance.copy_hdf5_file(self.copy_filename, subsamples=subsamples)

    expected = {
      'a' : np.array( [ [4., 6.], [7., 5.] ] ),
      'b' : np.array( [ [2., 3.], [10., 1.] ] ),
      }

    
    g = h5py.File(self.copy_filename, 'r')
    for group in ['Group1', 'Group2']:
      for key in g[group].keys():
        npt.assert_allclose( g[group][key][...], expected[key] )

    os.system( 'rm {}'.format(self.copy_filename) )

  ########################################################################

  @patch('numpy.random.randint')
  def test_copy_data_subsample_particle_like_data(self, mock_randint):

    # Make a side effect for the random integer
    def side_effect(low, high, number):
      return np.array([3, 5, 2, 4])
    mock_randint.side_effect = side_effect

    group_1 = {
      'a' : np.array( [ 1., 2., 3., 4., 5., 6.,  ] ),
      'b' : np.array( [ [3., -9., 7., 2., 1., 3.], [7., 10., 2., 5., 3., 1.] ] ).transpose(),
      'attrs' : { 'attrs_exist' : True, },
    }
    group_2 = {
      'a' : np.array( [ 1., 2., 3., 4., 5., 6.,  ] ),
      'b' : np.array( [ [3., -9., 7., 2., 1., 3.], [7., 10., 2., 5., 3., 1.] ] ).transpose(),
      'attrs' : { 'attrs_exist' : True, },
    }
    group_3 = {
      'attrs' : { 'attrs_exist' : True, },
    }
    subsample_data = {
      'Group1' : group_1,
      'Group2' : group_2,
      'Group3' : group_3,
    }

    # Make the original dataset
    self.test_instance.clear_hdf5_file()
    self.test_instance.save_data(subsample_data)

    # Make the copy
    subsamples = 4
    self.test_instance.copy_hdf5_file(self.copy_filename, subsamples=subsamples, particle_data=True)

    expected = {
      'a' : np.array( [ 4., 6., 3., 5. ] ),
      'b' : np.array( [ [2., 3., 7., 1.], [5., 1., 2., 3.] ] ).transpose(),
      }

    
    g = h5py.File(self.copy_filename, 'r')
    for group in ['Group1', 'Group2']:
      for key in g[group].keys():
        npt.assert_allclose( g[group][key][...], expected[key] )

    os.system( 'rm {}'.format(self.copy_filename) )

  ########################################################################

  def test_copy_and_subsample_particle_snapshot( self ):

    filedir = './tests/data/sdir/output'
    snum = 600
    subsamples = 2

    copy_filedir = './tests/data/output_data/'

    # Set the random seed
    np.random.seed(1234)

    hdf5_wrapper.copy_snapshot( filedir, snum, copy_filedir, subsamples )

    f = h5py.File( os.path.join(copy_filedir, 'snapdir_600/snapshot_600.0.hdf5'), 'r' )

    actual = f['PartType0']['ParticleIDs'][...]
    
    expected = np.array([50743870, 66266902])
    npt.assert_allclose( actual, expected )

  ########################################################################

  def test_save_data_group_attrs(self):

    group_1 = {
      'a' : np.array( [ [1., 2., 3.], [4., 5., 6.], [6., 7., 8.] ] ),
      'b' : np.array( [ [3., -9., 7.], [2., 1., 3.], [7., 10., 2.] ] ),
      'attrs' : {'attrs_exist' : True},
    }
    group_2 = {
      'a' : np.array( [ [1., 2., 3.], [4., 5., 6.], [6., 7., 8.] ] ),
      'b' : np.array( [ [3., -9., 7.], [2., 1., 3.], [7., 10., 2.] ] ),
      'attrs' : {'attrs_exist' : True},
    }
    group_3 = {
      'attrs' : {'attrs_exist' : True},
    }
    data = {
      'Group1' : group_1,
      'Group2' : group_2,
      'Group3' : group_3,
    }

    # Make the original dataset
    self.test_instance.clear_hdf5_file()
    self.test_instance.save_data(data)

    f = h5py.File(self.filename, 'r')
    for group in ['Group1', 'Group2', 'Group3']:
      for key in f[group].keys():
        if key != 'attrs':
          npt.assert_allclose( f[group][key][...], data[group][key] )
        else:
          for attr_key in f[group].attrs.keys():
            assert data[group][key][attr_key] == f[group].attrs[attr_key]

    os.system( 'rm {}'.format(self.copy_filename) )

  ########################################################################

  @patch('numpy.random.randint')
  def test_copy_data_subsample_attrs_included(self, mock_randint):

    # Make a side effect for the random integer
    def side_effect(low, high, number):
      return np.array([3, 5, 7, 4])
    mock_randint.side_effect = side_effect

    subsample_data = {
    'Group1' : {
      'a' : np.array( [ [1., 2., 3.], [4., 5., 6.], [6., 7., 8.] ] ),
      'b' : np.array( [ [3., -9., 7.], [2., 1., 3.], [7., 10., 2.] ] ),
      'attrs' : { 'attrs_exist' : True, },
      },
    'Group2' : {
      'a' : np.array( [ [1., 2., 3.], [4., 5., 6.], [6., 7., 8.] ] ),
      'b' : np.array( [ [3., -9., 7.], [2., 1., 3.], [7., 10., 2.] ] ),
      'attrs' : { 'attrs_exist' : True, }
      },
    }

    # Make the original dataset
    self.test_instance.clear_hdf5_file()
    self.test_instance.save_data(subsample_data)

    # Make the copy
    subsamples = 2
    self.test_instance.copy_hdf5_file(self.copy_filename, subsamples=subsamples)

    expected = {
      'a' : np.array( [ [4., 6.], [7., 5.] ] ),
      'b' : np.array( [ [2., 3.], [10., 1.] ] ),
      }
    
    g = h5py.File(self.copy_filename, 'r')
    for group in g.keys():
      for key in g[group].keys():
        npt.assert_allclose(expected[key], g[group][key])

      assert g[group].attrs['attrs_exist'] == True

    os.system( 'rm {}'.format(self.copy_filename) )
