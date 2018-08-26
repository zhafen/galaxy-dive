'''Testing for hdf5_wrapper.py
'''

import h5py
from mock import patch
import numpy as np
import numpy.testing as npt
import os
import shutil
import unittest

import galaxy_dive.utils.hdf5_wrapper as hdf5_wrapper

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

        self.test_data = {
            'a': np.array([1., 2., 3., 4.]),
            'b': np.array([6., 4., 2., 8.]),
            'c': np.array([9., -1., -3., 4.]),
        }

        self.test_data_complex = {
            'a': np.array([1., 2., 3., 4.]),
            'b': np.array([ [6., 7., 8.], [4., 3., 2.], [-2., -3., -4.], [-8., -7., -6.], ]),
            'c': np.array([9., -1., -3., 4.]),
            'd': np.array(['dog', 'cat', 'rabbit', 'mouse']),
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
                                'b': np.array([2., 4., 8., 6.]),
                                'c': np.array([-3., -1., 4., 9.]),
                              }

        for key in expected:
            npt.assert_allclose(actual[key], expected[key])

    ########################################################################

    def test_insert_data(self):
        '''Test that we can add to an existing dataset.'''

        # Make the original dataset
        self.test_instance.clear_hdf5_file()
        self.test_instance.save_data(self.test_data)

        data_to_insert = {'a': 2.5, 'b': 4., 'c': 6.}
        index_key = 'a'

        self.test_instance.insert_data(data_to_insert, index_key)

        f = h5py.File(self.filename, 'r')
        expected_data = {'a': np.array([1., 2., 2.5, 3., 4.]),
                                            'b': np.array([6., 4., 4., 2., 8.]),
                                            'c': np.array([9., -1., 6., -3., 4.]),
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

        data_to_insert = {'a': 2.5, 'b': np.array([1., -1., 5.]), 'c': 6., 'd': 'grimalkin'}
        index_key = 'a'

        self.test_instance.insert_data(data_to_insert, index_key)

        f = h5py.File(self.filename, 'r')
        expected_data = {'a': np.array([1., 2., 2.5,  3., 4.]),
                                            'b': np.array([ [6., 7., 8.], [4., 3., 2.], [1., -1., 5.], [-2., -3., -4.], [-8., -7., -6.], ]),
                                            'c': np.array([9., -1., 6., -3., 4.]),
                                            'd': np.array(['dog', 'cat', 'grimalkin', 'rabbit', 'mouse']),
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

        data_to_insert = {'a': 3., 'b': 2., 'c': -3.}
        index_key = 'a'

        self.test_instance.insert_data(data_to_insert, index_key)

        f = h5py.File(self.filename, 'r')
        expected_data = {'a': np.array([1., 2., 3., 4.]),
                                            'b': np.array([6., 4., 2., 8.]),
                                            'c': np.array([9., -1., -3., 4.]),
                                          }

        for key in expected_data:
            npt.assert_allclose(expected_data[key], f[key])
        f.close()

    ########################################################################

    def test_insert_data_empty_dataset(self):
        '''Test that we can use insert on an empty dataset.'''

        self.test_instance.clear_hdf5_file()

        data_to_insert = {'a': 2.5, 'b': 4., 'c': 6.}
        index_key = 'a'

        self.test_instance.insert_data(data_to_insert, index_key)

        f = h5py.File(self.filename, 'r')
        expected_data = {'a': np.array([ 2.5, ]),
                                            'b': np.array([ 4., ]),
                                            'c': np.array([ 6., ]),
                                          }

        for key in expected_data:
            npt.assert_allclose(expected_data[key], f[key])
        f.close()

    ########################################################################

    def test_insert_data_no_file(self):
        '''Test that we can use insert on a file that doesn't exist yet.'''

        # Delete the file
        os.system( 'rm {}'.format(self.filename) )

        data_to_insert = {'a': 2.5, 'b': 4., 'c': 6.}
        index_key = 'a'

        self.test_instance.insert_data(data_to_insert, index_key)

        f = h5py.File(self.filename, 'r')
        expected_data = {'a': np.array([ 2.5, ]),
                                            'b': np.array([ 4., ]),
                                            'c': np.array([ 6., ]),
                                          }

        for key in expected_data:
            npt.assert_allclose(expected_data[key], f[key])
        f.close()

    ########################################################################

    def test_insert_data_only_one_row_exists(self):

        # Make the original dataset
        self.test_instance.clear_hdf5_file()
        test_data = {'a': 2.,
                                            'b': 4.,
                                            'c': -1.,
                                          }

        self.test_instance.save_data(test_data)

        data_to_insert = {'a': 1., 'b': 6., 'c': 9.}
        index_key = 'a'

        self.test_instance.insert_data(data_to_insert, index_key)

        f = h5py.File(self.filename, 'r')
        expected_data = {'a': np.array([1., 2., ]),
                                            'b': np.array([6., 4., ]),
                                            'c': np.array([9., -1., ]),
                                          }

        for key in expected_data:
            npt.assert_allclose(expected_data[key], f[key])
        f.close()

    ########################################################################

    def test_insert_data_column(self):
        '''Test that we can insert columns of data with insert data.
        '''

        # Make the original dataset
        self.test_instance.clear_hdf5_file()
        test_data = {
            'a': [ 2, 3, ],
            'b': [ 4., 5., ],
            'c': [ -1., 0., ],
        }

        self.test_instance.save_data(test_data)

        data_to_insert = {'a': [3, 2], 'd': [ 6., 7., ] }
        index_key = 'a'

        self.test_instance.insert_data(data_to_insert, index_key, True)

        f = h5py.File(self.filename, 'r')
        expected_data = {
            'a': np.array([2, 3, ]),
            'b': np.array([4., 5., ]),
            'c': np.array([-1., 0., ]),
            'd': np.array([7., 6., ]),
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
            'a': np.array( [ [1., 2., 3.], [4., 5., 6.], [6., 7., 8.] ] ),
            'b': np.array( [ [3., -9., 7.], [2., 1., 3.], [7., 10., 2.] ] ),
        }

        # Make the original dataset
        self.test_instance.clear_hdf5_file()
        self.test_instance.save_data(subsample_data)

        result = self.test_instance.subsample_hdf5_file(2,)

        mock_randint.assert_called_once_with(0, 9, 4)

        expected = {
            'a': np.array( [ [4., 6.], [7., 5.] ] ),
            'b': np.array( [ [2., 3.], [10., 1.] ] ),
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
            'a': np.array( [ [1., 2., 3.], [4., 5., 6.], [6., 7., 8.] ] ),
            'b': np.array( [ [3., -9., 7.], [2., 1., 3.], [7., 10., 2.] ] ),
        }

        # Make the original dataset
        self.test_instance.clear_hdf5_file()
        self.test_instance.save_data(subsample_data)

        # Make the copy
        subsamples = 2
        self.test_instance.copy_hdf5_file(self.copy_filename, subsamples=subsamples)

        expected = {
            'a': np.array( [ [4., 6.], [7., 5.] ] ),
            'b': np.array( [ [2., 3.], [10., 1.] ] ),
            }


        g = h5py.File(self.copy_filename, 'r')
        for key in g.keys():
            npt.assert_allclose(expected[key], g[key])

        os.system( 'rm {}'.format(self.copy_filename) )

    ########################################################################

    def test_save_complicated_structure(self):

        group_1 = {
            'a': np.array( [ [1., 2., 3.], [4., 5., 6.], [6., 7., 8.] ] ),
            'b': np.array( [ [3., -9., 7.], [2., 1., 3.], [7., 10., 2.] ] ),
        }
        group_2 = {
            'a': np.array( [ [1., 2., 3.], [4., 5., 6.], [6., 7., 8.] ] ),
            'b': np.array( [ [3., -9., 7.], [2., 1., 3.], [7., 10., 2.] ] ),
        }
        group_3 = {},
        data = {
            'Group1': group_1,
            'Group2': group_2,
            'Group3': group_3,
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
            'a': np.array( [ [1., 2., 3.], [4., 5., 6.], [6., 7., 8.] ] ),
            'b': np.array( [ [3., -9., 7.], [2., 1., 3.], [7., 10., 2.] ] ),
        }
        group_2 = {
            'a': np.array( [ [1., 2., 3.], [4., 5., 6.], [6., 7., 8.] ] ),
            'b': np.array( [ [3., -9., 7.], [2., 1., 3.], [7., 10., 2.] ] ),
        }
        group_3 = {},
        subsample_data = {
            'Group1': group_1,
            'Group2': group_2,
            'Group3': group_3,
        }

        # Make the original dataset
        self.test_instance.clear_hdf5_file()
        self.test_instance.save_data(subsample_data)

        # Make the copy
        subsamples = 2
        self.test_instance.copy_hdf5_file(self.copy_filename, subsamples=subsamples)

        expected = {
            'a': np.array( [ [4., 6.], [7., 5.] ] ),
            'b': np.array( [ [2., 3.], [10., 1.] ] ),
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
            'a': np.array( [ 1., 2., 3., 4., 5., 6.,  ] ),
            'b': np.array( [ [3., -9., 7., 2., 1., 3.], [7., 10., 2., 5., 3., 1.] ] ).transpose(),
            'attrs': { 'attrs_exist': True, },
        }
        group_2 = {
            'a': np.array( [ 1., 2., 3., 4., 5., 6.,  ] ),
            'b': np.array( [ [3., -9., 7., 2., 1., 3.], [7., 10., 2., 5., 3., 1.] ] ).transpose(),
            'attrs': { 'attrs_exist': True, },
        }
        group_3 = {
            'attrs': { 'attrs_exist': True, },
        }
        subsample_data = {
            'Group1': group_1,
            'Group2': group_2,
            'Group3': group_3,
        }

        # Make the original dataset
        self.test_instance.clear_hdf5_file()
        self.test_instance.save_data(subsample_data)

        # Make the copy
        subsamples = 4
        self.test_instance.copy_hdf5_file(self.copy_filename, subsamples=subsamples, particle_data=True)

        expected = {
            'a': np.array( [ 4., 6., 3., 5. ] ),
            'b': np.array( [ [2., 3., 7., 1.], [5., 1., 2., 3.] ] ).transpose(),
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
            'a': np.array( [ [1., 2., 3.], [4., 5., 6.], [6., 7., 8.] ] ),
            'b': np.array( [ [3., -9., 7.], [2., 1., 3.], [7., 10., 2.] ] ),
            'attrs': {'attrs_exist': True},
        }
        group_2 = {
            'a': np.array( [ [1., 2., 3.], [4., 5., 6.], [6., 7., 8.] ] ),
            'b': np.array( [ [3., -9., 7.], [2., 1., 3.], [7., 10., 2.] ] ),
            'attrs': {'attrs_exist': True},
        }
        group_3 = {
            'attrs': {'attrs_exist': True},
        }
        data = {
            'Group1': group_1,
            'Group2': group_2,
            'Group3': group_3,
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
        'Group1': {
            'a': np.array( [ [1., 2., 3.], [4., 5., 6.], [6., 7., 8.] ] ),
            'b': np.array( [ [3., -9., 7.], [2., 1., 3.], [7., 10., 2.] ] ),
            'attrs': { 'attrs_exist': True, },
            },
        'Group2': {
            'a': np.array( [ [1., 2., 3.], [4., 5., 6.], [6., 7., 8.] ] ),
            'b': np.array( [ [3., -9., 7.], [2., 1., 3.], [7., 10., 2.] ] ),
            'attrs': { 'attrs_exist': True, }
            },
        }

        # Make the original dataset
        self.test_instance.clear_hdf5_file()
        self.test_instance.save_data(subsample_data)

        # Make the copy
        subsamples = 2
        self.test_instance.copy_hdf5_file(self.copy_filename, subsamples=subsamples)

        expected = {
            'a': np.array( [ [4., 6.], [7., 5.] ] ),
            'b': np.array( [ [2., 3.], [10., 1.] ] ),
            }

        g = h5py.File(self.copy_filename, 'r')
        for group in g.keys():
            for key in g[group].keys():
                npt.assert_allclose(expected[key], g[group][key])

            assert g[group].attrs['attrs_exist'] == True

        os.system( 'rm {}'.format(self.copy_filename) )

########################################################################

class TestCopySnapshot( unittest.TestCase ):

    def setUp(self):

        self.kwargs = {
            'sdir': './tests/data/sdir/output',
            'snum': 600,
            'n_files': 8,
            'copy_dir': './tests/data/temp_sdir/output',
            'redistribute': True,
            }

    ########################################################################

    def tearDown( self ):

        if os.path.isdir( './tests/data/temp_sdir' ):
            shutil.rmtree( './tests/data/temp_sdir' )

    ########################################################################

    def test_copy_snapshot_and_redistribute( self ):

        # Set the random seed
        np.random.seed(1234)

        hdf5_wrapper.copy_snapshot( **self.kwargs )

        f = h5py.File( os.path.join( self.kwargs['copy_dir'], 'snapdir_600/snapshot_600.0.hdf5'), 'r' )

        expected_ids = {
            'PartType0'  : np.array([56577266,  2447814,  5836451, 48854626, 38697472, 26844748, 12518198, 13929795]),
            'PartType1': np.array([ 96490067, 111969149,  82963761, 105912179, 130202505,  99168777, 81451547,  86626678]),
            'PartType2': np.array([143876937, 142690251, 141208178, 142867765, 146105043, 144348523, 143790176, 144671722]),
            'PartType4': np.array([ 8240338, 33395336, 50845628, 53558278,  8091731, 33163814, 33193203, 37236498]),
        }

        expected_positions = {
            'PartType0': np.array([[ 43147.37910547,  40849.73566323,  42018.36946875,  41953.76112891,
                42250.75959483,  42921.24354077,  41515.60304779,  41709.71414106],
            [ 43788.15997097,  43738.97513012,  44166.32677499,  46120.83087771,
                44294.84369966,  46882.80661966,  44056.62971023,  40841.09006986],
            [ 45874.27118392,  45494.31487345,  46320.23228283,  46816.0623442 ,
                46870.14027289,  47536.57624099,  46369.82810828,  45640.13294909]]),
            'PartType1': np.array([[ 41875.66483633,  41871.32568498,  40495.20591946,  41840.55990359,
                42598.06746686,  42654.54906594,  41768.28244483,  41861.53839883],
            [ 44118.35014121,  44072.60902362,  43457.33636481,  45991.11822595,
                44444.10578658,  46092.61460387,  44187.48020816,  44126.8824762 ],
            [ 46261.71664927,  46307.86194507,  45084.47439923,  46928.75842962,
                46494.5582603 ,  46606.12689646,  46387.78823959,  46145.42036851]]),
            'PartType2': np.array([[ 42718.6585258 ,  42702.82772914,  41160.28685959,  42389.72019772,
                69109.89212334,  40110.83523464,  41484.2452451 ,  14234.55600016],
            [ 46788.61036456,  46854.27699445,  43801.19749786,  41226.63322432,
                31753.93759704,  40179.45607042,  42103.70372868,  15006.74252232],
            [ 47164.62123673,  47055.86509581,  47683.73835005,  46284.58074773,
                19872.44520005,  43478.28376903,  44507.59821399,  70046.93697334]]),
            'PartType4': np.array([[ 41882.74558912,  41884.14086056,  41867.59334168,  41875.93178495,
                41875.21304509,  41876.46763611,  41874.55928862,  41875.64058087],
            [ 44120.2944586 ,  44123.60968022,  44116.0175617 ,  44122.7025734 ,
                44123.03437806,  44121.16050547,  44124.66589757,  44121.83243625],
            [ 46253.94805193,  46252.28734946,  46256.24757957,  46257.53836431,
                46257.84478371,  46257.60302959,  46258.12604825,  46257.58911288]]),
        }

        # Correct for hubble...
        for key in expected_positions.keys():
            expected_positions[key] *= 0.70199999999999996

        for i in range( self.kwargs['n_files'] ):
            orig_filename = 'snapdir_600/snapshot_600.0.hdf5'
            orig_filepath = os.path.join( self.kwargs['sdir'], orig_filename )
            filename = 'snapdir_600/snapshot_600.{}.hdf5'.format( i )
            out_filepath = os.path.join( self.kwargs['copy_dir'], filename )

            actual = h5py.File( out_filepath, 'r' )
            expected = h5py.File( orig_filepath, 'r' )

            # Compare headers
            for key in expected['Header'].attrs.keys():
                if key == 'NumPart_ThisFile':
                    expected_ = np.array( [ 5, 5, 5, 0, 5, 0, ] )
                elif key == 'NumFilesPerSnapshot':
                    expected_ = self.kwargs['n_files']
                else:
                    expected_ = expected['Header'].attrs[key]
                actual_ = actual['Header'].attrs[key]
                if isinstance( expected_, np.ndarray ):
                    npt.assert_allclose( expected_, actual_ )
                else:
                    self.assertEqual( expected_, actual_ )

            # Compare values
            for ptype in expected_ids.keys():

                expected_id = expected_ids[ptype][i]
                actual_id = actual[ptype]['ParticleIDs'][...][0]
                npt.assert_allclose( expected_id, actual_id )

                expected_pos = expected_positions[ptype][:,i]
                actual_pos = actual[ptype]['Coordinates'][...][0]
                npt.assert_allclose( expected_pos, actual_pos )


