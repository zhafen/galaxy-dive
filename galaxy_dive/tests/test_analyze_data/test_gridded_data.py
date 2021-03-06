'''Testing for gridded_data.py
'''

import mock
import numpy as np
import numpy.testing as npt
import pdb
import unittest

import galaxy_dive.analyze_data.gridded_data as gridded_data
import galaxy_dive.read_data.snapshot as read_snapshot

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

        self.test_class = gridded_data.GriddedData

        self.kwargs = {
            'sdir' : './tests/data/analysis_dir',
            'snum' : 600,
            'Nx' : 10,
            'gridsize' : 'vary',
            'ionized' : 'test',
            'ahf_index' : 600,
            }

    ########################################################################

    def test_init(self):

        instance = self.test_class( **self.kwargs )

        actual = instance.data['Den'][...][0,0,0]
        expected = 8.9849813e-08
        npt.assert_allclose(actual, expected)

    ########################################################################

    def test_load_ion_grid(self):

        self.kwargs['ion_grid'] = True

        instance = self.test_class( **self.kwargs )

        assert instance.ion_grid

        actual = instance.data['H_p0_number_density'][...][0,0,0]
        expected = 1.0840375843397891e-11
        npt.assert_allclose(actual, expected)

    ########################################################################

    def test_calc_ion_column_densties(self):

        self.kwargs['ion_grid'] = True

        instance = self.test_class( **self.kwargs )

        hi_col_den = instance.calc_column_density( 'H_p0_number_density', 2 )

        # Assert that we conserve the key data, with an outside source
        actual = hi_col_den.sum()/instance.data_attrs['cell_length']/3.086e21
        expected = 0.00013289821045735719
        npt.assert_allclose(actual, expected)

    ########################################################################

    def test_calc_rx_and_ry_for_ions(self):

        self.kwargs['ion_grid'] = True

        instance = self.test_class( **self.kwargs )

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

        self.kwargs['ion_grid'] = True

        instance = self.test_class( **self.kwargs )

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

    @mock.patch( 'galaxy_dive.analyze_data.simulation_data.SimulationData.get_data' )
    def test_calc_impact_parameter( self, mock_get_data ):

        instance = self.test_class( **self.kwargs )

        mock_get_data.side_effect = [
            np.array( [ [ -1., -1.], [ 1., 1.] ] ),
            np.array( [ [ -1., 1.], [ 1., -1.] ] ),
        ]

        instance.calc_impact_parameter( 'R_face_xy' )

        actual = instance.data[ 'R_face_xy' ]
        expected = np.ones( (2, 2) )*np.sqrt( 2. )
        npt.assert_allclose(actual, expected)

