'''Testing for data_operations.py
'''

from mock import patch
import numpy as np
import numpy.testing as npt
import pdb
import unittest

import galaxy_dive.utils.data_operations as data_operations

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

class TestRotateCoordinates(unittest.TestCase):

    def setUp(self):

        # What is the name of the function.
        self.fn = data_operations.rotate_coordinates

        self.input_arr = np.random.rand(4, 3)
        self.angle = -np.pi/2.
        self.axis = np.array([0., 0., 1.])

        self.default_args = [self.input_arr, self.angle, self.axis]

    ########################################################################

    def test_output_format(self):
        '''Test that we return an array of the same dimensions'''

        expected = self.input_arr.shape

        result = self.fn(*self.default_args)

        actual = result.shape

        self.assertEqual(expected, actual)

    ########################################################################

    def test_rotate_vector_in_xy_plane_by_90(self):

        # Change the input array.
        self.default_args[0] = np.array([1., 0., 0.])

        result = self.fn(*self.default_args)

        expected = np.array([0., 1., 0.])

        npt.assert_allclose(result, expected, atol=1e-8)

        # Second test
        # Change the input array.
        self.default_args[0] = np.array([0., 1., 0.])

        result = self.fn(*self.default_args)

        expected = np.array([-1., 0., 0.])

        npt.assert_allclose(result, expected, atol=1e-8)

        # Third test
        # Change the input array.
        self.default_args[0] = np.array([0., 0., 1.])

        result = self.fn(*self.default_args)

        expected = np.array([0., 0., 1.])

        npt.assert_allclose(result, expected, atol=1e-8)

    ########################################################################

    def test_rotate_array_in_xy_plane_by_90(self):

        # Change the input array.
        self.default_args[0] = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [0., 1., 0.]])

        result = self.fn(*self.default_args)

        expected = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.], [-1., 0., 0.],])

        npt.assert_allclose(expected, result)

    ########################################################################

    def test_rotate_array_down_x_axis_by_90(self):

        # Change the input array.
        self.default_args[0] = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [0., 1., 0.]])

        # Change the axis
        self.default_args[2] = np.array([1., 0., 0.])

        result = self.fn(*self.default_args)

        expected = np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.], [0., 0., 1.],])

        npt.assert_allclose(expected, result)

    ########################################################################

    def test_rotate_by_arbitrary_amount(self):

        # Change the input array.
        self.default_args[0] = np.array([[1., 0., 0.], [0., 1., 0.], [2., 0., 0.]])

        # Change the angle to an arbitrary angle
        self.default_args[1] = np.pi/5.

        # Change the vector to an arbitrary vector
        self.default_args[2] = np.array([1., 2., 3. ])

        result = self.fn(*self.default_args)

        # Test we keep the same angle with the vector we're rotating around
        expected_dot_w_rot_vector = np.dot(self.default_args[0][0], self.default_args[2])
        actual_dot_w_rot_vector = np.dot(result[0], self.default_args[2])
        npt.assert_allclose(expected_dot_w_rot_vector, actual_dot_w_rot_vector)

        # Test we get out the magnitude we're expecting
        expected_magnitude = 2.
        actual_magnitude = np.linalg.norm(result[2])
        npt.assert_allclose(expected_magnitude, actual_magnitude)

        # Test we get out the rotation we're expecting
        # (In the sense of the dot product should be np.pi/2, i.e. both should be rotated the same amount)
        expected_rotation = np.cos(-np.pi/2.)
        actual_rotation = np.dot(result[1], result[0])
        npt.assert_allclose(expected_rotation, actual_rotation, atol=1e-7)

        # Test we get out the rotation we're expecting
        # (In the sense of the dot product of the projection should be the angle it's rotated by.)
        expected_rotation = np.cos(self.default_args[1])
        dir_unit = self.default_args[2]/np.linalg.norm(self.default_args[2])
        # Test for each part of the coordinates
        for result_, orig_ in zip(result, self.default_args[0]):

            # Get the projections
            orig_proj = orig_ - np.dot(dir_unit, orig_)*dir_unit
            result_proj = result_ - np.dot(dir_unit, result_)*dir_unit

            actual_rotation = np.dot(orig_proj, result_proj)/np.linalg.norm(orig_proj)/np.linalg.norm(result_proj)

            npt.assert_allclose(expected_rotation, actual_rotation, atol=1e-14)

########################################################################

class TestAlignAxes(unittest.TestCase):

    def setUp(self):

        # What is the name of the function.
        self.fn = data_operations.align_axes

        self.input_arr = np.random.rand(4, 3)
        self.new_z_axis = np.array([1., 2., 3.])

        self.default_args = [self.input_arr, self.new_z_axis]

    ########################################################################

    def test_output_format(self):
        '''Test that we return an array of the same dimensions'''

        expected = self.input_arr.shape

        result = self.fn(*self.default_args)

        actual = result.shape

        self.assertEqual(expected, actual)

    ########################################################################

    def test_rotate_array_to_align_z_with_x(self):

        # Change the input array.
        self.default_args[0] = np.array([ 1., 0., 0., ])
        #self.default_args[0] = np.array([0., 1., 0.])

        # Align z with x
        self.default_args[1] = np.array([ 1., 0., 0. ])
        result = self.fn(*self.default_args)

        #expected = np.array([0., 0., 1.])
        expected = np.array([0., 0., 1.])

        npt.assert_allclose(expected, result)

    ########################################################################

    def test_rotate_array_to_align_z_with_x_and_check_products(self):

        # Change the input array.
        self.default_args[0] = np.array([
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
            [1., 0., 0.],
        ])

        # Align z with x
        self.default_args[1] = np.array([1., 0., 0.])

        result = self.fn(*self.default_args)

        # Make sure that our results have the right angles
        expected = 0.
        actual = np.dot(self.default_args[0][0], result[0])
        npt.assert_allclose(expected, actual, atol=1e-14)

        # Make sure that our results have the right angles
        expected = 1.
        actual = np.dot(np.array([0., 0., 1.]), result[0])
        npt.assert_allclose(expected, actual, atol=1e-14)

        # Make sure that our results have the right angles
        expected = 0.
        actual = np.dot(np.array([0., 0., 1.]), result[1])
        npt.assert_allclose(expected, actual, atol=1e-14)

        # Make sure that our results have the right angles
        expected = 0.
        actual = np.dot(np.array([0., 0., 1.]), result[2])
        npt.assert_allclose(expected, actual, atol=1e-14)

    ########################################################################

    def test_rotate_doesnt_do_anything(self):
        '''Test that nothing happens when new_z = old_z'''

        # Change the input array.
        self.default_args[0] = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])

        # Align z with z
        self.default_args[1] = np.array([0., 0., 1.])

        # Add an arbitrary point
        self.default_args.append( np.array([1., 2., 3.]) )

        result = self.fn(*self.default_args)

        npt.assert_allclose(self.default_args[0], result)

    ########################################################################

    def test_rotate_about_point(self):

        # Change the input array.
        self.default_args[0] = np.array([1., 0., 0.],)

        # Align z with y
        self.default_args[1] = np.array([0., 1., 0.])

        # Add an arbitrary point
        self.default_args.append( np.array([0., 0., 1.]) )

        result = self.fn(*self.default_args)

        # Test if we have the transformed arrays right
        npt.assert_allclose( result, np.array([1., 1., 1.]) )

    ########################################################################

    def test_rotate_about_point_arr(self):
        '''Test that we can do an offset for a number of coordinates.'''

        # Change the input array.
        self.default_args[0] = np.array([[1., 0., 0.], [0., 1., 1.], [0., 0., 1.], [1., 0., 0.]])

        # Align z with y
        self.default_args[1] = np.array([0., 1., 0.])

        # Add an arbitrary point
        self.default_args.append( np.array([0., 0., 1.]) )

        result = self.fn(*self.default_args)

        # Test if we have the transformed arrays right
        npt.assert_allclose( result[0], np.array([1., 1., 1.]) )
        npt.assert_allclose( result[1], np.array([0., 0., 2.]) )

    ########################################################################

    def test_align_vectors( self ):
        '''Test that we can align with a random vector.'''

        # Vector to align
        v = np.random.uniform( -1., 1., 3 )

        # Build the right answer
        v_norm = np.linalg.norm( v )
        expected = np.tile( v/v_norm, (3, 1) )

        # Create the input
        self.default_args[0] = np.array([
            [ 0., 0., 1., ],
            [ 0., 0., 1., ],
            [ 0., 0., 1., ],
        ])

        # Align z with vector
        self.default_args[1] = v

        # Actual calculation
        actual = self.fn( align_frame=False, *self.default_args )

        npt.assert_allclose( expected, actual )

########################################################################
########################################################################

class TestCumsum2D( unittest.TestCase ):

    def test_cumsum2d( self ):

        arr = np.array([
              [6, 5, 1],
              [7, 0, 1],
              [1, 1, 8]]
        )

        expected = np.array([
              [6, 11, 12],
              [13, 18, 20],
              [14, 20, 30]]
        )

        actual = data_operations.cumsum2d( arr )

        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_cumsum2d_dir( self ):

        arr = np.array([
              [6, 5, 1],
              [7, 0, 1],
              [1, 1, 8],
        ])

        expected = np.array([
              [12, 6, 1],
              [20, 7, 2],
              [30, 16, 10],
        ])

        actual = data_operations.cumsum2d( arr, directions=[1,-1] )

        npt.assert_allclose( expected, actual )
                
