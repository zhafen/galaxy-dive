#!/usr/bin/env python
'''Tools for generic data operations

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import copy
import numpy as np
import scipy
import scipy.signal

import galaxy_dive.utils.transformations as transformations

########################################################################
# Functions for doing simple things to data
########################################################################


def apply_floor( arr, floor ):
    '''Raises all values in the array below the floor to the floor value.

    Args:
        arr (np.arr): Array to raise the values of.
        floor (float): Floor value that will act as the minimum.

    Returns:
        floored_arr (np.arr): Array with all values below floor set to floor.
    '''

    return np.where( arr >= floor, arr, floor * np.ones(arr.shape))

########################################################################


def merge_dict( dict1, dict2 ):
    '''Merge two dictionaries, with dict2's keyword overridding dict1's keywords when there are duplicates

    Args:
        dict1 (dict): First dictionary.
        dict2 (dict): Second dictionary.

    Returns:
        merged_dict (dict): Merged dictionary.
    '''

    merged_dict = copy.deepcopy(dict1)

    merged_dict.update(dict2)

    return merged_dict

########################################################################


def block_mean( arr, fact ):
    '''Downsample an array

    Args:
        arr (np.arr): Array to downsample.
        fact (float): Factor by which to downsample.

    Returns:
        res (np.arr): Downsampled array.
    '''

    assert isinstance(fact, int), type(fact)

    sx, sy = arr.shape
    X, Y = np.ogrid[0:sx, 0:sy]
    regions = sy / fact * (X / fact) + Y / fact

    res = ndimage.mean(arr, labels=regions, index=np.arange(regions.max() + 1))
    res.shape = (sx / fact, sy / fact)

    return res

########################################################################


def cartesian( arrays, out=None ):
    """
    Generate a cartesian product of input arrays.
    Produced on Stack Overflow: http://stackoverflow.com/a/1235363/6651313

    Args:
        arrays (list of array-like): 1-D arrays to form the cartesian product of.
        out (ndarray): Array to place the cartesian product in.

    Returns:
        out (ndarray) 2-D array of shape (M, len(arrays)) containing cartesian products
                formed of input arrays.

    Examples:
        >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
        array([[1, 4, 6],
                      [1, 4, 7],
                      [1, 5, 6],
                      [1, 5, 7],
                      [2, 4, 6],
                      [2, 4, 7],
                      [2, 5, 6],
                      [2, 5, 7],
                      [3, 4, 6],
                      [3, 4, 7],
                      [3, 5, 6],
                      [3, 5, 7]])
    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
            out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:, 0] = np.repeat(arrays[0], m)
    if arrays[1:]:
            cartesian(arrays[1:], out=out[0:m, 1:])
            for j in xrange(1, arrays[0].size):
                    out[j * m:(j + 1) * m, 1:] = out[0:m, 1:]
    return out

########################################################################


def rotate_coordinates( arr, angle, axis, point=None ):
    '''Rotate a series of 3d coordinates clockwise looking down an axis.

    Args:
        arr (ndarray):
            The coordinates to be rotated. Dimensions must be (n_coords, 3).

        angle (float):
            The angle by which to rotate. Positive means clockwise.

        axis (vector arr):
            The axis to rotate around.

        point (vector arr):
            The point to rotate around. If none, defaults to the origin.

    Returns:
        rotated_arr (ndarray): The rotated array.
    '''

    # Get the rotation matrix out
    rot_matrix_4d = transformations.rotation_matrix( angle, axis, point )

    # Do only in 3d
    rot_matrix = rot_matrix_4d[:3, :3]

    # Apply the rotation
    rotated_arr = np.dot( arr, rot_matrix )

    # Set any low values to 0
    rotated_arr = np.where( np.abs( rotated_arr ) <= 1e-15, 0., rotated_arr )

    return rotated_arr

########################################################################


def align_axes(
    arr,
    new_z_axis,
    point = np.array([0., 0., 0.]),
    align_frame = True,
):
    '''Rotate a series of 3d coordinates s.t. the new z axis is aligned with
    the input vector new_z_axis

    Args:
        arr (ndarray):
            The coordinates to be rotated. Dimensions must be (n_coords, 3).

        new_z_axis (vector arr):
            The vector along which the new z axis will be aligned.

        point (vector arr):
            The point to rotate around. If none, defaults to the origin.

        align_frame (bool):
            If True, the coordinates are aligned along new_z_axis, but
            the data keep their same coordinates as before. I.e. we rotate our
            perspective s.t. new_z_axis is the new z axis.
            If False, we rotate the data instead, and keep the coordinates
            the same. I.e. we rotate the data z-axis to be aligned with
            new_z_axis, but we still view the data from the original
            perspective.
            This just changes the sign of the angle of rotation.

    Returns:
        rotated_arr (ndarray): The rotated array.
    '''
    z_axis = np.array([0., 0., 1.])

    if align_frame:
        angle_sign = 1.
    else:
        angle_sign = -1.

    # If nothing to rotate, don't try.
    if np.allclose( z_axis, new_z_axis ):
        return arr

    # Find the angle we want to rotate by
    angle = np.arccos( np.dot( z_axis, new_z_axis ) / ( np.linalg.norm( z_axis ) * np.linalg.norm( new_z_axis ) ) )
    # Based on how we define our angles, we need to multiply by 1 to rotate in
    # the opposite direction
    angle *= angle_sign

    # Find the vector we want to rotate around
    rot_vec = np.cross( z_axis, new_z_axis )

    # Rotate the array. Steps: 1. Move to the point you'll rotate around. 2. Rotate. 3. Move back.
    arr_to_be_rotated = arr - point
    rotated_arr_displaced = rotate_coordinates( arr_to_be_rotated, angle, rot_vec )
    rotated_arr = rotated_arr_displaced + point

    return rotated_arr

########################################################################


def contiguous_regions( condition ):
    '''Finds contiguous True regions of the boolean array "condition".
    Taken from https://stackoverflow.com/a/4495197

    Args:
        condition (boolean array-like):
            Input array.

    Returns:
        2D array-like:
            The first column is the start index of the region and the
            second column is the end index.
    '''

    # Find the indicies of changes in "condition"
    d = np.diff(condition)
    idx, = d.nonzero()

    # We need to start things after the change in
    # "condition". Therefore,
    # we'll shift the index by 1 to the right.
    idx += 1

    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]

    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, condition.size]  # Edit

    # Reshape the result into two columns
    idx.shape = (-1, 2)
    return idx

########################################################################


def cumsum2d( arr, axes=[0, 1], directions=[1, 1], ):
    '''Get the cumulative sum of an array.

    Args:
        arr (array-like) :
            Array to get the sum of.

        axes (list) :
            What axes to sum over.

        directions (list) :
            What direction to sum along.
                1 -> standard direction
                -1 -> reverse direction

    Returns:
        summed_arr (np.ndarray) :
            The i,j element of summed_arr is the sum of arr[:i+1,:j+1]
    '''

    summed_arr = np.copy( arr )
    for i, axis in enumerate( axes ):

        # Interpret directions, watching out for mistakes
        if directions[i] == 1:
            flip = False
        elif directions[i] == -1:
            flip = True
        else:
            raise Exception( "Unknown direction, {}".format( directions[i] ) )

        # Flip to change direction for summing
        if flip:
            summed_arr = np.flip( summed_arr, axis=axis )

        # Do the sum
        summed_arr = np.cumsum( summed_arr, axis=axis )

        # Flip back so that the array itself is unchanged
        if flip:
            summed_arr = np.flip( summed_arr, axis=axis )

    return summed_arr

########################################################################

def smooth_arr( arr, width=20, std=10 ):
    '''Apply a gaussian filter to an array.
    Credit to Cameron Hummels for the nice simple code I'm using.

    Args:
        arr (1D np.ndarray) :
            array to smooth

        width (int) :
            Number of elements over which to smooth.

        std (int) :
            Sigma of the guassian for which to smooth.

    Returns:
        smoothed_arr (1D np.ndarray) :
            Smoothed version of arr.
    '''

    b = scipy.signal.gaussian( width, std )
    smoothed_arr = scipy.ndimage.filters.convolve1d( arr, b/b.sum() )

    return smoothed_arr

def smooth( x, window_len=11, window='flat' ):                                    
    '''Simple moving-window signal smoothing (from scipy cookbook)

    Args:
        x (1D np.ndarray) :
            array to smooth

        wind_len (int) :
            Number of elements over which to smooth.

        window (str) :
            Type of window.

    Returns:
        smoothed_arr (1D np.ndarray) :
            Smoothed version of arr.
    '''

    if x.ndim != 1:                                                          
        raise ValueError( "smooth only accepts 1 dimension arrays." )
    if x.size < window_len:                                                  
        raise ValueError( "Input vector needs to be bigger than window size." )
    if window_len<3:                                                         
        return x                                                         
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']: 
        raise ValueError( "Window is not one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'" )

    s=np.r_[2*x[0]-x[window_len-1::-1],x,2*x[-1]-x[-1:-window_len:-1]]       

    if window == 'flat': #moving average                                     
        w=np.ones(window_len,'d')                                        
    else:                                                                    
        w=eval('np.'+window+'(window_len)')                              

    y=np.convolve(w/w.sum(),s,mode='same')                                   
    return y[window_len:-window_len+1]    
