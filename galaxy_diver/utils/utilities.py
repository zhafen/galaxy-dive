
import numpy as np
import transformations

########################################################################
# Functions for doing simple things to data 
########################################################################

# Takes in an array and raises all values in the array below the floor to the floor value.

def apply_floor(arr, floor):
  return np.where( arr >= floor, arr, floor*np.ones(arr.shape))

########################################################################

# Merge two dictionaries, with dict2's keyword overridding dict1's keywords where relevant.

def merge_dict(dict1, dict2):

  merged_dict = copy.deepcopy(dict1)

  merged_dict.update(dict2)

  return merged_dict

########################################################################

# Downsample an array

def block_mean(ar, fact):
  assert isinstance(fact, int), type(fact)

  sx, sy = ar.shape
  X, Y = np.ogrid[0:sx, 0:sy]
  regions = sy/fact * (X/fact) + Y/fact

  res = ndimage.mean(ar, labels=regions, index=np.arange(regions.max() + 1))
  res.shape = (sx/fact, sy/fact)

  return res

########################################################################

def cartesian(arrays, out=None):
  """
  Generate a cartesian product of input arrays.
  Produced on Stack Overflow: http://stackoverflow.com/a/1235363/6651313

  Parameters
  ----------
  arrays : list of array-like
      1-D arrays to form the cartesian product of.
  out : ndarray
      Array to place the cartesian product in.

  Returns
  -------
  out : ndarray
      2-D array of shape (M, len(arrays)) containing cartesian products
      formed of input arrays.

  Examples
  --------
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
  out[:,0] = np.repeat(arrays[0], m)
  if arrays[1:]:
      cartesian(arrays[1:], out=out[0:m,1:])
      for j in xrange(1, arrays[0].size):
          out[j*m:(j+1)*m,1:] = out[0:m,1:]
  return out

########################################################################

def rotate_coordinates(arr, angle, axis, point=None):
  '''Rotate a series of 3d coordinates clockwise looking down an axis.

  Args --
  arr : The coordinates to be rotated. Dimensions must be (n_coords, 3).
  angle : The angle by which to rotate. Positive means clockwise.
  axis : The axis to rotate around.
  point : The point to rotate around. If none, defaults to the origin.
  '''

  # Get the rotation matrix out
  rot_matrix_4d = transformations.rotation_matrix(angle, axis, point)

  # Do only in 3d
  rot_matrix = rot_matrix_4d[:3, :3]

  # Apply the rotation
  rotated_arr = np.dot(arr, rot_matrix)

  # Set any low values to 0
  rotated_arr = np.where(np.abs(rotated_arr) <= 1e-15, 0., rotated_arr)

  return rotated_arr

########################################################################

def align_axes(arr, new_z_axis, point=np.array([0., 0., 0.]) ):
  '''Rotate a series of 3d coordinates s.t. the new z axis is aligned with the input vector new_z_axis

  Args --
  arr : The coordinates to be rotated. Dimensions must be (n_coords, 3).
  new_z_axis : The vector along which the new z axis will be aligned.
  point : The point to rotate around. If none, defaults to the origin.
  '''
  z_axis = np.array([0., 0., 1.])

  # If nothing to rotate, don't try.
  if np.allclose(z_axis, new_z_axis):
    return arr

  # Find the angle we want to rotate by
  angle = np.arccos( np.dot(z_axis, new_z_axis) / ( np.linalg.norm(z_axis)*np.linalg.norm(new_z_axis) ) )

  # Find the vector we want to rotate around
  rot_vec = np.cross(z_axis, new_z_axis)

  # Rotate the array. Steps: 1. Move to the point you'll rotate around. 2. Rotate. 3. Move back.
  arr_to_be_rotated = arr - point
  rotated_arr_displaced = rotate_coordinates(arr_to_be_rotated, angle, rot_vec)
  rotated_arr = rotated_arr_displaced + point

  return rotated_arr
  
