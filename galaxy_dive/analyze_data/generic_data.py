#!/usr/bin/env python
'''Class for analyzing simulation data.

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

# Base python imports
import copy
from functools import wraps
import h5py
import numpy as np
import numpy.testing as npt
import warnings
import galaxy_dive.utils.constants as constants
import galaxy_dive.utils.utilities as utilities

########################################################################

# For default values
default = object()

########################################################################
########################################################################

class GenericData( object ):
  '''Very generic data class, with getting and masking functionality.

    Args:
      key_parser (object) : KeyParser instance to use to interpret data keys.
      data_masker (object) : DataMasker instance to use to filter and mask data.
      verbose (bool) : Print out additional information.
      z_sun (float) : Used mass fraction for solar metallicity.
    '''

  @utilities.store_parameters
  def __init__( self,
    key_parser = None,
    data_masker = None,
    verbose = False,
    z_sun = constants.Z_MASSFRAC_SUN,
    **kwargs ):

    # For storing and creating masks to pass the data
    if data_masker is None:
      self.data_masker = DataMasker( self )

    # Setup a data key parser
    if key_parser is None:
      self.key_parser = DataKeyParser()

  ########################################################################
  # Properties
  ########################################################################

  @property
  def length_scale( self ):
    '''Property for fiducial length scale. By default, is 1.
    However, a more advanced subclass might set this differently, or this might
    change in the future.
    '''

    # TODO: Address this.
    raise Exception( "Current thinking: this should not be called." )

    return 1.

  ########################################################################

  @property
  def velocity_scale( self ):
    '''Property for fiducial velocity scale. By default is 1.
    However, a more advanced subclass might set this differently, or this might
    change in the future.
    '''

    # TODO: Address this.
    raise Exception( "Current thinking: this should not be called." )

    return 1.

  ########################################################################

  @property
  def metallicity_scale( self ):
    '''Property for fiducial metallicity scale. By default is z_sun
    However, a more advanced subclass might set this differently, or this might
    change in the future.
    '''

    return self.z_sun

  ########################################################################

  @property
  def base_data_shape( self ):
    '''Property for simulation redshift.'''

    if not hasattr( self, '_base_data_shape' ):
      self._base_data_shape = self.data.values()[0].shape

    return self._base_data_shape

  @base_data_shape.setter
  def base_data_shape( self, value ):
    '''Setting function for simulation redshift property.'''

    # If we try to set it, make sure that if it already exists we don't change it.
    if hasattr( self, '_base_data_shape' ):
      assert self._base_data_shape == value

    else:
      self._base_data_shape = value

  ########################################################################
  # Data Retrieval
  ########################################################################

  def get_data( self, data_key, sl=None, data_storage=default ):
    '''Get the data from within the class. Only for getting the data. No post-processing or changing the data
    (putting it in particular units, etc.) The idea is to calculate necessary quantities as the need arises,
    hence a whole function for getting the data.

    Args:
      data_key (str) : Key in the data dictionary for the key we want to get
      sl (slice) : Slice of the data, if requested.

    Returns:
      data (np.ndarray) : Requested data.
    '''

    if data_storage is default:
      data_storage = self.data

    # Loop through, handling issues
    n_tries = 10
    for i in range( n_tries ):
      try:

        # Arbitrary functions of the data
        if data_key == 'Function':

          raise Exception( "TODO: Test this" )

          # Use the keys to get the data
          function_data_keys = self.kwargs['function_args']['function_data_keys']
          input_data = [ self.get_data(function_data_key) for function_data_key in function_data_keys ]

          # Apply the function
          data = self.kwargs['function_args']['function']( input_data )

        # Other
        else:
          data = self.data[data_key]

      # Calculate missing data
      except KeyError as e:
        self.handle_data_key_error( data_key )
        continue

      break

    if 'data' not in locals().keys():
      raise KeyError( "After {} tries, unable to find or create data_key, {}".format( i+1, data_key ) )

    if sl is not None:
      return data[sl]

    return data

  ########################################################################

  def get_processed_data( self, data_key, data_method=default, *args, **kwargs ):
    '''Get post-processed data. (Accounting for fractions, log-space, etc.).

    Args:
      data_key (str) : What data to get.
      data_method (str) : What method to use for getting the data itself. Defaults to using self.get_data
      *args, **kwargs : Passed to get_data()

    Returns:
      processed_data (np.ndarray) : Requested data, including formatting.
    '''

    # Account for fractional data keys
    data_key, fraction_flag = self.key_parser.is_fraction_key( data_key )

    # Account for logarithmic data
    data_key, log_flag = self.key_parser.is_log_key( data_key )

    # Choose what method we're using for getting data.
    if data_method is default:
      get_data_method = self.get_data
    else:
      get_data_method = getattr( self, data_method )

    # Get the data and make a copy to avoid altering
    data = copy.deepcopy( get_data_method( data_key, *args, **kwargs ) )

    # Actually calculate the fractional data
    if fraction_flag:

      # Put distances in units of the virial radius
      if self.key_parser.is_position_key( data_key ):

        data /= self.length_scale

      # Put velocities in units of the circular velocity
      elif self.key_parser.is_velocity_key( data_key ):
        data /= self.velocity_scale

      # Put the metallicity in solar units
      elif data_key == 'Z':
        data /= self.metallicity_scale

      else:
        raise Exception('Fraction type not recognized')

    # Make appropriate units into log
    if log_flag:
      data =  np.log10( data )

    return data

  ########################################################################

  def get_selected_data( self, *args, **kwargs ):
    '''Wrapper for getting masked data.'''

    return self.data_masker.get_selected_data( *args, **kwargs )

  def mask_data( self, *args, **kwargs ):
    '''Wrapper for masking data.'''

    return self.data_masker.mask_data( *args, **kwargs )

  ########################################################################

  def shift( self, data, data_key ):
    '''Shift or multiply the data by some amount. Note that this is applied after logarithms are applied.

    data : data to be shifted
    data_key : Data key for the parameters to be shifted
    Parameters are a subdictionary of self.kwargs
    'data_key' : What data key the data is shifted for
    '''

    raise Exception( "TODO: Test this" )

    shift_p = self.kwargs['shift']

    # Exit early if not the right data key
    if shift_p['data_key'] != data_key:
      return 0

    # Shift by the mass metallicity relation.
    if 'MZR' in shift_p:

      # Gas-phase ISM MZR fit from Ma+2015.
      # Set to be no shift for gas with the mean metallicity of LLS at z=0 (This has a value of -0.45, as I last calculated)
      log_shift = 0.93*(np.exp(-0.43*self.redshift) - 1.) - 0.45

      data -= log_shift

  ########################################################################

  def handle_data_key_error( self, data_key ):
    '''Method for attempting to generate data on the fly.

    Args:
      data_key (str) : Type of data to attempt to generate data for.
    '''

    raise Exception( "This method should be replaced in the subclass!" )

  ########################################################################
  # Meta Methods (for doing generic things with class methods)
  ########################################################################

  def iterate_over_method( self, method_str, iter_arg, iter_values, method_args ):
    '''Iterate over a specified method, and get the results out.

    Args:
      method_str (str) :
        Which method to use.

      iter_arg (str) :
        Which argument of the method to iterate over.

      iter_values (list of values) :
        Which values to change.

      method_args (dict) :
        Default args to pass to the method

    Returns:
      results (list) : [ method( **used_args ) for used_args in all_variations_of_used_args ]
    '''

    method = getattr( self, method_str )

    results = []
    for iter_value in iter_values:

      method_args[iter_arg] = iter_value

      result = method( **method_args )

      results.append( result )

    return results

########################################################################
########################################################################

class DataKeyParser( object ):
  '''Class for parsing data_keys provided to SimulationData.'''

  ########################################################################

  def is_position_key( self, data_key ):
    '''Checks if the data key deals with position primarily.

    Args:
      data_key (str) : Data key to check.

    Returns:
      is_position_key (bool) : True if deals with position.
    '''

    return ( data_key[0] == 'R' ) | ( data_key == 'P' )

  ########################################################################

  def is_velocity_key( self, data_key ):
    '''Checks if the data key deals with velocity primarily.

    Args:
      data_key (str) : Data key to check.

    Returns:
      is_velocity_key (bool) : True if deals with velocity.
    '''

    return ( data_key[0] == 'V' )


  ########################################################################

  def is_fraction_key( self, data_key ):
    '''Check if the data should be some fraction of its relevant scale.

    Args:
      data_key (str) : Data key to check.

    Returns:
      is_fraction_key (bool) : True if the data should be scaled as a fraction of the relevant scale.
    '''

    fraction_flag = False
    if data_key[-1] == 'f':
      data_key = data_key[:-1]
      fraction_flag = True

    return data_key, fraction_flag

  ########################################################################

  def is_log_key( self, data_key ):
    '''Check if the data key indicates the data should be put in log scale.

    Args:
      data_key (str) : Data key to check.

    Returns:
      is_log_key (bool) : True if the data should be taken log10 of
    '''

    log_flag = False
    if data_key[0:3] == 'log':
      data_key = data_key[3:]
      log_flag = True

    return data_key, log_flag

########################################################################
########################################################################

class DataMasker( object ):

  def __init__( self, data_object ):
    '''Class for masking data.

    Args:
      data_object (GenericData object) : Used for getting data to find mask ranges.
    '''

    self.data_object = data_object

    self.masks = []
    self.optional_masks = {}

  ########################################################################

  def mask_data( self,
    data_key,
    data_min = default,
    data_max = default,
    data_value = default,
    custom_mask = default,
    return_or_store = 'store',
    optional_mask = False,
    mask_name = default,
    *args, **kwargs ):
    '''Get only the particle data within a certain range. Note that it retrieves the processed data.

    Args:
      data_key (str) :
        Data key to base the mask off of.

      data_min (float) :
        Everything below data_min will be masked.

      data_max (float) :
        Everything above data_max will be masked.

      data_value (float) :
        Everything except for data_value will be masked.

      custom_mask (bool) :
        If provided, take in a custom mask instead, using data_key as the label for the mask.

      return_or_store (str) :
        Whether to store the mask as part of the masks dictionary, or to return it.

      optional_mask (bool) :
        If True, store in the dictionary self.optional_masks instead.

      mask_name (str) :
        What name to associate with this mask? Currently only relevant if optional_mask is True.
        By default uses the data_key as the name.

      *args, **kwargs :
        Passed to self.data_object.get_processed_data()

    Returns:
      data_mask (np.array of bools) :
        If requested, the mask for data in that range.

    Modifies:
      self.masks (list of dicts) :
        Appends a dictionary describing the mask.
    '''

    # Process what type of mask to get
    mask_outside = ( data_min is not default ) and ( data_max is not default )
    mask_discrete = data_value is not default
    mask_custom = custom_mask is not default
    assert ( mask_outside + mask_discrete + mask_custom )==1, "Bad combination of masks!"

    if not mask_custom:
      data = self.data_object.get_processed_data( data_key, *args, **kwargs )

    # Get the mask
    if mask_outside:
      data_ma = np.ma.masked_outside( data, data_min, data_max )
      mask = data_ma.mask
    elif mask_discrete:
      mask = np.invert( data == data_value )
    elif mask_custom:
      mask = custom_mask
    else:
      raise NameError( "Unspecified combination of data masking." )

    # Handle the case where an entire array is masked or none of it is masked
    # (Make into an array for easier combination with other masks)
    if mask.size == 1:
      mask = mask*np.ones( shape=data.shape, dtype=bool )

    if return_or_store == 'store':

      if mask_outside:
        mask_dict = {'data_key': data_key, 'data_min': data_min, 'data_max': data_max, 'mask': mask}
      elif mask_discrete:
        mask_dict = {'data_key': data_key, 'data_value' : data_value, 'mask': mask}
      elif mask_custom:
        mask_dict = {'data_key': data_key, 'custom_mask' : True, 'mask': mask}

      if optional_mask:
        if mask_name is default:
          mask_name = data_key

        assert mask_name not in self.optional_masks.keys(), "A mask with that name already exists!"

        self.optional_masks[mask_name] = mask_dict

      else:
        self.masks.append( mask_dict )

    elif return_or_store == 'return':
      return mask

    else:
      raise Exception('NULL return_or_store')

  ########################################################################

  def get_total_mask( self, optional_masks=None ):
    '''Get the result of combining all masks in the data.

    Args:
      optional_masks (list-like) : List of names of optional masks to use (must be found in self.optional_masks).

    Returns:
      total_mask (np.array of bools) : Result of all masks.
    '''

    # Compile masks
    all_masks = []
    for mask_dict in self.masks:
      all_masks.append( mask_dict['mask'] )

    # Get any requested optional masks
    if optional_masks is not None:
      for optional_mask in optional_masks:
        all_masks.append( self.optional_masks[optional_mask]['mask'] )

    # Combine masks
    return np.any( all_masks, axis=0, keepdims=True )[0]

  ########################################################################

  def get_selected_data( self,
    data_key,
    mask = 'total',
    optional_masks = None,
    sl = None,
    apply_slice_to_mask = True,
    fix_invalid = False,
    compress = True,
    mask_multidim_data = True,
    *args, **kwargs ):
    '''Get all the data that doesn't have some sort of mask applied to it. Use the processed data.

    Args:
      data_key (str) : Data key to get the data for.
      mask (str or np.array of bools) : Mask to apply. If none, use the total mask.
      optional_masks (list-like) : List of names of optional masks to use (must be found in self.optional_masks).
      sl (slice) : Slice to apply to the data
      apply_slice_to_mask (bool) : Whether or not to apply the same slice you applied to the data to the mask.
      fix_invalid (bool) : Whether or not to also mask invalid data.
      compress (bool) : Whether or not to return compressed data.
      mask_multidim_data (bool) : Whether or not to change the mask to fit multidimensional data.
      *args, **kwargs : Passed to get_proceesed_data.

    Returns:
      data_ma (np.array) : Compressed masked data. Because it's compressed it may not have the same shape as the
        original data.
    '''

    data = self.data_object.get_processed_data( data_key, sl=sl, *args, **kwargs )

    # Get the appropriate mask
    if isinstance( mask, np.ndarray ):
      used_mask = mask
    elif isinstance( mask, bool ) or isinstance( mask, np.bool_ ):
      if not mask:
        if fix_invalid:
          return np.ma.fix_invalid( data ).compressed()
        else:
          return data

      raise Exception( "All data is masked." )
    elif mask == 'total':
      used_mask = self.get_total_mask( optional_masks=optional_masks )
    else:
      raise KeyError( "Unrecognized type of mask, {}".format( mask ) )

    if ( sl is not None ) and apply_slice_to_mask:
      used_mask = used_mask[sl]

    if fix_invalid:
      array_to_ma_array_fn = np.ma.fix_invalid
    else:
      array_to_ma_array_fn = np.ma.array

    # Test for if the data fits the mask, or if it's multi-dimensional
    if mask_multidim_data:
      if len( data.shape ) > len( self.data_object.base_data_shape ):
        data_ma = [ array_to_ma_array_fn( data_part, mask=used_mask ) for data_part in data ]
        data_ma = [ data_ma_part.compressed() for data_ma_part in data_ma ]
        data_ma = np.array( data_ma )

      else:
        data_ma = array_to_ma_array_fn( data, mask=used_mask )

        if compress:
          data_ma = data_ma.compressed()

    else:
      data_ma = array_to_ma_array_fn( data, mask=used_mask )

      if compress:
        data_ma = data_ma.compressed()

    return data_ma

  ########################################################################

  def clear_masks( self, clear_optional_masks=False ):
    '''Reset the masks in total to nothing.

    Modifies:
      self.masks (lists) : Sets to empty
    '''

    self.masks = []

    if clear_optional_masks:
      self.optional_masks = {}
