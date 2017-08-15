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
import string
import warnings

import galaxy_diver.utils.constants as constants

########################################################################
########################################################################

class GenericData( object ):
  '''Very generic data class, with getting and masking functionality.'''

  def __init__( self,
    verbose = False,
    z_sun = constants.Z_MASSFRAC_SUN,
    **kwargs ):
    '''Initialize.

    Args:
      verbose (bool) : Print out additional information.
      z_sun (float) : Used mass fraction for solar metallicity.
    '''

    # Store the arguments
    for arg in locals().keys():
      setattr( self, arg, locals()[arg] )
    self.kwargs = kwargs

    # For storing and creating masks to pass the data
    self.data_masker = DataMasker( self )

    # Setup a data key parser
    self.key_parser = DataKeyParser()

  ########################################################################
  # Properties
  ########################################################################

  @property
  def length_scale( self ):
    '''Property for fiducial length scale. By default, is 1.
    However, a more advanced subclass might set differently, or this might
    change in the future.
    '''

    return 1.

  ########################################################################

  @property
  def velocity_scale( self ):
    '''Property for fiducial velocity scale. By default is 1.
    However, a more advanced subclass might set this differently, or this might
    change in the future.
    '''

    return 1.

  ########################################################################

  @property
  def metallicity_scale( self ):
    '''Property for fiducial metallicity scale. By default is z_sun
    However, a more advanced subclass might set differently, or this might
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

  def get_data( self, data_key, sl=None ):
    '''Get the data from within the class. Only for getting the data. No post-processing or changing the data
    (putting it in particular units, etc.) The idea is to calculate necessary quantities as the need arises,
    hence a whole function for getting the data.

    Args:
      data_key (str) : Key in the data dictionary for the key we want to get
      sl (slice) : Slice of the data, if requested.

    Returns:
      data (np.ndarray) : Requested data.
    '''

    data = self.data[data_key]

    if sl is not None:
      return data[sl]
  
    return data

  ########################################################################

  def get_processed_data( self, data_key, sl=None ):
    '''Get post-processed data. (Accounting for fractions, log-space, etc.).'''

    # Account for fractional data keys
    data_key, fraction_flag = self.key_parser.is_fraction_key( data_key )

    # Account for logarithmic data
    data_key, log_flag = self.key_parser.is_log_key( data_key )

    # Get the data and make a copy to avoid altering
    data_original = self.get_data( data_key, sl=sl )
    data = copy.deepcopy( data_original )

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

  def get_masked_data( self, *args, **kwargs ):
    '''Wrapper for getting masked data.'''

    return self.data_masker.get_masked_data( *args, **kwargs )

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

  def __init__( self, generic_data ):
    '''Class for masking data.

    Args:
      generic_data (GenericData object) : Used for getting data to find mask ranges.
    '''

    self.generic_data = generic_data

    self.masks = []

  ########################################################################

  def mask_data( self, data_key, data_min, data_max, return_or_store='store' ):
    '''Get only the particle data within a certain range. Note that it retrieves the processed data.

    Args:
      data_key (str) : Data key to base the mask off of.
      data_min (float) : Everything below data_min will be masked.
      data_max (float) : Everything above data_max will be masked.
      return_or_store (str) : Whether to store the mask as part of the masks dictionary, or to return it.

    Returns:
      data_mask (np.array of bools) : If requested, the mask for data in that range.

    Modifies:
      self.masks (list of dicts) : Appends a dictionary describing the mask.
    '''

    # Get the mask
    data = self.generic_data.get_processed_data( data_key )
    data_ma = np.ma.masked_outside( data, data_min, data_max )

    # Handle the case where an entire array is masked or none of it is masked
    # (Make into an array for easier combination with other masks)
    if data_ma.mask.size == 1:
      data_ma.mask = data_ma.mask*np.ones( shape=data.shape, dtype=bool )

    if return_or_store == 'store':
      self.masks.append( {'data_key': data_key, 'data_min': data_min, 'data_max': data_max, 'mask': data_ma.mask} )

    elif return_or_store == 'return':
      return data_ma.mask

    else:
      raise Exception('NULL return_or_store')

  ########################################################################

  def get_total_mask( self ):
    '''Get the result of combining all masks in the data.

    Returns:
      total_mask (np.array of bools) : Result of all masks.
    '''

    # Compile masks
    all_masks = []
    for mask_dict in self.masks:
      all_masks.append( mask_dict['mask'] )

    # Combine masks
    return np.any( all_masks, axis=0, keepdims=True )[0]

  ########################################################################

  def get_masked_data( self, data_key, mask='total', sl=None, apply_slice_to_mask=True ):
    '''Get all the data that doesn't have some sort of mask applied to it. Use the processed data.

    Args:
      data_key (str) : Data key to get the data for.
      mask (str or np.array of bools) : Mask to apply. If none, use the total mask.
      sl (slice) : Slice to apply to the data
      apply_slice_to_mask (bool) : Whether or not to apply the same slice you applied to the data to the mask.

    Returns:
      data_ma (np.array) : Compressed masked data. Because it's compressed it may not have the same shape as the
        original data.
    '''
    data = self.generic_data.get_processed_data( data_key, sl=sl )

    # Get the appropriate mask
    if isinstance( mask, np.ndarray ):
      used_mask = mask
    elif isinstance( mask, bool ) or isinstance( mask, np.bool_ ):
      if not mask:
        return data
      raise Exception( "All data is masked." )
    elif mask == 'total':
      used_mask = self.get_total_mask()
    else:
      raise KeyError( "Unrecognized type of mask, {}".format( mask ) )

    if ( sl is not None ) and apply_slice_to_mask:
      used_mask = used_mask[sl]

    # Test for if the data fits the mask, or if it's multi-dimensional
    if len( data.shape ) > len( self.generic_data.base_data_shape ):
      data_ma = [ np.ma.array( data_part, mask=used_mask ) for data_part in data ]
      data_ma = [ data_ma_part.compressed() for data_ma_part in data_ma ]
      data_ma = np.array( data_ma )

    else:
      data_ma = np.ma.array( data, mask=used_mask )
      data_ma = data_ma.compressed()

    return data_ma

  ########################################################################

  def clear_masks( self ):
    '''Reset the masks in total to nothing.
    
    Modifies:
      self.masks (lists) : Sets to empty
    '''

    self.masks = []
