#!/usr/bin/env python
'''Tools for handling halo data.

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import copy
import glob
import numpy as np
import os
import pandas as pd

import galaxy_dive.read_data.ahf as read_ahf
import galaxy_dive.read_data.metafile as read_metafile
import galaxy_dive.utils.utilities as utilities
import galaxy_dive.analyze_data.generic_data as generic_data

########################################################################
########################################################################

class HaloData( generic_data.GenericData ):

  @utilities.store_parameters
  def __init__( self, data_dir, tag=None, index=None, mt_kwargs={} ):
    '''Constructor for HaloData

    Args:
      data_dir (str) : Directory storing the data.
      tag (str) : If provided, what is an identifying tag for the halo data?
      index (int) : If provided, what is the final snapshot number for the halo data? Necessary for some AHF data.
      mt_kwargs (dict) : When loading merger tree halo files, additional arguments should be passed here.
    '''

    self.ahf_reader = read_ahf.AHFReader( data_dir )

    key_parser = HaloKeyParser()

    super( HaloData, self ).__init__( key_parser=key_parser )

  ########################################################################
  # Properties
  ########################################################################

  @property
  def mt_halos( self ):
    '''Attribute for accessing merger tree data.
    '''

    if not hasattr( self.ahf_reader, 'mtree_halos' ):

      self.ahf_reader.get_mtree_halos( index=self.index, tag=self.tag, **self.mt_kwargs )

    return self.ahf_reader.mtree_halos

  ########################################################################
  # Data Retrieval
  ########################################################################

  def get_data( self, data_key, snum ):
    '''Get halo data at a specific snapshot.

    Args:
      data_key (str) : What data to get.
      snum (int) : What snapshot to open.

    Returns:
      data (np.ndarray) : Requested data.
    '''

    self.ahf_reader.get_halos( snum )
    self.ahf_reader.get_halos_add( snum )

    return self.ahf_reader.ahf_halos[data_key].values

  ########################################################################

  def get_mt_data( self, data_key, mt_halo_id=0, snums=None, a_power=None, return_values_only=True ):
    '''Get halo data for a specific merger tree.

    Args:
      data_key (str) : What data to get.
      mt_halo_id (int) : What merger tree halo ID to select.
      snums (array-like) : If specified, get the values at these snapshots.
      a_power (float) : If given, multiply the result by the scale factor 1/(1 + redshift) to this power.
      return_values_only (bool) : If True, get rid of pandas data formatting

    Returns:
      mt_data (np.ndarray) : Requested data.
    '''

    mt_data = copy.copy( self.mt_halos[mt_halo_id][data_key] )

    # For converting coordinates
    if a_power is not None:
      mt_data *= ( 1. + self.get_mt_data( 'redshift', mt_halo_id ) )**-a_power

    if snums is not None:
      mt_data = mt_data.loc[snums]

    if return_values_only:
     mt_data = mt_data.values

    return mt_data

  ########################################################################

  def get_selected_data( self, *args, **kwargs ):

    return super( HaloData, self ).get_selected_data( mask_multidim_data=False, *args, **kwargs )

########################################################################
########################################################################

class HaloKeyParser( generic_data.DataKeyParser ):

  def get_radius_key( self, multiplier, length_scale ):
    '''Get a key for Halo data, based on a length scale and a multiple of it.

    Args:
      multiplier (float) :
        multiplier*length_scale defines the radius around the center of the halo(s).

      length_scale (str) :
        multiplier*length_scale defines the radius around the center of the halo(s).

    Returns:
      radius_key (str) :
        Combination of length_scale and multiplier.
    '''

    if np.isclose( multiplier, 1.0 ):
      radius_key = length_scale
    else:
      radius_key = '{}{}'.format( multiplier, length_scale )

    return radius_key

  ########################################################################

  def get_enclosed_mass_key( self, ptype, multiplier, length_scale ):
    '''Get a key for Halo data, corresponding to a data column that records an enclosed mass.

    Args:
      ptype (str) :
        The particle type for the enclosed mass.

      multiplier (float) :
        multiplier*length_scale defines the radius around the center of the halo within which to get the mass.

      length_scale (str) :
        multiplier*length_scale defines the radius around the center of the halo within which to get the mass.

    Returns:
      enclosed_mass_key (str)
    '''

    return 'M{}({})'.format( ptype, self.get_radius_key( multiplier, length_scale ) )

  ########################################################################

  def get_average_quantity_key( self, data_key, ptype, multiplier, length_scale ):
    '''Get a key for Halo data, corresponding to a data column that records the average quantity inside a galaxy.

    Args:
      data_key (str) :
        What the enclosed quantity is.

      ptype (str) :
        The particle type for the enclosed mass.

      multiplier (float) :
        multiplier*length_scale defines the radius around the center of the halo within which to get the mass.

      length_scale (str) :
        multiplier*length_scale defines the radius around the center of the halo within which to get the mass.

    Returns:
      average_quantity_key (str)
    '''

    return '{}{}({})'.format( data_key, ptype, self.get_radius_key( multiplier, length_scale ) )


  ########################################################################

  def get_velocity_at_radius_key( self, velocity_key, multiplier, length_scale ):
    '''Get a key for Halo data, corresponding to a data column that records the velocity at at a specified radius

      velocity_key (str) :
        What velocity to get.

      multiplier (float) :
        multiplier*length_scale defines the radius around the center of the halo within which to get the mass.

      length_scale (str) :
        multiplier*length_scale defines the radius around the center of the halo within which to get the mass.

    Returns:
      velocity_at_radius_key (str)
    '''

    return '{}({})'.format( velocity_key, self.get_radius_key( multiplier, length_scale ) )

