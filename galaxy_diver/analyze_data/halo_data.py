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

import galaxy_diver.read_data.ahf as read_ahf
import galaxy_diver.read_data.metafile as read_metafile
import galaxy_diver.utils.utilities as utilities
import galaxy_diver.analyze_data.generic_data as generic_data

########################################################################
########################################################################

class HaloData( generic_data.GenericData ):

    @utilities.store_parameters
    def __init__(
        self,
        data_dir,
        halo_finder = 'AHF',
        mt_kwargs = {},
        **kwargs
    ):
        '''Constructor for HaloData

        Args:
            data_dir (str) :
                Directory storing the data.

            halo_finder (str) :
                What code was the halo data generated with?
                Options: 'AHF', 'Rockstar'

            mt_kwargs (dict) :
                When loading merger tree halo files,
                additional arguments should be passed here.

            kwargs (dict) :
                When loading single-snapshot halo files, additional arguments
                should be passed here.
        '''

        if halo_finder == 'AHF':
            self.data_reader = read_ahf.AHFReader( data_dir )
        elif halo_finder == 'Rockstar':
            self.data_reader = read_rockstar.RockstarReader( data_dir )

        key_parser = HaloKeyParser()

        super( HaloData, self ).__init__( key_parser=key_parser )

    ########################################################################
    # Attributes
    ########################################################################

    def __getattr__( self, attr ):
        '''There are certain attributes held by self.data_reader that we
        would like to easily access here.
        '''

        data_reader_used_attrs = [
            'halos',
            'halos_snum',
            'mtree_halo_filepaths',
            'index', 
            'tag',
        ]

        if attr in data_reader_used_attrs:
            return getattr( self.data_reader, attr )

    ########################################################################
    # Properties
    ########################################################################

    @property
    def halos( self ):
        '''Attribute for accessing halo data.
        '''

        return self.data_reader.halos

    ########################################################################

    @property
    def halos_snum( self ):
        '''Attribute for accessing halo data.
        '''

        return self.data_reader.halos_snum

    ########################################################################

    @property
    def mtree_halos( self ):
        '''Attribute for accessing merger tree data.
        '''

        if not hasattr( self.data_reader, 'mtree_halos' ):

            self.data_reader.get_mtree_halos( **self.mt_kwargs )

        return self.data_reader.mtree_halos

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

        # Check if the key is not unique to the halo data,
        # and if so, get the keys used for this particular data
        if data_key in self.data_reader.general_use_data_names:
            data_key = self.data_reader.general_use_data_names[data_key]

        # Load the necessary data
        self.get_halos( snum )

        return self.data_reader.halos[data_key].values

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

    def get_halos( self, snum, **kwargs ):
        '''Get halo data.

        Args:
            snum (int) : What snapshot to get the halo data for.

            **kwargs :
                Additional arguments for loading the data.
                In the event of conflict with those provided when constructing
                the instance, arguments provided here will be preferred.
        '''

        # Get the used arguments
        used_kwargs = utilities.merge_two_dicts(
            kwargs,
            self.kwargs,
        )

        # Try to load additional postprocessed data
        self.data_reader.get_halos( snum, **used_kwargs )
        try:
            self.data_reader.get_halos_add( snum, **used_kwargs )
        except NameError:
            pass

    ########################################################################

    def get_mtree_halos( self, **kwargs ):
        '''Get halo data.

        Args:
            snum (int) : What snapshot to get the halo data for.

            **kwargs :
                Additional arguments for loading the data.
                In the event of conflict with those provided when constructing
                the instance, arguments provided here will be preferred.
        '''

        # Get the used arguments
        used_kwargs = utilities.merge_two_dicts(
            kwargs,
            self.mt_kwargs,
        )

        # Try to load additional postprocessed data
        self.data_reader.get_mtree_halos( **used_kwargs )

    ########################################################################

    def get_n_halos( self, snum ):
        '''Get the number of halos (non-tracked) at a given redshift.'''

        self.get_halos( snum )

        return self.data_reader.halos.index.size

    ########################################################################

    def get_masked_data( self, *args, **kwargs ):

        return super( HaloData, self ).get_masked_data(
            mask_multidim_data = False,
            *args, **kwargs
        )

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

