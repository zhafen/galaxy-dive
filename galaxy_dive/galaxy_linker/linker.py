#!/usr/bin/env python
'''Means to associate particles with galaxies and halos at any given time.

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import copy
import numpy as np
import scipy.spatial

import unyt

import galaxy_dive.analyze_data.halo_data as h_data
import galaxy_dive.utils.utilities as utilities

########################################################################
########################################################################


class GalaxyLinker( object ):
    '''Find the association with galaxies and halos for a given set of particles
    at a given redshift.
    '''

    @utilities.store_parameters
    def __init__(
        self,
        particle_positions,
        redshift,
        snum,
        hubble,
        galaxy_cut,
        length_scale,
        mt_length_scale = None,
        halo_length_scale = 'Rvir',
        particle_masses = None,
        minimum_criteria = 'n_star',
        minimum_value = 10,
        ids_to_return = None,
        ids_with_supplementary_data = [],
        supplementary_data_keys = [],
        halo_data = None,
        halo_finder = 'AHF',
        halo_data_dir = None,
        halo_file_tag = None,
        mtree_halos_index = None,
        main_mt_halo_id = None,
        low_memory_mode = True,
        memory_mode_divisions = 10,
    ):
        '''Initialize.

        Args:
            particle_positions (array-like) :
                Positions with dimensions (n_particles, 3).
                If given a np.ndarray, assumes pkpc units.
                Also accepts a unyt array.

            redshift (float) :
                Redshift the particles are at.

            snum (int) :
                Snapshot the particles correspond to.

            hubble (float) :
                Cosmological hubble parameter (little h)

            galaxy_cut (float) :
                The fraction of the length scale a particle must be inside to be
                counted as part of a galaxy.

            length_scale (str) :
                Anything within galaxy_cut*length_scale is counted as being
                inside the galaxy.

            mt_length_scale (str) :
                Same as length scale, but for merger tree halos. Defaults to
                length_scale.

            halo_length_scale (str) :
                Anything within halo_length_scale is counted as being
                inside the halo.

            particle_masses (np.ndarray, optional) :
                Masses of particles.

            minimum_criteria (str) :
                Options...
                'n_star' -- halos must contain a minimum number of stars to
                    count as containing a galaxy.
                'M_star' -- halos must contain a minimum stellar mass to count
                    as containing a galaxy.

            minimum_value (int or float) :
                The minimum amount of something (specified in minimum criteria)
                in order for a galaxy to count as hosting a halo.

            ids_to_return (list of strs, optional) :
                The types of id you want to get out.

            ids_with_supplementary_data (list of strs, optional) :
                What types of IDs should include supplementary data pulled
                from the halo files.

            supplementary_data_keys (list of strs, optional) :
                What data keys in the halo files should be accessed and
                included as part of supplementary data.

            halo_data (HaloData object, optional) :
                An instance of an object that retrieves halo data
                If not given initiate one using the halo_data_dir in kwargs

            halo_data_dir (str, optional) :
                Directory the AHF data is in. Necessary if halo_data is not
                provided.

            halo_finder (str, optional) :
                What code were the halos found with? Examples are 'AHF',
                'Rockstar'.

            halo_file_tag (int, optional) :
                What identifying tag to use for the merger tree files? Necessary
                if using merger tree information.

            mtree_halos_index (str or int, optional)  :
                The index argument to pass to AHFReader.get_mtree_halos().
                For most cases this should be the final snapshot number, but see
                AHFReader.get_mtree_halos's documentation.

            main_mt_halo_id (int, optional) :
                Index of the main merger tree halo.

            low_memory_mode (bool) :
                If True, use less memory at the cost of reduced performance.

            memory_mode_divisions (int) :
                When low_memory_mode is True, we reduce the memory cost by doing
                less work at once. This divides certain
                arrays into a number of divisions equal to
                memory_mode_divisions, and then the results are calculated for
                each division before bringing everything together.
        '''

        if self.mt_length_scale is None:
            self.mt_length_scale = length_scale

        # Setup the default halo_data
        if halo_data is None:
            self.halo_data = h_data.HaloData(
                self.halo_data_dir,
                halo_finder = self.halo_finder,
                mt_kwargs = { 
                    'index' : self.mtree_halos_index,
                    'tag' : self.halo_file_tag,
                },
        )

        if not isinstance( particle_positions, unyt.array.unyt_array ):
            particle_positions = copy.copy( particle_positions ) * unyt.kpc

        # In the case of a minimum stellar mass, we need to divide the minimum
        # value by 1/h when getting its values out.
        if self.minimum_criteria in [ 'M_star', 'Mvir']:
            self.min_conversion_factor = self.hubble
        else:
            self.min_conversion_factor = 1

        # Derived properties
        self.n_particles = self.particle_positions.shape[0]

    ########################################################################
    # General Properties Unique to an Instance
    ########################################################################

    @property
    def valid_halo_inds( self ):
        '''
        Returns:
            valid_halo_inds (np.ndarray) :
                Indices of *AHF_halos halos that satisfy our minimum criteria
                for containing a galaxy.
        '''

        if not hasattr( self, '_valid_halo_inds' ):

            # Apply a cut on containing a minimum amount of stars
            min_criteria = self.halo_data.get_data(
                self.minimum_criteria,
                snum = self.snum,
            )
            has_minimum_value = \
                min_criteria / self.min_conversion_factor >= self.minimum_value

            # Figure out which indices satisfy the criteria and choose only
            # those halos
            self._valid_halo_inds = np.where( has_minimum_value )[0]

        return self._valid_halo_inds

    ########################################################################

    @property
    def valid_halo_pos( self ):
        '''
        Returns:
            valid_halo_pos (np.ndarray) :
               Positions in proper kpc of positions we would like to link to.
        '''

        if not hasattr( self, '_valid_halo_pos' ):

            # Get the halo positions
            halo_pos_comov = np.array([
                self.halo_data.get_data( 'X', self.snum, units='kpc' ),
                self.halo_data.get_data( 'Y', self.snum, units='kpc' ),
                self.halo_data.get_data( 'Z', self.snum, units='kpc' ),
            ]).transpose()
            halo_pos = halo_pos_comov / ( 1. + self.redshift ) / self.hubble
            self._valid_halo_pos = halo_pos[self.valid_halo_inds]

        return self._valid_halo_pos

    ########################################################################

    @property
    def dist_to_all_valid_halos( self ):
        '''
        Returns:
            dist_to_all_valid_halos (np.ndarray) :
                Distance between self.particle_positions and all *.AHF_halos
                halos containing a galaxy (in pkpc).
        '''

        if not hasattr( self, '_dist_to_all_valid_halos' ):

            self._dist_to_all_valid_halos = \
                self.dist_to_all_valid_halos_fn( self.particle_positions )

        return self._dist_to_all_valid_halos

    def dist_to_all_valid_halos_fn( self, particle_positions ):
        '''
        Args:
            particle_positions (np.ndarray) :
                Location of particles to find the distance for.

        Returns:
            dist_to_all_valid_halos (np.ndarray) :
                Distance between the particle positions and all *.AHF_halos
                halos containing a galaxy (in pkpc).
        '''

        # Get the distances
        # Output is ordered such that dist[:,0] is the distance to the center of
        # halo 0 for each particle
        return scipy.spatial.distance.cdist(
            particle_positions, self.valid_halo_pos )

    ########################################################################

    @property
    def ahf_halos_length_scale_pkpc( self ):
        '''Actual values of the length scale used, e.g. Rvir.'''

        if not hasattr( self, '_ahf_halos_length_scale_pkpc' ):

            # Get the relevant length scale
            if self.length_scale == 'r_scale':
                # Get the scale radius
                r_vir = self.halo_data.get_data(
                    self.halo_length_scale,
                    self.snum,
                    units = 'kpc',
                )
                length_scale = r_vir / self.halo_data.get_data(
                    'cAnalytic',
                    self.snum,
                    )
            else:
                length_scale = self.halo_data.get_data(
                    self.length_scale,
                    self.snum,
                    units = 'kpc',
                )
            self._ahf_halos_length_scale_pkpc = \
                length_scale / ( 1. + self.redshift ) / self.hubble

        return self._ahf_halos_length_scale_pkpc

    ########################################################################
    # ID Finding Routines
    ########################################################################

    def find_ids( self ):
        '''Find relevant halo and galaxy IDs.

        Returns:
            galaxy_and_halo_ids (dict): Keys are...
            Parameters:
                halo_id (np.array of ints):
                    ID of the least-massive halo the particle is part of.
                host_halo_id (np.array of ints):
                    ID of the host halo the particle is part of.
                gal_id (np.array of ints):
                    ID of the smallest galaxy the particle is part of.
                host_gal_id (np.array of ints):
                    ID of the host galaxy the particle is part of.
                mt_halo_id (np.array of ints):
                    Merger tree ID of the least-massive halo the particle is
                    part of.
                mt_gal_id (np.array of ints):
                    Merger tree ID of the smallest galaxy the particle is
                    part of.
        '''

        # Dictionary to store the data in.
        galaxy_and_halo_ids = {}

        # Typically halo files aren't created for the first snapshot.
        # Account for this.
        if self.snum == 0:
            for id_type in self.ids_to_return:
                galaxy_and_halo_ids[id_type] = np.empty( self.n_particles )
                galaxy_and_halo_ids[id_type].fill( -2. )

            return galaxy_and_halo_ids

        # Halo ID works well, but not all the time.
        # In particular, it doesn't work when the length scale used is not
        # the virial radius. Break when this happens
        returning_a_halo_id = False
        for data_type in self.ids_to_return:
            if 'halo' in data_type:
                returning_a_halo_id = True
        diff_length_scale = (
            ( self.length_scale != 'Rvir' ) or
            ( self.mt_length_scale != 'Rvir' )
        )
        halo_id_broken = returning_a_halo_id and diff_length_scale
        assert not halo_id_broken, "Cannot currently return a halo ID when" + \
            " using a different length scale. This is because halo ID and" + \
            " gal ID use the same length scale, but halo ID assumes it's Rvir."

        # Actually get the data
        for id_type in self.ids_to_return:

            supplementary_data = id_type in self.ids_with_supplementary_data

            if id_type == 'halo_id':
                galaxy_and_halo_ids['halo_id'] = self.find_halo_id( supplementary_data=supplementary_data )
            elif id_type == 'host_halo_id':
                galaxy_and_halo_ids['host_halo_id'] = self.find_host_id()
            elif id_type == 'gal_id':
                galaxy_and_halo_ids['gal_id'] = \
                    self.find_halo_id( self.galaxy_cut, supplementary_data=supplementary_data  )
            elif id_type == 'host_gal_id':
                galaxy_and_halo_ids['host_gal_id'] = \
                    self.find_host_id( self.galaxy_cut, )
            elif id_type == 'mt_halo_id':
                galaxy_and_halo_ids['mt_halo_id'] = \
                    self.find_halo_id( type_of_halo_id='mt_halo_id', supplementary_data=supplementary_data  )
            elif id_type == 'mt_gal_id':
                galaxy_and_halo_ids['mt_gal_id'] = self.find_halo_id(
                    self.galaxy_cut, type_of_halo_id='mt_halo_id',  )
            elif id_type == 'd_gal':
                galaxy_and_halo_ids['d_gal'] = \
                    self.find_d_gal( supplementary_data=supplementary_data )
            elif id_type == 'd_other_gal':
                galaxy_and_halo_ids['d_other_gal'] = \
                    self.find_d_other_gal()
            elif id_type == 'd_other_gal_scaled':
                galaxy_and_halo_ids['d_other_gal_scaled'] = \
                    self.find_d_other_gal( scaled=True )
            # Custom scales
            else:
                galaxy_cut, length_scale = id_type.split( '_', 1 )
                galaxy_cut = float( galaxy_cut )

                galaxy_and_halo_ids[id_type] = \
                    self.find_halo_id( galaxy_cut, length_scale=length_scale )

        # Check for supplementary data and store it in a proper format
        galaxy_and_halo_ids_orig = copy.deepcopy( galaxy_and_halo_ids )
        for key, item in galaxy_and_halo_ids_orig.items():
            if isinstance( item, tuple ):
                arr, s_data = item
                galaxy_and_halo_ids[key] = arr
                for s_key, s_item in s_data.items():
                    stored_key = '{}_{}'.format( key, s_key )
                    galaxy_and_halo_ids[stored_key] = s_item

        return galaxy_and_halo_ids

    ########################################################################

    def find_d_gal( self, supplementary_data=False ):
        '''Find the distance to the center of the closest halo that contains a
        galaxy.

        Args:
            supplementary_data (bool) :
                If True, extract additional data about the closest galaxy

        Returns:
            d_gal (np.ndarray) : For particle i, d_gal[i] is the distance in
            pkpc to the center of the nearest galaxy.
        '''

        # Handle when no halos exist.
        if self.halo_data.get_n_halos( self.snum ) == 0:
             return -2. * np.ones( (self.n_particles,) )

        min_inds = np.argmin( self.dist_to_all_valid_halos, axis=1 )
        first_col_inds = np.arange( len( self.dist_to_all_valid_halos ) )
        d_gal = self.dist_to_all_valid_halos[first_col_inds,min_inds]

        if not supplementary_data:
            return d_gal

        # When including supplementary_data
        s_data = {}
        for key in self.supplementary_data_keys:

            # Handle when no halos exist.
            if self.halo_data.get_n_halos( self.snum ) == 0:
                 s_data[key] = -2. * np.ones( (self.n_particles,) )

            raw_data = np.array( self.halo_data.get_data( key, self.snum, ) )
            valid_data = raw_data[self.valid_halo_inds]
            s_data[key] = valid_data[min_inds]

        return d_gal, s_data

    ########################################################################

    def find_d_other_gal( self, scaled=False ):
        '''Find the distance to the center of the closest halo that contains a
        galaxy, other than the main galaxy.

        Args:
            scaled (bool) : If True, scale d_other_gal by length_scale

        Returns:
            d_other_gal (np.ndarray) :
                For particle i, d_other_gal[i] is the distance in pkpc to the
                center of the nearest galaxy, besides the main galaxy.
        '''

        # Handle when no halos exist.
        if self.halo_data.get_n_halos( self.snum ) == 0:
             return -2. * np.ones( (self.n_particles,) )

        # Handle when all the halos aren't massive enough
        if self.valid_halo_inds.size == 0:
            return -2. * np.ones( (self.n_particles,) )

        self.halo_data.data_reader.get_mtree_halos(
            self.mtree_halos_index, self.halo_file_tag )

        mtree_halo = self.halo_data.data_reader.mtree_halos[ self.main_mt_halo_id ]

        if self.snum < mtree_halo.index.min():
            # This mimics what would happen if ind_main_gal wasn't
            # in self.valid_halo_inds
            ind_main_gal_in_valid_inds = np.array( [] )
        else:
            # The indice for the main galaxy is the same as the AHF_halos
            # ID for it.
            ind_main_gal = mtree_halo['ID'][ self.snum ]

            valid_halo_ind_is_main_gal_ind = \
                self.valid_halo_inds == ind_main_gal
            ind_main_gal_in_valid_inds = \
                np.where( valid_halo_ind_is_main_gal_ind )[0]

        # If the there's no index because it's automatically okay
        if ind_main_gal_in_valid_inds.size == 0:
            dist_to_all_valid_other_gals = self.dist_to_all_valid_halos
            valid_halo_inds_sats = self.valid_halo_inds

        elif ind_main_gal_in_valid_inds.size == 1:

            dist_to_all_valid_other_gals = np.delete(
                self.dist_to_all_valid_halos,
                ind_main_gal_in_valid_inds[0],
                axis=1
            )
            valid_halo_inds_sats = \
                np.delete( self.valid_halo_inds, ind_main_gal_in_valid_inds[0] )

            # Handle when the only valid halo is the main halo
            if dist_to_all_valid_other_gals.size == 0:
                return -2. * np.ones( (self.n_particles,) )

        else:
            raise Exception(
                "ind_main_gal_in_valid_inds too big, is size {}".format(
                    valid_ind_main_gal.size )
            )

        if not scaled:
            return np.min( dist_to_all_valid_other_gals, axis=1 )

        inds_sat = np.argmin( dist_to_all_valid_other_gals, axis=1 )

        # Now scale
        length_scale_sats = self.ahf_halos_length_scale_pkpc[ valid_halo_inds_sats ]

        # Remove the unyt part of things so that we save properly without
        # having to invoke unyt's saving feature
        if isinstance( length_scale_sats, unyt.array.unyt_array ):
            length_scale_sats = length_scale_sats.to_value()

        dist_to_all_valid_other_gals_scaled = dist_to_all_valid_other_gals/length_scale_sats[np.newaxis,:]

        d_other_gal_scaled = dist_to_all_valid_other_gals_scaled[ np.arange( self.n_particles ), inds_sat ]

        return d_other_gal_scaled

    ########################################################################

    def find_host_id( self, radial_cut_fraction=1. ):
        '''Find the host halos our particles are inside of some radial cut of.
        This is the host ID at a given snapshot, and is not the same as the merger tree halo ID.

        Args:
            radial_cut_fraction (float): A particle is in a halo if it's in radial_cut_fraction*length_scale from the center.

        Returns:
            host_halo (np.array of ints): Shape ( n_particles, ).
                The ID of the least massive substructure the particle's part of.
                If it's -1, then the halo ID is the host ID.
                If it's -2, then that particle is not part of any halo, within radial_cut_fraction*length_scale .
        '''

        # Get the halo ID
        halo_id = self.find_halo_id( radial_cut_fraction )

        ahf_host_id =  self.halo_data.get_data( 'hostHalo', self.snum )

        # Handle the case where we have an empty ahf_halos, because there are no halos at that redshift.
        # In this case, the ID will be -2 throughout
        if ahf_host_id.size == 0:
            return halo_id

        # Setup the default array.
        host_id = np.full( halo_id.shape, -2 )

        # Get the host halo ID for valid halos
        valid_halo_id = halo_id != -2
        host_id[valid_halo_id] = ahf_host_id[ halo_id[valid_halo_id] ]

        # Fix the invalid values
        # (which come from not being associated with any halo)
        host_id_fixed = np.ma.fix_invalid( host_id, fill_value=-2 )

        return host_id_fixed.data.astype( int )

    ########################################################################

    def find_halo_id(
        self,
        radial_cut_fraction = 1.,
        type_of_halo_id = 'halo_id',
        length_scale = None,
        supplementary_data = False,
    ):
        '''Find the smallest halos our particles are inside of some radial cut
        of (we define this as the halo ID). In the case of using MT halo ID,
        we actually find the most massive halo our particles are inside some
        radial cut of.

        Args:
            radial_cut_fraction (float):
                A particle is in a halo if it's in radial_cut_fraction*length_scale from the center.

            type_of_halo_id (str):
                If 'halo_id' then this is the halo_id at a given snapshot.
                If 'mt_halo_id' then this is the halo_id according to the
                merger tree.

            length_scale (str):
                If using a custom length scale for the halo ID, what scale?

            supplementary_data (bool):
                If True, also return supplementary data according to the keys
                in self.supplementary_data_keys

        Returns:
            halo_id (np.array of ints): Shape ( n_particles, ).
                The ID of the least massive substructure the particle's part of.
                In the case of using the 'mt_halo_id', this is the ID of the
                most massive merger tree halo the particle's part of. If it's
                -2, then that particle is not part of any halo, within
                radial_cut_fraction*length_scale .
        '''

        # Choose parameters of the rest of the function based on what type of halo ID we're using
        if type_of_halo_id == 'halo_id':

            # Get the virial masses. It's okay to leave in comoving, since we're just finding the minimum
            m_vir = self.halo_data.get_data( 'Mvir', self.snum )

            # Handle the case where we have an empty ahf_halos, because there are no halos at that redshift.
            # In this case, the halo ID will be -2 throughout
            if m_vir.size == 0:
                halo_id = np.empty( self.n_particles )
                halo_id.fill( -2. )
                return halo_id

            # Functions that change.
            find_containing_halos_fn = self.find_containing_halos
            arg_extremum_fn = np.argmin
            extremum_fn = np.min

        elif type_of_halo_id == 'mt_halo_id':

            # Functions that change.
            find_containing_halos_fn = self.find_mt_containing_halos
            arg_extremum_fn = np.argmax
            extremum_fn = np.max

            # Get the virial masses. It's okay to leave in comoving, since we're just finding the maximum
            m_vir = self.halo_data.data_reader.get_mtree_halo_quantity(
                quantity = 'Mvir',
                indice = self.snum,
                index = self.mtree_halos_index,
                tag = self.halo_file_tag
            )

        else:
            raise Exception( "Unrecognized type_of_halo_id" )

        # Get the cut
        part_of_halo = find_containing_halos_fn(
            radial_cut_fraction = radial_cut_fraction,
            length_scale = length_scale,
        )

        # Mask the data
        tiled_m_vir = np.tile( m_vir, ( self.n_particles, 1 ) )
        tiled_m_vir_ma = np.ma.masked_array( tiled_m_vir, mask=np.invert( part_of_halo ), )

        # Take the extremum of the masked data
        if type_of_halo_id == 'halo_id':

            halo_inds = arg_extremum_fn( tiled_m_vir_ma, axis=1 )

            # Get the halo ID
            all_halo_ids = self.halo_data.get_data( 'ID', self.snum )
            halo_id = all_halo_ids[halo_inds]

            # When including supplementary data
            if supplementary_data:
                supplementary_data = {}
                for data_key in self.supplementary_data_keys:
                    full_data = self.halo_data.get_data( data_key, self.snum )

                    # Make sure it's a numpy array (instead of a unyt array)
                    full_data = np.array( full_data )

                    supplementary_data[data_key] = full_data[halo_inds]

        elif type_of_halo_id == 'mt_halo_id':

            if supplementary_data:
                raise Exception( "Supplementary data not yet compatible with MT halo IDs." )

            halo_inds = arg_extremum_fn( tiled_m_vir_ma, axis=1 )
            halo_ids = np.array( sorted( self.halo_data.data_reader.mtree_halos.keys() ) )
            halo_id = halo_ids[halo_inds]

        # Account for the fact that the argmin defaults to 0 when there's nothing there
        mask = extremum_fn( tiled_m_vir_ma, axis=1 ).mask
        halo_id = np.ma.filled( np.ma.masked_array(halo_id, mask=mask), fill_value=-2 )

        # Do the same for the supplementary data
        if type_of_halo_id in self.ids_with_supplementary_data:
            supplementary_data[data_key] = np.ma.filled(
                np.ma.masked_array( supplementary_data[data_key], mask=mask ),
                fill_value = -2
            )

        # Return
        if not supplementary_data:
            return halo_id
        else:
            return halo_id, supplementary_data

    ########################################################################

    def find_containing_halos( self, radial_cut_fraction=1., length_scale=None, ):
        '''Find which halos our particles are inside of some radial cut of.

        Args:
            radial_cut_fraction (float): A particle is in a halo if it's in radial_cut_fraction*length_scale from the center.

        Returns:
            part_of_halo (np.array of bools): Shape (n_particles, n_halos).
                If index [i, j] is True, then particle i is inside radial_cut_fraction*length_scale of the jth halo.
        '''

        if length_scale is None:
            length_scale = self.length_scale

        # Get the relevant length scale
        if length_scale == 'r_scale':
            # Get the scale radius
            r_vir = self.halo_data.get_data(
                self.halo_length_scale,
                self.snum,
                units = 'kpc',
            )
            length_scale_arr = r_vir / self.halo_data.get_data(
                'cAnalytic',
                self.snum,
                )
        else:
            length_scale_arr = self.halo_data.get_data(
                length_scale,
                self.snum,
                units = 'kpc',
            )

        halos_length_scale_pkpc = \
            length_scale_arr / ( 1. + self.redshift ) / self.hubble

        # Get the radial cut
        radial_cut = radial_cut_fraction*halos_length_scale_pkpc[self.valid_halo_inds]

        # Find the halos that our particles are part of (provided they passed the minimum cut)
        part_of_halo_success = self.dist_to_all_valid_halos < radial_cut[np.newaxis,:]

        # Get the full array out
        part_of_halo = np.zeros( (self.n_particles, halos_length_scale_pkpc.size) ).astype( bool )
        part_of_halo[:,self.valid_halo_inds] = part_of_halo_success

        return part_of_halo

    ########################################################################

    def find_mt_containing_halos(
        self,
        radial_cut_fraction = 1.,
        length_scale = None,
    ):
        '''Find which MergerTrace halos our particles are inside of some
        radial cut of.

        Args:
            radial_cut_fraction (float):
                A particle is in a halo if it's in
                radial_cut_fraction*length_scale from the center.

        Returns:
            part_of_halo (np.array of bools): Shape (n_particles, n_halos).
                If index [i, j] is True, then particle i is inside
                radial_cut_fraction*length_scale of the jth halo, defined
                via the MergerTrace ID.
        '''

        assert length_scale is None, \
            "find_mt_containing_halos has not yet been updated to accept" + \
            " a length_scale as an argument like find_containing_halos."

        # Load up the merger tree data
        self.halo_data.data_reader.get_mtree_halos(
            self.mtree_halos_index,
            self.halo_file_tag
        )

        part_of_halo = []
        for halo_id in sorted( self.halo_data.data_reader.mtree_halos.keys() ):
            mtree_halo = self.halo_data.data_reader.mtree_halos[ halo_id ]

            # Only try to get the data if we're in the range we actually have
            # the halos for.
            above_minimum_snap = self.snum >= mtree_halo.index.min()

            # Only try to get the data if we have the minimum stellar mass
            if above_minimum_snap:
                halo_value = (
                    mtree_halo[ self.minimum_criteria ][ self.snum ]
                    / self.min_conversion_factor
                )
                has_minimum_value = halo_value >= self.minimum_value
            else:
                # If it's not at the point where it can be traced, it
                # definitely doesn't have the minimum stellar mass.
                has_minimum_value = False

            # Usual case
            if has_minimum_value:

                # Get the halo position
                halo_pos_comov = np.array([
                    mtree_halo['Xc'][ self.snum ],
                    mtree_halo['Yc'][ self.snum ],
                    mtree_halo['Zc'][ self.snum ],
                ])
                halo_pos = halo_pos_comov/( 1. + self.redshift )/self.hubble

                # Make halo_pos 2D for compatibility with cdist
                halo_pos = halo_pos[np.newaxis]

                # Get the distances
                dist = scipy.spatial.distance.cdist( self.particle_positions, halo_pos )

                # Get the relevant length scale
                if self.mt_length_scale == 'r_scale':
                    # Get the scale radius
                    r_vir = mtree_halo[self.halo_length_scale][ self.snum ]
                    length_scale = r_vir/mtree_halo['cAnalytic'][ self.snum ]
                else:
                    length_scale = mtree_halo[self.mt_length_scale][self.snum]
                length_scale_pkpc = length_scale/( 1. + self.redshift )/self.hubble

                # Get the radial distance
                radial_cut = radial_cut_fraction*length_scale_pkpc

                # Find if our particles are part of this halo
                part_of_this_halo = dist < radial_cut

            # Case where there isn't a main halo at that redshift.
            else:
                part_of_this_halo = np.zeros( (self.n_particles, 1) ).astype( bool )

            part_of_halo.append( part_of_this_halo )

        # Format part_of_halo correctly
        part_of_halo = np.array( part_of_halo ).transpose()[0]

        return part_of_halo

    ########################################################################
    # Routines for Calculating Properties of Galaxies
    ########################################################################

    @property
    def mass_inside_galaxy_cut( self ):
        '''
        Returns:
            mass_inside_galaxy_cut (np.ndarray) :
                Mass inside the galaxy_cut*length_scale for *.AHF_halos halos that containing a galaxy.
        '''

        if not hasattr( self, '_mass_inside_galaxy_cut' ):

            self._mass_inside_galaxy_cut = self.summed_quantity_inside_galaxy_valid_halos( self.particle_masses, 0. )

        return self._mass_inside_galaxy_cut

    ########################################################################

    @property
    def mass_inside_all_halos( self ):
        '''
        Returns:
            mass_inside_galaxy_cut (np.ndarray) :
                Mass inside the galaxy_cut*length_scale for all *.AHF_halos halos.
        '''

        if not hasattr( self, '_mass_inside_all_halos' ):
            self._mass_inside_all_halos = np.empty( self.ahf_halos_length_scale_pkpc.shape )
            self._mass_inside_all_halos.fill( np.nan )

            # Case where no halos meet the minimum criteria yet.
            if self.valid_halo_inds.size == 0:
                return self._mass_inside_all_halos

            self._mass_inside_all_halos[self.valid_halo_inds] = self.mass_inside_galaxy_cut

        return self._mass_inside_all_halos

    ########################################################################

    @property
    def cumulative_mass_valid_halos( self ):
        '''Get the cumulative mass, going outwards from the center of each valid *AHF_halos halo.

        Returns:
            cumulative_mass (np.ndarray) : Shape (n_particles, n_halos)
                Index (i,j) is the cumulative mass in halo j at the location of particle i.
        '''

        if not hasattr( self, '_cumulative_mass_valid_halos' ):
            # Sort the mass according to distance
            sorted_inds = np.argsort( self.dist_to_all_valid_halos, axis=0 )
            sorted_mass = self.particle_masses[sorted_inds]

            sorted_cumulative_mass = np.cumsum( sorted_mass, axis=0 )

            # Undo the sorting, now that we have the cumulative mass
            undo_inds = np.argsort( sorted_inds, axis=0 )
            cumulative_mass = sorted_cumulative_mass[undo_inds, np.arange(undo_inds.shape[1])]

            self._cumulative_mass_valid_halos = cumulative_mass

        return self._cumulative_mass_valid_halos

    ########################################################################

    def get_mass_radius( self, mass_fraction ):
        '''Get the radius at which mass_fraction*mass_inside_galaxy_cut is exceeded.

        Args:
            mass_fraction (float) : Mass fraction to consider.

        Returns:
            mass_radius (np.ndarray) :
                The ith index is the distance from the center of halo i to the center of the last particle before
                mass_fraction*mass_inside_galaxy_cut is exceeded.
        '''

        mass_allowed_in_halo = mass_fraction*self.mass_inside_galaxy_cut[np.newaxis,:]
        greater_than_mass_fraction = self.cumulative_mass_valid_halos > mass_allowed_in_halo

        dist_ma = np.ma.masked_array( self.dist_to_all_valid_halos, mask=greater_than_mass_fraction )

        # Fill values in where there's insufficient stellar mass inside that halo, or where we can't resolve the radii.
        # This should be unnecessary if requiring a minimum amount of stellar mass, but it doesn't hurt.
        mass_radius_valid_inds = dist_ma.max( axis=0 )
        mass_radius_valid_inds.fill_value = np.nan
        mass_radius_valid_inds = mass_radius_valid_inds.filled()

        # Now get the mass radius for the full thing.
        mass_radius = np.empty( self.ahf_halos_length_scale_pkpc.shape )
        mass_radius.fill( np.nan )
        mass_radius[self.valid_halo_inds] = mass_radius_valid_inds

        # Convert back to comoving kpc/h
        mass_radius *= ( 1. + self.redshift )*self.hubble

        return mass_radius

    ########################################################################

    def summed_quantity_inside_galaxy_valid_halos( self, particle_quantities, fill_value=0. ):
        '''Get sum( particles_quantities ) for each galaxy (i.e. for particles fulfilling the galaxy cut requirements),
        for halos that are "valid" (i.e. usually meeting some minimum criteria).

        Args:
            particle_quantities (np.ndarray) :
                Quantities to sum.

            fill_value (float) :
                When there are are no particles in a galaxy, what value should be used?

        Returns:
            summed_quantity_inside_galaxy_valid (np.ndarray)
        '''

        def work_fn( particle_quantities_passed, particle_positions_passed=None ):
            '''Private function that actually does what we want.
            '''

            # Case where there are no halos formed yet.
            if self.ahf_halos_length_scale_pkpc.size == 0:
                return np.array( [] )

            # Case where no halos meet the minimum criteria yet.
            if self.valid_halo_inds.size == 0:
                return np.array( [] )

            # If we don't have any particles, just return a filled array.
            if particle_quantities_passed.size == 0:
                return np.array( [ fill_value, ]*self.valid_halo_inds.size )

            if self.low_memory_mode:
                dist_to_all_valid_halos_used = self.dist_to_all_valid_halos_fn( particle_positions_passed )

            else:

                # Make sure we're not trying to use any weird particle positions, i.e. we're using the defaults
                assert particle_positions_passed is None

                dist_to_all_valid_halos_used = self.dist_to_all_valid_halos

            # Make a radial cut.
            valid_radial_cut_pkpc = self.galaxy_cut*self.ahf_halos_length_scale_pkpc[self.valid_halo_inds]
            outside_radial_cut = dist_to_all_valid_halos_used > valid_radial_cut_pkpc[np.newaxis,:]

            # Tile the quantity for proper masking
            quantity_tiled = np.tile( particle_quantities_passed, ( self.valid_halo_inds.size, 1 ) ).transpose()

            quantity_ma = np.ma.masked_array( quantity_tiled, mask=outside_radial_cut )

            # Do the actual sum.
            summed_quantity_inside_galaxy_valid = quantity_ma.sum( axis=0 )

            # Fill in any values where there are no particles in the galaxy.
            summed_quantity_inside_galaxy_valid.fill_value = fill_value
            summed_quantity_inside_galaxy_valid = summed_quantity_inside_galaxy_valid.filled()

            # Make sure that we don't count halos with NaN length scales as containing all particles.
            has_bad_value = np.ma.fix_invalid( valid_radial_cut_pkpc ).mask
            summed_quantity_inside_galaxy_valid = np.where( has_bad_value, np.nan, summed_quantity_inside_galaxy_valid, )

            return summed_quantity_inside_galaxy_valid

        if self.low_memory_mode:

            # Low memory node will not necessarily work for non-zero fill_values, so assert that fill_value == 0
            np.testing.assert_allclose( 0., fill_value, atol=1e-7 )

            # Split the particle quantities into smaller lists
            particle_quantities_chunked = utilities.chunk_list( particle_quantities, self.memory_mode_divisions )
            particle_positions_chunked = utilities.chunk_list( self.particle_positions, self.memory_mode_divisions )

            # Get the result
            summed_quantities_split = [ work_fn( particle_quantities_chunk, particle_positions_chunk ) for \
                                                                    particle_quantities_chunk, particle_positions_chunk in \
                                                                    zip( particle_quantities_chunked, particle_positions_chunked ) ]

            # Sum and return results
            return np.array( summed_quantities_split ).sum( axis=0 )

        else:
            return work_fn( particle_quantities )

    ########################################################################

    def summed_quantity_inside_galaxy( self, particle_quantities, fill_value ):
        '''Get sum( particles_quantities ) for each galaxy (i.e. for particles fulfilling the galaxy cut requirements).
        Args:
            particle_quantities (np.ndarray) :
                Quantities to sum.

            fill_value (float) :
                When there are are no particles in a galaxy, what value should be used?

        Returns:
            summed_quantity_inside_galaxy (np.ndarray)
        '''

        summed_quantity_inside_galaxy = np.empty( self.ahf_halos_length_scale_pkpc.shape )
        summed_quantity_inside_galaxy.fill( np.nan )

        # Case where no halos meet the minimum criteria yet.
        if self.valid_halo_inds.size == 0:
            return summed_quantity_inside_galaxy

        summed_quantity_inside_galaxy[self.valid_halo_inds] = self.summed_quantity_inside_galaxy_valid_halos(
            particle_quantities,
            fill_value,
        )

        return summed_quantity_inside_galaxy

    ########################################################################

    def weighted_summed_quantity_inside_galaxy( self, particle_quantities, particle_weights, fill_value ):
        '''Get sum( particles_quantities ) for each galaxy (i.e. for particles fulfilling the galaxy cut requirements),
        weighted by particle_weights.

        Args:
            particle_quantities (np.ndarray) :
                Quantities to tile.

            particle_weights (np.ndarray) :
                Weights to apply.

            fill_value (float) :
                When there are are no particles in a galaxy, what value should be used?

        Returns:
            summed_quantity_inside_galaxy (np.ndarray)
        '''

        weighted_and_summed = self.summed_quantity_inside_galaxy( particle_quantities*particle_weights, fill_value )
        sum_of_weights = self.summed_quantity_inside_galaxy( particle_weights, fill_value )

        return weighted_and_summed/sum_of_weights






















