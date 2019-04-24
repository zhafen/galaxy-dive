#!/usr/bin/env python
'''Tools for reading AHF output files.

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import glob
import numba
import numpy as np
import os
import pandas as pd
import unyt

import galaxy_dive.read_data.metafile as read_metafile

########################################################################

default = object()

########################################################################
########################################################################

class AHFReader( object ):
    '''Read AHF data.
    Note! All positions are in comoving coordinates, and everything has 1/h's sprinkled throughout.
    '''

    def __init__( self, data_dir ):
        '''Initializes.

        Args:
            data_dir (str): Simulation directory to load the AHF data from.
        '''

        self.data_dir = data_dir

        # Common global names, and what columns those correspond to in the
        # data
        self.general_use_data_names = {
            'X' : 'Xc',
            'Y' : 'Yc',
            'Z' : 'Zc',
            'Vx' : 'VXc',
            'Vy' : 'VYc',
            'Vz' : 'VZc',
        }

        # Data units
        self.units = {
            'Mvir' : unyt.Msun,
            'Xc' : unyt.kpc,
            'Yc' : unyt.kpc,
            'Zc' : unyt.kpc,
            'VXc' : unyt.km / unyt.s,
            'VYc' : unyt.km / unyt.s,
            'VZc' : unyt.km / unyt.s,
            'Rvir' : unyt.kpc,
            'Rstar0.5' : unyt.kpc,
            'Rstar0.9' : unyt.kpc,
        }

    ########################################################################
    # Load Data
    ########################################################################

    def get_mtree_halos( self, index=None, tag=None, adjust_default_labels=default, ):
        '''Get halo files (e.g. halo_00000.dat) in a dictionary of pandas DataFrames.

        Args:
            index (str or int) :
                What type of index to use. Defaults to None, which raises an exception. You *must* choose
                an index, to avoid easy mistakes. Options are...
                'snum' : Indexes by snapshot number, starting at 600 and counting down. Only viable with snapshot steps of 1!!
                                  Identical to passing the integer 600. (See below.)
                'range' : Index by an increasing range.
                int : If an integer, then this integer should be the final snapshot number for the simulation.
                            In this case, indexes by snapshot number, starting at the final snapshot number and counting down.
                            Only viable with snapshot steps of 1!!

            tag (str) :
                Additional identifying tag for the files, e.g. 'smooth', means this function will look for
                'halo_00000_smooth.dat', etc.

            adjust_default_label (bool) :
                Whether or not to adjust the column headings, etc. Defaults to True if tag is None.

        Modifies:
            self.mtree_halos (dict of pd.DataFrames) :
                DataFrames containing the requested data. The key for a given dataframe
                is that dataframe's Merger Tree Halo ID

            self.mtree_halo_filepaths (dict of strs) :
                What files the data was loaded from.

            self.index (str) :
                The users value for the index.

            self.tag (str) :
                What tag was used for the files.
        '''

        if adjust_default_labels is default:
            if tag is None:
                adjust_default_labels = True
            else:
                adjust_default_labels = False

        def get_halo_filepaths( unexpanded_filename ):
            '''Function for getting a list of filepaths'''
            filepath_unexpanded = os.path.join( self.data_dir, unexpanded_filename )
            halo_filepaths = glob.glob( filepath_unexpanded )
            return set( halo_filepaths )

        # Get the filename to search for
        if tag is not None:
            ahf_filename = 'halo_*_{}.dat'.format( tag )
            halo_filepaths = get_halo_filepaths( ahf_filename )

        else:
            ahf_filename = 'halo_*.dat'
            halo_filepaths = get_halo_filepaths( ahf_filename )

            # Find all files that are modified.
            ahf_modified_filename = 'halo_*_*.dat'
            halo_modified_filepaths = get_halo_filepaths( ahf_modified_filename )

            # Remove all the modified filepaths from the search list.
            halo_filepaths -= halo_modified_filepaths

        # Raise an exception if there are no files to load
        if len( halo_filepaths ) == 0:
            raise NameError( 'No files to load in {}'.format( self.data_dir ) )

        # Set up the data storage
        self.mtree_halos = {}
        self.mtree_halo_filepaths = {}

        # Loop over each file and load it
        for halo_filepath in halo_filepaths:

            # Load the data
            mtree_halo = pd.read_csv( halo_filepath, sep='\t', )

            # Extra tweaking to read the default AHF file format
            if adjust_default_labels:
                # Delete a column that shows up as a result of formatting
                del mtree_halo[ 'Unnamed: 93' ]

                # Remove the annoying parenthesis at the end of each label.
                mtree_halo.columns = [ label.split( '(' )[0] for label in list( mtree_halo ) ]

                # Remove the pound sign in front of the first column's name
                mtree_halo = mtree_halo.rename( columns = {'#redshift':'redshift', ' ID':'ID'} )

            # Get a good key
            base_filename = os.path.basename( halo_filepath )
            halo_num_str = base_filename[5:]
            if tag is not None:
                halo_num_str = halo_num_str.split( '_' )[0]
            halo_num = int( halo_num_str.split( '.' )[0] )

            if index == 'range':
                pass
            elif (index == 'snum') or isinstance( index, int ):
                if index == 'snum':
                    final_snapshot_number = 600
                else:
                    final_snapshot_number = index
                # Set the index, assuming we have steps of one snapshot
                n_rows = mtree_halo.shape[0]
                mtree_halo['snum'] = range( final_snapshot_number, final_snapshot_number - n_rows, -1)
                mtree_halo = mtree_halo.set_index( 'snum', )
            elif 'snum' in mtree_halo.columns:
                mtree_halo = mtree_halo.set_index( 'snum', )
            else:
                raise Exception( "index type not selected" )

            # Store the data
            self.mtree_halos[ halo_num ] = mtree_halo
            self.mtree_halo_filepaths[ halo_num ] = halo_filepath

        # Save the index and tag as an attribute
        self.index = index
        self.tag = tag

    ########################################################################

    def get_halos( self, snum, force_reload=False ):
        '''Get *.AHF_halos file for a particular snapshot.

        Args:
            snum (int): Snapshot number to load.

            force_reload (bool): Force reloading, even if there's already an halos file loaded.

        Modifies:
            self.halos (pd.DataFrame): Dataframe containing the requested data.
        '''

        if hasattr( self, 'halos' ):
            if (self.halos_snum == snum) and not force_reload:
                return

        # Load the data
        self.halos_path = self.get_filepath( snum, 'AHF_halos' )
        self.halos = pd.read_csv( self.halos_path, sep='\t', index_col=0 )

        # Delete a column that shows up as a result of formatting
        del self.halos[ 'Unnamed: 92' ]

        # Remove the annoying parenthesis at the end of each label.
        self.halos.columns = [ label.split('(')[0] for label in list( self.halos ) ]

        # Rename the index to a more suitable name, without the '#' and the (1)
        self.halos.index.names = ['ID']

        # Save the snapshot number of the ahf halos file.
        self.halos_snum = snum

        # Note that we haven't added the additional halos data yet.
        self.halos_added = False

    ########################################################################

    def get_halos_add( self, snum, force_reload=False ):
        '''Get *.AHF_halos_add file for a particular snapshot.

        Args:
            snum (int): Snapshot number to load.

            force_reload (bool): Force reloading, even if there's already an halos file loaded.

        Modifies:
            self.halos_add (pd.DataFrame): Dataframe containing the requested data.
        '''

        if self.halos_added:
            if (self.halos_snum == snum) and not force_reload:
                return

        # Load the data
        self.halos_add_path = self.get_filepath( snum, 'AHF_halos_add' )
        halos_add = pd.read_csv( self.halos_add_path, sep='\t', index_col=0 )

        if not hasattr( self, 'halos' ) :
            self.get_halos( snum )
        if self.halos_snum != snum:
            self.get_halos( snum )

        self.halos = pd.concat( [ self.halos, halos_add ], axis=1 )

        self.halos_added = True

    ########################################################################

    def get_mtree_idx( self, snum ):
        '''Get *.AHF_mtree_idx file for a particular snapshot.

        Args:
            snum (int): Snapshot number to load.

        Modifies:
            self.ahf_mtree_idx (pd.DataFrame): Dataframe containing the requested data.
        '''

        # If the data's already loaded, don't load it again.
        if hasattr( self, 'ahf_mtree_idx' ):
            return self.ahf_mtree_idx

        # Load the data
        ahf_mtree_idx_path = self.get_filepath( snum, 'AHF_mtree_idx' )
        self.ahf_mtree_idx = pd.read_csv( ahf_mtree_idx_path, delim_whitespace=True, names=['HaloID(1)', 'HaloID(2)'], skiprows=1  )

    ########################################################################

    def get_profiles( self, snum ):
        '''Get *.AHF_mtree_idx file for a particular snapshot.

        Args:
            snum (int): Snapshot number to load.

        Modifies:
            self.ahf_mtree_idx (pd.DataFrame): Dataframe containing the requested data.
        '''

        # If the data's already loaded, don't load it again.
        if hasattr( self, 'profiles' ):
            return self.profiles

        # Load the data
        profiles_path = self.get_filepath( snum, 'AHF_profiles' )
        self.profiles = pd.read_csv(
            profiles_path,
            sep = '\t', 
        )

        # Delete a column that shows up as a result of formatting
        del self.profiles['Unnamed: 27']

        # Remove the annoying parenthesis at the end of each label.
        self.profiles.columns = [ label.split('(')[0] for label in list( self.profiles ) ]

        # Remove the # at the beginning of the first column
        self.profiles.rename( columns={'#r': 'r'}, inplace=True )

        # Get the mapping for where the halo IDs are in the profile data
        @numba.jit( 'i8[:](f8[:])', nopython=True )
        def profile_mapping( r ):
            
            ind_mapping = [ 0, ]
            for i, r_i in enumerate( r ):

                # Skip first loop
                if i==0:
                    continue

                # Record when the values start over
                if r_i <  r[i-1]:
                    ind_mapping.append( i )

                # Last index
                if i == r.size - 1:
                    ind_mapping.append( i+1 )

            return np.array( ind_mapping )

        # Run the function
        self.profile_id_mapping = profile_mapping(
            np.abs( self.profiles['r'].values )
        )

    ########################################################################
    # Utilities
    ########################################################################

    def get_filepath( self, snum, ahf_file_type ):
        '''Get the filepath for a specified type of AHF file.

        Args:
            snum (int): Snapshot number to load.
            ahf_file_type (str): Can be AHF_halos or AHF_mtree_idx.

        Returns:
            ahf_filepath (str): The filepath to the specified file.
        '''

        # Load the data
        ahf_filename = 'snap{:03d}Rpep..z*.*.{}'.format( snum, ahf_file_type )
        ahf_filepath_unexpanded = os.path.join( self.data_dir, ahf_filename )
        possible_filepaths = glob.glob( ahf_filepath_unexpanded )
        if len( possible_filepaths ) == 0:
            raise NameError( 'No files to load for snum {} in {}'.format( snum, self.data_dir ) )
        elif len( possible_filepaths ) > 1:
            raise Exception( 'Multiple possible *.{} files to load'.format( ahf_file_type ) )
        ahf_filepath = possible_filepaths[0]

        return ahf_filepath

    ########################################################################

    def check_files_exist( self, snum_start, snum_end, snum_step, file_str='AHF_halos' ):
        '''Check that AHF was successfully run for all files in the given range.

        Args:
            snum_start (int) : Starting snapshot.
            snum_end (int) : Ending snapshot.
            snum_step (int) : Step between snapshots.

        Returns:
            files_exist (bool) : If True, all files in the given range exist.
        '''

        snums = range( snum_start, snum_end + 1, snum_step )

        missing_snums = []
        for snum in snums:

            file_to_check = 'snap{:03d}Rpep*{}'.format( snum, file_str )
            filepath_to_check = os.path.join( self.data_dir, file_to_check )
            files = glob.glob( filepath_to_check )

            missing_ahf_files_at_this_snapshot = len( files ) == 0
            if missing_ahf_files_at_this_snapshot:
                missing_snums.append( snum )

        if len( missing_snums ) > 0:

            print( 'Missing snums:' )
            for missing_snum in missing_snums:
                print( '{},'.format( missing_snum ) )

            return False

        return True

    ########################################################################
    # Get Specific Data Arrays
    ########################################################################

    def get_mtree_halo_quantity( self, quantity, indice, index=None, tag=None ):
        '''Get a desired quantity for all merger tree halos at a particular snapshot.

        Args:
            quantity (str): mtree_halo key to load in the dataset
            indice (int): Indice of the quantity to load, as indicated by the index.
            index (str or int) : What type of index to use. Defaults to None, which raises an exception.
                You *must* choose an index, to avoid easy mistakes. See get_mtree_halos() for a full description.
            tag (str) : Additional identifying tag for the files, e.g. 'smooth', means this function will look for
                'halo_00000_smooth.dat', etc.

        Returns:
            mtree_halo_quantity (np.array): The ith index is the requested quantity for ith MT halo.
        '''

        # Load the data if it's not already loaded.
        if not hasattr( self, 'mtree_halos' ):
            self.get_mtree_halos( index=index, tag=tag )
        else:
            assert index == self.index
            assert tag == self.tag

        mtree_halo_quantity = []
        for halo_id in sorted( self.mtree_halos.keys() ):

            try:
                mtree_halo_quantity.append( self.mtree_halos[ halo_id ][ quantity ][ indice ] )

            # When we're past the point of galaxies being identified (i.e. high redshift, galaxies aren't formed yet),
            # set the values by hand.
            except KeyError:

                if quantity == 'Mvir':
                    mtree_halo_quantity.append( 0. )

                else:
                    raise Exception( "Value of {} not specified before galaxies form".format( quantity ) )

        return np.array( mtree_halo_quantity )

    ########################################################################

    def get_pos_or_vel( self, pos_or_vel, halo_id, inds, type_of_halo_id='merger_tree' ):
        '''Get the position or velocity of a mt halo (three dimensional).

        Args:
            pos_or_vel (str): Get position ('pos') or velocity ('vel').
            halo_id (int): Merger tree halo ID for the position or velocity you want.
            inds (int or np.array of ints): Indices you want the position or velocity for.
              If type_of_halo_id == 'merger_tree', uses same index as mtree_halos.
              Elif type_of_halo_id == 'halos', can only be a single int,
              which should be the snapshot number.
            type_of_halo_id (str): 'merger_tree' if the halo id is a merger tree halo id.
                                                          'halos' if the halo id is a *.AHF_halos halo id.

        Returns:
            p_or_v ( [len(inds), 3] np.array ): Position or velocity for the specified inds.
        '''

        # Choose the indices we'll access the data through
        if pos_or_vel == 'pos':
            keys = [ 'Xc', 'Yc', 'Zc' ]
        elif pos_or_vel == 'vel':
            keys = [ 'VXc', 'VYc', 'VZc' ]
        else:
            raise Exception( 'Unrecognized pos_or_vel, {}'.format( pos_or_vel ) )

        # Get the ahf_halo data, if requested.
        if type_of_halo_id == 'halos':
            self.get_halos( inds )

        # Get the data.
        p_or_v = []
        for key in keys:

            # Get the part
            if type_of_halo_id == 'merger_tree':
                p_or_v_part = self.mtree_halos[ halo_id ][ key ][ inds ]
            elif type_of_halo_id == 'halos':
                p_or_v_part = self.halos[ key ][ halo_id ]
            else:
                raise Exception( 'Unrecognized type_of_halo_id, {}'.format( type_of_halo_id ) )

            p_or_v.append( p_or_v_part )

        # Finish formatting.
        p_or_v = np.array( p_or_v ).transpose()

        return p_or_v

