#!/usr/bin/env python
'''Wrapper for hdf5 objects opened with h5py. Serves various needs not inherent in h5py.'''

import copy
import h5py
import numpy as np
import os
import pandas as pd
import six
import shutil

import galaxy_dive.utils.utilities as utilities

########################################################################
__author__ = 'Zachary Hafen'

__maintainer__ = 'Zachary Hafen'
__email__ = 'zachary.h.hafen@gmail.com'
__status__ = 'Beta'

########################################################################

default = object()

########################################################################

class HDF5Wrapper(object):

    def __init__(self, filename):

        self.filename = filename

    ########################################################################

    def save_data(self, data, index_key=None):
        '''Save data at self.filename.

        Args:
            data (dict): The data to be saved.
        '''

        self.f = h5py.File(self.filename, 'a')

        for key in data:

            # If it's a group
            if type(data[key]) == dict:
                self.f.create_group(key)
                for key2 in data[key].keys():
                    if key2 != 'attrs':
                        full_data_key = '{}/{}'.format(key, key2)
                        self.f.create_dataset(full_data_key, data=data[key][key2])

                    # Include any attributes
                    else:
                        for attr_key in data[key]['attrs'].keys():
                            self.f[key].attrs[attr_key] = data[key]['attrs'][attr_key]

            else:
                try:
                    self.f.create_dataset(key, data=data[key])

                # Handle the case where there's an empty dataset
                except TypeError:
                    if data[key][0] == {}:
                        continue
                    else:

                        # Workaround for hdf5 and np.arrays
                        arr = np.array(
                            data[key],
                            dtype=h5py.special_dtype( vlen=six.text_type ),
                        )
                        self.f.create_dataset( key, data=arr )

        # Catalog which key 'Indexes' the dataset, if any.
        if index_key is not None:
            self.f.attrs['index_key'] = index_key

        self.f.close()

    ########################################################################

    def insert_data(self, new_data, index_key=None, insert_columns=False):
        '''Insert data. Does this by making whole new copy, appending to it,
        sorting that by the index, and saving the sorted version.

        Args:
            new_data (dict):
                The new data to be sorted.

            index_key (str):
                What key to sort the data by. Defaults to None, which will use
                the key inherent in the data set, and fail otherwise.

            insert_columns (bool):
                If True, the data in new_data should be included as new columns.
        '''

        try:
            self.f = h5py.File(self.filename, 'r')
        except IOError:
            self.save_data(new_data, index_key)
            return 0

        # Make sure we use a consistent index.
        if index_key is not None:
            if 'index_key' in self.f.attrs.keys():
                assert index_key == self.f.attrs['index_key']
        else:
            if 'index_key' in self.f.attrs.keys():
                index_key = self.f.attrs['index_key']
            else:
                raise Exception('No index_key specified at any point.')

        # Handle an empty data set
        if len(self.f.keys()) == 0:
            self.f.close()
            self.save_data(new_data, index_key)
            return 0

        # Load the data into lists
        data = {}
        for key in self.f.keys():
            data[key] = self.f[key][...].tolist()

        # Append the new data
        if not insert_columns:
            # Case where we have rows.
            try:
                for key in new_data:
                    data[key].append(new_data[key])

            # Deal with single floats
            except AttributeError:

                for key in data:
                    data[key] = [ data[key], ]

                for key in new_data:
                    data[key].append(new_data[key])

        # Columns case (using pandas)
        else:
            # Set up DataFrames
            df = pd.DataFrame( data )
            df_new = pd.DataFrame( new_data )

            # Merge dataframes and get result out
            merged_df = pd.merge( df, df_new, on=index_key )
            data = merged_df.to_dict( 'list' )

        # Sort the data
        data = self.sort_dictionary(data, index_key)

        # Only include unique values, according to the index
        unique_index_data, unique_inds = np.unique(data[index_key], True)
        for key in data:
            data[key] = np.array(data[key])[unique_inds]

        # Delete the old data.
        self.f.close()
        self.clear_hdf5_file()

        # Store the data again
        self.save_data(data, index_key=index_key)

        return 0

    ########################################################################

    def sort_dictionary(self, data, index_key):
        '''Sort a dictionary.

        Args:
            data (dict): Dictionary to sort.
            index_key (str): Key to sort by.
        '''

        # Sort the data
        sorted_inds = np.argsort(data[index_key])
        for key in data:
            data[key] = np.array(data[key])[sorted_inds]

        return data

    ########################################################################

    def subsample_hdf5_file(self, subsamples, particle_data=False ):
        '''Create a dictionary containing subsamples of the data.

        Args:
            subsamples (int): The number of subsamples per axis, for a total of subsamples^n_dims subsamples.
            particle_data (bool or int): Whether or not the data contains particle-like data.
                If not False, then it's the number of files the snapshot is spread across
        '''

        def subsample_dataset(dataset):

            subsampled_data = {}

            if len( dataset ) == 0:
                return subsampled_data
            else:

                # Get the first value for easy reference
                first_value = list( dataset.values() )[0]

                # Get the shape of the subsample
                n_dims = first_value[...].ndim
                subsample_shape = tuple([subsamples for i in range(n_dims)])
                n_subsamples = np.product(subsample_shape)


                # Subsample indices
                if particle_data:
                    n_particles = max( first_value.shape )
                    subsample_inds = np.random.randint(0, n_particles, subsamples)
                else:
                    subsample_inds = np.random.randint(0, first_value.size, n_subsamples)

                # Do the subsampling
                for key in dataset:

                    # Subsample when using particle data
                    if particle_data:

                        n_additional_dims = dataset[key].size // n_particles

                        if n_additional_dims > 1:

                            # Subsample the data
                            subsampled_data_for_key = []
                            for i in range(n_additional_dims):
                                subsampled_data_for_key.append( dataset[key][...][:,i][ subsample_inds ] )

                            # Format the returned data
                            subsampled_data[key] = np.array( subsampled_data_for_key ).transpose()

                        else:

                            subsampled_data[key] = dataset[key][...][ subsample_inds ]

                    # Standard scenario
                    else:
                        values = dataset[key][...].flatten()[subsample_inds]
                        subsampled_data[key] = np.reshape(values, subsample_shape)

                return subsampled_data

        f = h5py.File(self.filename, 'r')

        subsampled_data = {}

        # Subsample differently if we have a number of groups
        first_value = list( f.values() )[0]
        if type(first_value) == h5py._hl.group.Group:
            for key in f.keys():
                subsampled_data[key] = subsample_dataset( f[key] )

                # Include attributes
                group_attrs = {}
                for attr_key in f[key].attrs.keys():

                    # When copying particle data, manually change some of the attrs.
                    if attr_key == 'NumPart_ThisFile':
                        # Don't overwrite when 0 particles
                        nonzero_numpart = f[key].attrs[attr_key] != 0
                        group_attrs[attr_key] = np.where(nonzero_numpart, subsamples, 0)

                    elif attr_key == 'NumPart_Total':
                        # Don't overwrite when 0 particles
                        nonzero_numpart = f[key].attrs[attr_key] != 0
                        group_attrs[attr_key] = np.where(nonzero_numpart, particle_data*subsamples, 0)

                    elif attr_key == 'NumFilesPerSnapshot':
                        group_attrs[attr_key] = particle_data


                    # Standard case
                    else:
                        group_attrs[attr_key] = f[key].attrs[attr_key]

                subsampled_data[key]['attrs'] = group_attrs


        # Standard subsampling
        else:
            subsampled_data = subsample_dataset( f )


        f.close()

        return subsampled_data

    ########################################################################

    def copy_hdf5_file(self, copy_filename, subsamples=False, particle_data=False):
        '''Copy the hdf5 file into a new file. May not copy all attributes successfully?

        Args:
            copy_filename (str): Where the file should be copied to.
            subsamples (int): If chosen, number of subsamples per axis. Defaults to no subsampling.
        '''

        # Open files
        f = h5py.File(self.filename, 'r')

        # Copy the contents
        if not subsamples:

            g = h5py.File(copy_filename, 'a')

            for key in f:
                f.copy(key, g)

            for attr_key in f.attrs:
                g.attrs[attr_key] = f.attrs[attr_key]

        elif subsamples:

            subsampled_data = self.subsample_hdf5_file( subsamples, particle_data=particle_data )

            # Save the data
            g_h5_wrapper = HDF5Wrapper( copy_filename )
            g_h5_wrapper.save_data( subsampled_data )

            g = h5py.File(copy_filename, 'a')

            for attr_key in f.attrs:
                g.attrs[attr_key] = f.attrs[attr_key]

        f.close()

        return g

    ########################################################################

    def clear_hdf5_file(self):
        '''WARNING: Clears the entire hdf5 file of any useful data.'''

        f = h5py.File(self.filename, 'w')
        f.close()

########################################################################

def copy_snapshot( sdir, snum, copy_dir, subsamples=False, redistribute=False, n_files=default ):
    '''Copy a gadget/gizmo snapshot, and subsample it if you choose to.

    Args:
        sdir (str) :
            Snapshot directory to copy from.

        snum (int) :
            Which snapshot to copy over.

        copy_dir (str) :
            Where you want to save the copied snapshot.

        subsamples (bool or int) :
            If not False, the number of particles you will have in your subsampled snapshot,
            per file part.

        redistribute (bool) :
            If True, redistribute the particles evenly among the snapshots.

        n_files (str or int) :
            If not default, the number of pieces you want your copied snapshot broken into
            (must be less than the number of pieces in the original snapshot if subsampling).
            In general, should really only be used when subsampling or redistributing.
    '''

    snapdir = os.path.join( sdir, 'snapdir_{:03}'.format( snum ) )
    copy_snapdir = os.path.join( copy_dir, 'snapdir_{:03}'.format( snum ) )

    # Make sure the path exists
    utilities.make_dir( copy_snapdir )

    assert not (subsamples and redistribute), "Cannot both subsample and redistribute"

    n_files_orig = len( os.listdir( snapdir ) )
    if n_files is default:
        n_files = n_files_orig

    print( 'Starting copying...' )

    if redistribute:

        orig_file_basename = 'snapshot_{:03}.0.hdf5'.format( snum, 0 )
        orig_filename = os.path.join( snapdir, orig_file_basename )

        print( 'Loading all data...' )
        data = {}
        for i in range( n_files_orig ):

            print('  Loading file {}...'.format( i ))

            file_basename = 'snapshot_{:03}.{}.hdf5'.format( snum, i )
            filename = os.path.join( snapdir, file_basename )

            with h5py.File( filename, 'r' ) as f:

                ptypes = copy.copy( list( f.keys() ) )
                del ptypes[0]

                assert not ( 'Header' in ptypes ), "Deleted wrong key..."

                for ptype in ptypes:

                    if i == 0:
                        data[ptype] = {}

                    pdata = f[ptype]
                    for key in pdata.keys():

                        if i == 0:
                            data[ptype][key] = []

                        data[ptype][key].append( pdata[key][...] )

        print( 'Concatenating and splitting data...' )
        for ptype in data.keys():
            print( '  Concatenating {} data'.format( ptype ) )
            for key in data[ptype].keys():

                # Pull into a single snapshot
                data[ptype][key] = np.concatenate( data[ptype][key], axis=0 )
                data[ptype][key] = np.array_split( data[ptype][key], n_files, axis=0 )

        print( 'Storing data...' )
        for i in range( n_files ):

            print('  Copying {} of {}'.format( i+1, n_files ))

            file_basename = 'snapshot_{:03}.{}.hdf5'.format( snum, i )
            copy_filename = os.path.join( copy_snapdir, file_basename )

            shutil.copyfile( filename, copy_filename )

            print( '  Storing {} of {}'.format( i+1, n_files ) )

            with h5py.File( copy_filename, 'a' ) as f:

                n_particles_file = [0,]*len( f['Header'].attrs['NumPart_ThisFile'] )

                for ptype in ptypes:

                    n_particles_file[int(ptype[-1])] = data[ptype]['Masses'][i].size
                    for key in data[ptype].keys():

                        del f[ptype][key]
                        f[ptype][key] = data[ptype][key][i]

                    del f['Header'].attrs['NumPart_ThisFile']
                    f['Header'].attrs['NumPart_ThisFile'] = np.array( n_particles_file ).astype( np.int32 )

                del f['Header'].attrs['NumFilesPerSnapshot']
                f['Header'].attrs['NumFilesPerSnapshot'] = np.int32( n_files )

    else:
        for i in range( n_files ):

            print('  Copying {} of {}'.format( i+1, n_files ))

            file_basename = 'snapshot_{:03}.{}.hdf5'.format( snum, i )
            filename = os.path.join( snapdir, file_basename )
            copy_filename = os.path.join( copy_snapdir, file_basename )

            # Actually make the copy
            h5_wrapper = HDF5Wrapper( filename )
            h5_wrapper.copy_hdf5_file( copy_filename, subsamples=subsamples, particle_data=n_files, )

    print('Done!')

