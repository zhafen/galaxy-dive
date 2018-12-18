#!/usr/bin/env python
'''General utilities

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import collections
import copy
from contextlib import contextmanager
import errno
from functools import wraps
import h5py
import importlib
import inspect
import itertools
import numpy as np
import os
import shutil
import six
import subprocess
import sys
import time
# Works for py2 and py3
try:
    from StringIO import StringIO
except ImportError:
        from io import StringIO

########################################################################

def merge_two_dicts( dict_a, dict_b ):
    '''Merges two dictionaries into a shallow copy.

    Args:
        dict_a (dict) : First dictionary to merge.
        dict_b (dict) : Second dictionary to merge.

    Returns:
        merged_dict (dict) :
            Dictionary including elements from both.
            dict_a's entries take priority over dict_b.
    '''

    merged_dict = dict_b.copy()
    merged_dict.update( dict_a )

    return merged_dict

########################################################################

def dict_from_defaults_and_variations( defaults, variations ):
    '''Create a dictionary of dictionaries from a default dictionary and variations on it.

    Args:
        defaults (dict) : Default dictionary. What each individual dictionary should default to.
        variations (dict of dicts) : Each dictionary contains what should be different. The key for each dictionary should
            be a label for it.

    Returns:
        result (dict of dicts) : The results is basically variations, where each child dict is merged with defaults.
    '''

    if len( defaults ) == 0:
        return variations

    result = {}
    for key in variations.keys():

        defaults_copy = defaults.copy()

        defaults_copy.update( variations[key] )

        result[key] = defaults_copy

    return result

########################################################################

def deepgetattr( obj, attr ):
    '''Recurses through an attribute chain to get the ultimate value.
    Credit to http://pingfive.typepad.com/blog/2010/04/deep-getattr-python-function.html

    Args:
        obj (object) : Object for which to get the attribute.
        attr (str) : Attribute to get. Can be nested, e.g. obj.foo.bar.dog

    Returns:
        result (attr object) : The requested attribute.
    '''

    return reduce( getattr, attr.split('.'), obj )

########################################################################

def arrays_to_set( *args ):
    '''Convert arr1, arr2, arr3, ... to a set, where a single enrty in the set is ( arr1[j], arr2[j], ... )

    Args:
        n arrays ( multiple np.ndarrays ) : The arrays to convert.

    Returns:
        result ( set ) : The converted set.
    '''

    return set( zip( *args ) )

def set_to_arrays( set_to_convert ):
    '''Convert a set into multiple arrays.

    Args:
        result ( set ) : The set to convert

    Returns:
        n arrays ( multiple np.ndarrays ) : The converted arrays
    '''

    return np.array( list( set_to_convert ) ).transpose()

########################################################################

@contextmanager
def captured_output():
    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err

########################################################################

def chunk_list( l, n ):
    '''Breaks a list l into n chunks, as equally as possible.

    Args:
        l (list) : List to break into chunks.
        n (int) : Number of chunks to break the list into.

    Returns:
        chunked_l ( list of lists ) : The list broken into chunks.
    '''

    start_ind = 0
    end_ind = 0
    remainder_to_distribute = len( l ) % n

    chunked_l = []
    for i in range( n ):

        # l/n may not always be an int, so forcing it to be one as before.
        # But the fact that it isn't always an int seems like a bigger problem.
        end_ind += int(len( l )/n)

        if remainder_to_distribute > 0:
            end_ind += 1
            remainder_to_distribute -= 1

        chunk = l[ start_ind : end_ind ]

        chunked_l.append( chunk )

        start_ind = end_ind

    return chunked_l

########################################################################

def get_instance_source_dir( instance, instance_type='class' ):
    '''Get the directory containing the source code of an instance (class or module).

    instance (object) : Instance to get the code for.
    instance_type (str) : 'class' if the instance is a class, 'module' if the instance is a module
    '''

    if instance_type == 'class':
        inspection_object =  instance.__class__
    elif instance_type == 'module':
        inspection_object = instance
    else:
        raise KeyError( "Unrecognized instance_type, {}".format( instance_type ) )

    sourcefile = inspect.getabsfile( inspection_object )

    sourcedir = os.path.dirname( sourcefile )

    return sourcedir

########################################################################

def get_code_version( instance, instance_type='class' ):
    '''Get the current version (git tag/commit) of the parent code for a class or module

    instance (object) : Instance to get the code for.
    instance_type (str) : 'class' if the instance is a class, 'module' if the instance is a module
    '''

    cwd = os.getcwd()

    # Change to the directory of the instance
    instancedir = get_instance_source_dir( instance, instance_type=instance_type )
    os.chdir( instancedir )

    # Get the code version
    try:
        code_version = subprocess.check_output( [ 'git', 'describe', '--always' ] )
    # If we fail, don't break the code.
    except:
        return 'failed to get version'

    # Change back
    os.chdir( cwd )

    return code_version

########################################################################

def print_timer( timer_string='Time taken:' ):
    '''Decorator to time a function.

    Args:
        timer_string (str, optional) : Printed before printing the time.
    '''

    def _print_timer( func ):

        @wraps( func )
        def wrapped_func( *args, **kwargs ):

            time_start = time.time()

            result = func( *args, **kwargs )

            time_end = time.time()

            print_string = '{} {:.3g} seconds'.format( timer_string, time_end - time_start )
            print( print_string )

            return result

        return wrapped_func

    return _print_timer

########################################################################

def store_parameters( constructor ):
    '''Decorator for automatically storing arguments passed to a constructor.
    I.e. any args passed to constructor via test_object = TestObject( *args, **kwargs )
    will be stored in test_object, e.g. test_object.args

    Args:
        constructor (function) : Constructor to wrap.
    '''

    @wraps( constructor )
    def wrapped_constructor( self, *args, **kwargs ):

        parameters_to_store = inspect.getcallargs( constructor, self, *args, **kwargs )

        # Make sure we don't accidentally try to save the self argument
        del parameters_to_store['self']

        for parameter in parameters_to_store.keys():
            setattr( self, parameter, parameters_to_store[parameter] )

        self.stored_parameters = list( parameters_to_store.keys() )

        result = constructor( self, *args, **kwargs )

    return wrapped_constructor

########################################################################

def save_parameters( instance, f ):
    '''Save parameters to a hdf5 file.

    Args:
        instance (object) :
            Instance that has the attributes that are parameters.
            Typically stored in instance.stored_parameters.

        f (open h5py file object) :
            File to save the parameters to.

    Returns:
        param_grp (h5py group) :
            Group containing the parameters as attributes.
    '''

    param_grp = f.create_group( 'parameters' )
    for parameter_str in instance.stored_parameters:

        parameter = getattr( instance, parameter_str )

        try:
            if parameter is None:
                param_grp.attrs[parameter_str] = 'None'
            else:
                param_grp.attrs[parameter_str] = parameter

        except TypeError:
            try:
                # Workaround for h5py poorly handling np string arrays.
                parameter = np.array(
                    parameter,
                    dtype = h5py.special_dtype( vlen=six.text_type ),
                )
                param_grp.attrs[parameter_str] = parameter
            except TypeError:
                raise TypeError(
                    "Parameter {} = {} failed to save.".format(
                            parameter_str,
                            parameter
                        )
                    )

    return param_grp

########################################################################

def check_and_decode_bytes( x, coding='utf-8' ):
    '''Check if an object is a bytes object, if so decode it into a string.

    Args:
        x (str or bytes):
            Object to check.

        coding (str):
            What coding to use for decoding bytes objects.

    Returns:
        str:
            The original string or the decoded x.
    '''

    if isinstance( x, bytes ):
        x_out = x.decode( coding )
    else:
        x_out = copy.deepcopy( x )

    return x_out

########################################################################

def gen_derived_data_doc(
    object_import_path,
    save_filepath,
    header_file = None,
):
    '''Generate a document that lists the methods for calculating derived data
    quantities.

    Args:
        object_import_path (str):
            Name of the object to document the derived quantities of.

        save_filepath (str):
            Where to save the result.

        header_file (str):
            What to use as the header for the file, that explains what the
            document contains.
            Defaults to galaxy_dive/utils/derived_data_header.rst.
        
    Modifies:
        file at save_filepath (.rst file):
            Saves the output here.
    '''

    # Process the path
    module_path, class_name = os.path.splitext( object_import_path )
    # Remove the period
    class_name = class_name[1:]

    # Get the object out
    module = importlib.import_module( module_path )
    cl = getattr( module, class_name )

    # Setup the file
    if header_file is None:
        header_file = os.path.join(
            os.path.dirname( __file__ ),
            'derived_data_header.rst',
        )
    shutil.copy( header_file, save_filepath )

    # Write the automatically generated part
    with open( save_filepath, 'a' ) as f:

        # Class line
        f.write( '\n.. autoclass:: {}\n'.format( object_import_path ) )
        f.write( '    :show-inheritance:\n\n' )

        # Write out derived data methods
        for attr in dir( cl ):
            if attr[:5] == 'calc_':
                f.write( '    .. automethod:: {}\n'.format( attr ) )

########################################################################

# Make a path to a file

def make_dir( path ):

    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

    return 0

########################################################################

# Class modelled after unix tee, mainly for the purpose of writing to a log while also printing.

class Tee:
    def write( self, *args, **kwargs ):
        self.out1.write( *args, **kwargs )
        self.out2.write( *args, **kwargs )
    def __init__( self, out1, out2 ):
        self.out1 = out1
        self.out2 = out2

########################################################################

# Function for saving to a file.
def saveToLog( log_save_file ):

    sys.stdout = Tee(open(log_save_file, 'w'), sys.stdout)

    return 0

########################################################################
# Tools for dealing with cosmology-specific data files.
########################################################################

# Get information from a halo file for a particular redshift.

def getHaloDataRedshift(sdir, halo_number, redshift, convert_to_proper=True):
# Values returned are, in order, and in comoving units,
#   redshift, halo_ID, host_ID, xpeak, ypeak, zpeak, Rvir, Mvir, Mgas, Mstar, xCOM, yCOM, zCOM.

    # Get the halo data for all snapshots
    halo_file = getHaloFilename(sdir, halo_number)
    full_halo_data = dataio.tuples_to_arrays(dataio.read_data(halo_file))
    redshift_arr = full_halo_data[0]

    # Account for rounding errors at the edges.
    if redshift < redshift_arr.min():
        print( 'zhh_dataio.getHaloDataRedshift: Bumping redshift from {} to {}'.format(redshift, redshift_arr.min()) )
        redshift = redshift_arr.min()
    elif redshift > redshift_arr.max():
        print( 'zhh_dataio.getHaloDataRedshift: Bumping redshift from {} to {}'.format(redshift, redshift_arr.max()) )
        redshift = redshift_arr.max()

    try:
        # Get single values out of the halo data.
        arr_value_list = []
        for arr in full_halo_data:
            arr_interp = scipy.interpolate.interp1d(redshift_arr[::-1], arr[::-1])
            arr_value = arr_interp(redshift)
            arr_value_list.append(arr_value)
    except ValueError:
        arr_value_list = np.array(full_halo_data).flatten()

    # Convert halo data to proper coordinates (the 1/h factor's still there for many of the coords)
    if convert_to_proper==True:
        comoving_vals = [3, 4, 5, 6, 10, 11, 12]
        for val_index in comoving_vals:
            arr_value_list[val_index] /= (1. + arr_value_list[0])

    return arr_value_list

########################################################################

# Get particle data and useful halo file data for a given data, all in proper units. Combine them into a class.

class PartDataLong(object):

    def __init__(self, sdir, snum, ptype, halo_number=0):
        self.sdir = sdir
        self.snum = snum
        self.ptype = ptype
        self.halo_number = halo_number

        # Get the particles from a snapshot (in proper units)
        self.P = readsnap.readsnap(sdir, snum, ptype, cosmological=1)

        # Get the halo data
        halo_data = getHaloDataRedshift(sdir, halo_number, self.P['redshift'])

        # Get rid of 1/h factors in the halo data
        vals_to_be_converted = range(3, 13)
        for val_index in vals_to_be_converted:
            halo_data[val_index] /= self.P['hubble']

        # Add the halo data to the class.
        self.redshift = halo_data[0]
        self.halo_ID = halo_data[1]
        self.host_ID = halo_data[2]
        self.peak_coords = (halo_data[3], halo_data[4], halo_data[5])
        self.R_vir = halo_data[6]
        self.M_vir = halo_data[7]
        self.M_gas = halo_data[8]
        self.M_star = halo_data[9]
        self.CoM_coords = (halo_data[10], halo_data[11], halo_data[12])

    ########################################################################

    # Change coordinates to center on the main halo.
    def changeCoordsCenter(self):
        self.P['p'] -= np.array(self.peak_coords)

    ########################################################################

    # Calculate the distance from the origin for a given particle.
    def calcRadialDistance(self):

        coords = self.P['p']
        x, y, z = coords.transpose()
        self.P['r'] = np.sqrt(x**2. + y**2. + z**2.)

        return self.P['r']

