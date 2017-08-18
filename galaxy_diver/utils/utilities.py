#!/usr/bin/env python
'''General utilities

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import collections
from contextlib import contextmanager
from functools import wraps
import inspect
import os
from StringIO import StringIO
import subprocess
import sys
import time

########################################################################
########################################################################

class SmartDict( collections.Mapping ):
  '''Replacement for dictionary that allows easier access to the attributes and methods of the dictionary components.
  For example, if one has a smart dictionary of TestClassA objects, each of which has a TestClassB attribute, which
  in turn have a foo method, then smart_dict.test_class_b.foo(2) would be a dict with foo calculated for each.
  In other words, it would be equivalent to the following code:
  results = {}
  for key in smart_dict.keys():
    results[key] = smart_dict[key].test_class_b.foo( 2 )
  return results

  NOTE: In Python 3, the inheritance class probably needs to be switched to collections.abc.Mapping.
  '''

  def __init__( self, *args, **kwargs ):
    self._storage = dict( *args, **kwargs )

  def __iter__( self ):
    return iter( self._storage )

  def __len__( self ):
    return len( self._storage )

  def __getitem__( self, item ):
    return self._storage[item]

  def __repr__( self ):
    print self._storage

    return 'SmartDict object'

  def __getattr__( self, attr ):

    results = {}
    for key in self.keys():

      results[key] = getattr( self._storage[key], attr )

    return SmartDict( results )

  def __call__( self, *args, **kwargs ):

    results = {}
    for key in self.keys():
        
      results[key] = self._storage[key]( *args, **kwargs )

    return SmartDict( results )

  def call_custom_kwargs( self, kwargs, default_kwargs={} ):
    '''Perform call, but using custom keyword arguments per dictionary tag.

    Args:
      kwargs (dict) : Custom keyword arguments to pass.
      default_kwargs (dict) : Defaults shared between keyword arguments.

    Returns:
      results (dict) : Dictionary of results.
    '''

    used_kwargs = dict_from_defaults_and_variations( default_kwargs, kwargs )

    results = {}
    for key in self.keys():
        
      results[key] = self._storage[key]( **used_kwargs[key] )

    return SmartDict( results )

  ########################################################################
  # Operation Methods
  ########################################################################

  def __mul__( self, other ):

    results = {}

    for key in self.keys():

      results[key] = self._storage[key]*other

    return SmartDict( results )

  __rmul__ = __mul__

  ########################################################################
  # Construction Methods
  ########################################################################

  @classmethod
  def from_class_and_args( cls, contained_cls, args, default_args={}, ):
    '''Alternate constructor. Creates a SmartDict of contained_cls objects, with arguments passed to it from
    the dictionary created by defaults and variations.

    Args:
      contained_cls (type of object/constructor) : What class should the smart dict consist of?
      args (dict/other) : Arguments that should be passed to contained_cls. If not a dict, assumed to be the first
        and only argument for the constructor.
      default_args : Default arguments to fill in args

    Returns:
      result (SmartDict instance) : The constructed instance.
    '''

    kwargs = dict_from_defaults_and_variations( default_args, args )

    storage = {}
    for key in kwargs.keys():
      if isinstance( kwargs[key], dict ):
        storage[key] = contained_cls( **kwargs[key] )
      else:
        storage[key] = contained_cls( kwargs[key] )

    return cls( storage )

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
    
    end_ind += len( l )/n
    
    if remainder_to_distribute > 0:
      end_ind += 1
      remainder_to_distribute -= 1
    
    chunked_l.append( l[ start_ind : end_ind ] )
    
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

# Make a path to a file

def mkdirP(path):

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
  def write(self, *args, **kwargs):
    self.out1.write(*args, **kwargs)
    self.out2.write(*args, **kwargs)
  def __init__(self, out1, out2):
    self.out1 = out1
    self.out2 = out2

########################################################################

# Function for saving to a file.
def saveToLog(log_save_file):

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
    print 'zhh_dataio.getHaloDataRedshift: Bumping redshift from {} to {}'.format(redshift, redshift_arr.min())
    redshift = redshift_arr.min()
  elif redshift > redshift_arr.max():
    print 'zhh_dataio.getHaloDataRedshift: Bumping redshift from {} to {}'.format(redshift, redshift_arr.max())
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

