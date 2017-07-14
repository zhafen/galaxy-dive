#!/usr/bin/env python
'''General utilities

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import inspect
import os
import subprocess

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
    return None

  # Change back
  os.chdir( cwd )

  return code_version

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

