########################################################################
#
#  File Name: zhh_dataio.py
#  Author: Zach Hafen (zachary.h.hafen@gmail.com)
#  Purpose: Data Input/Output Processing classes and functions.
#
########################################################################

import copy
import errno
import h5py
import numpy as np
import os
import transformations
from scipy import ndimage
import scipy.interpolate
import string
import sys

import dataio
from gadget_lib import readsnap

########################################################################
# Classes and Functions for dealing with direct output (print statements)
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

def snum_to_redshift(snum, output_times_file='/home1/03057/zhafen/repos/zhh_python/tools/output_times.txt'):
# snum : Snapshot number
# output_times_file : Which file that stores the output times to use.
#                     The output times should be the scale factor a=1/(1+z)

  # From the file.
  output_times = np.loadtxt(output_times_file)

  # Get the redshift
  z = 1/output_times - 1

  # Return the value back (the snapshot number just acts as indices)
  return z[snum]

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

########################################################################
# Functions for dealing with files
########################################################################

# Create a copy of a hdf5 file and return it.

def copyHDF5(source_filename, copy_filename):

  # Open files
  f = h5py.File(source_filename, 'r')
  g = h5py.File(copy_filename, 'a')

  # Copy the contents
  for key in f:
    f.copy(key, g)
  for attr_key in f.attrs:
    g.attrs[attr_key] = f.attrs[attr_key]

  f.close()

  return g

########################################################################

# Add a character after the end of a general number of columns (this function is effectively made only to put MUSIC input files of points in a form legible by MUSIC, thus the default added character is a space).

def addCharEndLine(source_filename, char=' '):

  data = np.loadtxt(source_filename)

  np.savetxt(source_filename, data, fmt='%.6f', delimiter=char, newline=(char + '\n'))

  return 0

########################################################################
# Functions for dealing with labels and names 
########################################################################

# Get abbreviated names for simulations based off of their snapshot directory.

def abbreviatedName(snap_dir):

  # Dissect the name for the useful abbreviations.
  divided_sdir = string.split(snap_dir, '/')
  if divided_sdir[-1] == '':
    del divided_sdir[-1]
  if divided_sdir[-1] == 'output':
    sim_name_index = -2
  else:
    sim_name_index = -1
  sim_name_full = divided_sdir[sim_name_index]
  sim_name_divided = string.split(sim_name_full, '_')
  sim_name = sim_name_divided[0]

  # Attach the appropriate name, accounting for the sometimes weird naming conventions.
  if sim_name == 'B1':
    sim_name = 'm12v'
  elif sim_name == 'm12qq':
    sim_name = 'm12q'
  elif sim_name == 'm12v':
    sim_name = 'm12i'
  elif sim_name == 'massive':
    sim_name = 'MFz0_A2'
  elif sim_name == 'm12c':
    sim_name = 'm11.4a'
  elif sim_name == 'm12d':
    sim_name = 'm11.9a'

  # Check for alternate runs (e.g. turbdiff)
  if len(sim_name_divided) > 1:
    if sim_name_divided[-1] == 'turbdiff':
      sim_name += 'TD'
    elif sim_name_divided[1] == 'res7000':
      sim_name += '2'
    
      # Account for variants
      for addition in sim_name_divided[2:]:
        sim_name += addition

  return sim_name

########################################################################

# Get the abbreviated names for a list of simulations.

def abbreviated_name_list(snap_dir_list):
    # Loop over all simulations.
    sim_name_list = []
    for snap_dir in snap_dir_list:

      sim_name = abbreviatedName(snap_dir)

      sim_name_list.append(sim_name)

    return sim_name_list

########################################################################

# Get the ionization tag for grid and LOS filenames.

def ionTag(ionized):

  if ionized == False:
    ionization_tag = ''
    return ionization_tag

  elif ionized == 'RT':
    ionization_tag = '_ionized'
  elif string.split(ionized, '_')[0] == 'R13':
    ionization_tag = '_' + ionized
  else:
    ionization_tag = '_' + ionized

  return ionization_tag

########################################################################

# Get the gridded snapshot filename from a few inputs

def getGridFilename(sim_dir, snap_id, Nx, gridsize, ionized, ion_grid=False):
# sim_dir: Simulation directory, string.
# snap_id: Snapshot ID, integer
# Nx: Grid resolution, integer
# gridsize: Grid size, float or string

  ionization_tag = ionTag(ionized)

  grid_filename = 'grid_%i_%i_%s_sasha%s.hdf5'%(snap_id, Nx, str(gridsize), ionization_tag)

  if ion_grid:
    grid_filename = 'ion_' + grid_filename
  
  grid_path = os.path.join(sim_dir, grid_filename)

  return grid_path

########################################################################

# Break apart gridded snapshot file names

def breakGridFilename(gridded_snapshot_file):

  # Break the simulation path apart
  path_to_file = string.split(gridded_snapshot_file, '/')
  simulation = path_to_file[-2]
  grid_file = path_to_file[-1]

  # Break the grid file apart
  seperated_file = string.split(grid_file, '_')
  snap_id = seperated_file[1]
  Nx = seperated_file[2]
  gridsize = seperated_file[3]

  return (simulation, snap_id, Nx, gridsize)

########################################################################

# Get the halofile name from a few inputs

def getHaloFilename(sim_dir, halo_number):
  
  return '{}/halo_{:0>5d}.datsimple.txtsmooth'.format(sim_dir, halo_number)

########################################################################

# Get the location of the LOS data from a few inputs.

def getLOSDataFilename(snap_dir, Nx, gridsize, face, comp_method, ionized=False, den_weight='nH'):

  # Simulation name
  snap_dir_divided = string.split(snap_dir,'/')
  if snap_dir_divided[-1] == 'output':
    snap_dir_name = snap_dir_divided[-2]
  else:
    snap_dir_name = string.split(snap_dir,'/')[-1]

  ionization_tag = ionTag(ionized)

  if den_weight == 'nH':
    den_weight_tag = ''
  else:
    den_weight_tag = '_' + den_weight

  return '/work/03057/zhafen/LOSdata/LOS_{}_{}_{}_{}_{}{}{}.hdf5'.format(snap_dir_name, Nx, str(gridsize), face, comp_method, ionization_tag, den_weight_tag)

