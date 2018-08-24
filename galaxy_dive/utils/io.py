#!/usr/bin/env python
'''Input and output

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import h5py
import numpy as np
import os

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
  divided_sdir = snap_dir.split('/')
  if divided_sdir[-1] == '':
    del divided_sdir[-1]
  if divided_sdir[-1] == 'output':
    sim_name_index = -2
  else:
    sim_name_index = -1
  sim_name_full = divided_sdir[sim_name_index]
  sim_name_divided = sim_name_full.split('_')
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
  elif ionized.split('_')[0] == 'R13':
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
  path_to_file = gridded_snapshot_file.split('/')
  simulation = path_to_file[-2]
  grid_file = path_to_file[-1]

  # Break the grid file apart
  seperated_file = grid_file.split('_')
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
  snap_dir_divided = snap_dir.split('/')
  if snap_dir_divided[-1] == 'output':
    snap_dir_name = snap_dir_divided[-2]
  else:
    snap_dir_name = snap_dir.split('/')[-1]

  ionization_tag = ionTag(ionized)

  if den_weight == 'nH':
    den_weight_tag = ''
  else:
    den_weight_tag = '_' + den_weight

  return '/work/03057/zhafen/LOSdata/LOS_{}_{}_{}_{}_{}{}{}.hdf5'.format(snap_dir_name, Nx, str(gridsize), face, comp_method, ionization_tag, den_weight_tag)

