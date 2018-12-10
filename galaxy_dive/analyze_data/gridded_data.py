#!/usr/bin/env python
'''Subclass for analyzing particle data.

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import h5py
import numpy as np

import galaxy_dive.utils.io as io
import galaxy_dive.analyze_data.simulation_data as simulation_data
import galaxy_dive.utils.utilities as utilities

########################################################################

class GriddedData( simulation_data.SnapshotData ):
  '''Class for handling data that forms a Cartesian grid.'''

  @utilities.store_parameters
  def __init__( self, sdir=None, snum=None, Nx=None, gridsize=None, ionized=None, ion_grid=False, **kwargs ):
    '''
    Args:
      sdir (str) : Directory the snaphost is stored in
      snum (str) : Snapshot number
      Nx (int) : Number of grid cells on a side
      gridsize (float or str) : How large the grid is
      ionized (str) : Legacy name. Originally how the ionization state is calculated. 
                        Currently any identifying string added to the end of the filename.
                        'R13' : Ionization state calculated using Rahmati+13 fitting function
      ion_grid (bool) : Whether or not this is a grid containing ion information.
    '''

    # Note that we assume the grid is centered.
    super( GriddedData, self ).__init__( data_dir=sdir, snum=snum, centered=True, **kwargs )

    self.retrieve_data()

  ########################################################################

  def retrieve_data( self ):

    # Open file
    self.grid_file_name = io.getGridFilename( self.sdir, self.snum, self.Nx, self.gridsize, self.ionized, \
                          ion_grid=self.ion_grid )
    f = h5py.File( self.grid_file_name, 'r' )

    # Get the grid attributes
    self.data_attrs = {}
    for key in f.attrs.keys():
      self.data_attrs[key] = f.attrs[key]

    def load_data(f):

      # Get the data
      self.data = {}
      for key in f.keys():

        # Parse the keys and put in a more general format
        # All units should be the standard GIZMO output units
        # Gas Density
        if key == 'GasDen':
          self.data['Den'] = f[key][...]
        # Gas Neutral Hydrogen fraction
        elif key == 'GasNHI':
          self.data['nHI'] = f[key][...]
        # Should be the mean free electron number per proton
        elif key == 'GasNe':
          self.data['ne'] = f[key][...]
        # Same as above, but due to helium
        elif key == 'GasNe_He':
          self.data['ne_He'] = f[key][...]
        # Star Formation Rate
        elif key == 'GasSFR':
          self.data['SFR'] = f[key][...]
        # Temperature
        elif key == 'GasT':
          self.data['T'] = f[key][...]
        # X Velocity
        elif key == 'GasVx':
          vx = f[key][...]
        # Y Velocity
        elif key == 'GasVy':
          vy = f[key][...]
        # Z Velocity
        elif key == 'GasVz':
          vz = f[key][...]
        # Metallicity
        elif key == 'GasZ':
          self.data['Z'] = f[key][...]
        else:
          raise Exception('NULL key')

      # Finish organizing velocity
      self.data['V'] = np.array([vx, vy, vz])

    def load_ion_data(f):

      self.data = {}
      for key in f.keys():
        self.data[key] = f[key][...]

    # Load the data
    if not self.ion_grid:
      load_data(f)
    elif self.ion_grid:
      load_ion_data(f)
    else:
      raise Exception('Unrecognized ion_grid')

    f.close()

  ########################################################################
  ########################################################################

  def handle_data_key_error( self, data_key ):
    '''When get_data() fails to data_key in self.data, it passes the data_key to try and generate that data.

    Args:
      data_key (str) : Key to try and generate data for

    Modifies:
      self.data[data_key] (np.array) : If it finds a function to generate the data, it will do so
    '''

    if data_key ==  'Rx' or data_key ==  'Ry' or data_key ==  'Rz' or data_key == 'P':
      self.calc_positions()
    elif data_key[:-3] ==  'Rx_face' or data_key[:-3] ==  'Ry_face' or data_key[:-3] ==  'Rz_face':
      self.calc_face_positions( data_key )
    elif data_key[:-3] ==  'R_face':
      self.calc_impact_parameter( data_key )
    elif data_key == 'M':
      self.calc_mass()

    else:
      super( GriddedData, self ).handle_data_key_error( data_key )

  ########################################################################
  ########################################################################

  def calc_positions( self ):
    '''Calculate the positions of gridcells.'''

    # Get how the spacing on one side
    Nx = list( self.data.values() )[0].shape[0]
    gridsize = self.data_attrs['gridsize']
    pos_coords = np.linspace(-gridsize/2., gridsize/2., Nx)

    # Mesh the spacing together for all the grid cells
    x, y, z = np.meshgrid(pos_coords, pos_coords, pos_coords, indexing='ij')

    self.data['P'] = np.array([x, y, z])

  ########################################################################

  def calc_face_positions( self, data_key ):
    '''Calculate positions if you're just looking at one face of a grid.'''

    # Figure out which face to calculate for
    target_face = data_key.split( '_' )[-1]

    if target_face == 'xy':
      self.data['Rx_face_xy'] = self.get_data( 'Rx' )[:, :, 0]
      self.data['Ry_face_xy'] = self.get_data( 'Ry' )[:, :, 0]
    elif target_face == 'xz':
      self.data['Rx_face_xz'] = self.get_data( 'Rx' )[:, 0, :]
      self.data['Rz_face_xz'] = self.get_data( 'Rz' )[:, 0, :]
    elif target_face == 'yz':
      self.data['Ry_face_yz'] = self.get_data( 'Ry' )[0, :, :]
      self.data['Rz_face_yz'] = self.get_data( 'Rz' )[0, :, :]

  ########################################################################

  def calc_impact_parameter( self, data_key ):

    # Figure out which face to calculate for
    target_face = data_key.split( '_' )[-1]
    
    if target_face == 'xy':
      self.data[data_key] = np.sqrt( self.get_data( 'Rx_face_xy' )**2. + self.get_data( 'Ry_face_xy' )**2. ) 
    elif target_face == 'xz':
      self.data[data_key] = np.sqrt( self.get_data( 'Rx_face_xz' )**2. + self.get_data( 'Ry_face_xz' )**2. ) 
    elif target_face == 'yz':
      self.data[data_key] = np.sqrt( self.get_data( 'Rx_face_yz' )**2. + self.get_data( 'Ry_face_yz' )**2. ) 

  ########################################################################

  def calc_mass( self ):
    '''Calculate the mass per grid cell.'''

    # Calculate the mass per grid cell, for easy use later
    self.data['M'] = self.data['Den']*(self.data_attrs['gridsize']/self.data_attrs['Nx'])**3.

  ########################################################################

  def calc_column_density( self, key, face ):
    '''Calculate the column density for a targeted key.

    Args --
    key (str) : The data to project.
    face (int) : The face to use. 0=yz, 1=xz, 2=xy.

    Returns --
    key_col_den (np.array) : The column density in {key units}/cm^2. Assumes the grid uses fiducial GIZMO units.
    '''

    key_data = self.get_data( key )

    # Get the projected data
    summed_key_data = key_data.sum(face)

    # Get the column density
    self.data_attrs['cell_length'] = self.data_attrs['gridsize']/float( self.data_attrs['Nx'] - 1 )
    dx = self.data_attrs['cell_length']*3.086e21
    key_col_den = dx*summed_key_data

    return key_col_den


