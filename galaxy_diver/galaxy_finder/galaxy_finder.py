#!/usr/bin/env python
'''Means to associate particles with galaxies and halos at any given time.

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import gc
import h5py
import numpy as np
import os
import scipy.spatial
import subprocess
import sys
import time

import galaxy_diver.read_data.ahf as read_ahf
import galaxy_diver.utils.utilities as utilities

########################################################################
########################################################################

class GalaxyFinder( object ):
  '''Find the association with galaxies and halos for a given set of particles at a given redshift.'''

  @utilities.store_parameters
  def __init__( self,
    particle_positions,
    redshift,
    snum,
    hubble,
    galaxy_cut,
    length_scale,
    minimum_criteria,
    minimum_value,
    particle_masses = None,
    ids_to_return = None,
    ahf_reader = None,
    ahf_data_dir = None,
    halo_file_tag = None,
    mtree_halos_index = None,
    main_mt_halo_id = None,
    ):
    '''Initialize.

    Args:
      particle_positions (np.array) :
        Positions with dimensions (n_particles, 3).

      redshift (float) :
        Redshift the particles are at.

      snum (int) :
        Snapshot the particles correspond to.

      hubble (float) :
        Cosmological hubble parameter (little h)

      galaxy_cut (float) :
        The fraction of the length scale a particle must be inside to be counted as part
        of a galaxy.

      length_scale (str) :
        Anything within galaxy_cut*length_scale is counted as being inside the galaxy.

      minimum_criteria (str) :
        Options...
        'n_star' -- halos must contain a minimum number of stars to count as containing a galaxy.
        'M_star' -- halos must contain a minimum stellar mass to count as containing a galaxy.

      minimum_value (int or float) :
        The minimum amount of something (specified in minimum criteria)
        in order for a galaxy to count as hosting a halo.

      particle_masses (np.ndarray, optional) :
        Masses of particles.

      ids_to_return (list of strs, optional) :
        The types of id you want to get out.

      ahf_reader (AHFReader object, optional) :
        An instance of an object that reads in the AHF data.
        If not given initiate one using the ahf_data_dir in kwargs

      ahf_data_dir (str, optional) :
        Directory the AHF data is in. Necessary if ahf_reader is not provided.

      halo_file_tag (int, optional) :
        What identifying tag to use for the merger tree files? Necessary if using merger tree information.

      mtree_halos_index (str or int, optional)  :
        The index argument to pass to AHFReader.get_mtree_halos().
        For most cases this should be the final snapshot number, but see AHFReader.get_mtree_halos's documentation.

      main_mt_halo_id (int, optional) :
        Index of the main merger tree halo.
    '''

    # Setup the default ahf_reader
    if ahf_reader is None:
      self.ahf_reader = read_ahf.AHFReader( self.ahf_data_dir )

    # In the case of a minimum stellar mass, we need to divide the minimum value by 1/h when getting its values out.
    if self.minimum_criteria == 'M_star':
      self.min_conversion_factor = self.hubble 
    else:
      self.min_conversion_factor = 1

    # Derived properties
    self.n_particles = self.particle_positions.shape[0]

  ########################################################################
  # Properties
  ########################################################################

  @property
  def valid_halo_inds( self ):
    '''
    Returns:
      valid_halo_inds (np.ndarray) :
        Indices of *AHF_halos halos that satisfy our minimum criteria for containing a galaxy.
    '''

    if not hasattr( self, '_valid_halo_inds' ):

      self.ahf_reader.get_halos( self.snum )

      # Apply a cut on containing a minimum amount of stars
      min_criteria = self.ahf_reader.ahf_halos[ self.minimum_criteria ]
      has_minimum_value = min_criteria/self.min_conversion_factor >= self.minimum_value

      # Figure out which indices satisfy the criteria and choose only those halos
      self._valid_halo_inds = np.where( has_minimum_value )[0]

    return self._valid_halo_inds

  ########################################################################

  @property
  def dist_to_all_valid_halos( self ):
    '''
    Returns:
      dist_to_all_valid_halos (np.ndarray) :
        Distance between the particle positions and all *.AHF_halos halos containing a galaxy (in pkpc).
    '''

    if not hasattr( self, '_dist_to_all_valid_halos' ):

      self.ahf_reader.get_halos( self.snum )
        
      # Get the halo positions
      halo_pos_comov = np.array([
        self.ahf_reader.ahf_halos['Xc'],
        self.ahf_reader.ahf_halos['Yc'],
        self.ahf_reader.ahf_halos['Zc'],
      ]).transpose()
      halo_pos = halo_pos_comov/( 1. + self.redshift )/self.hubble
      halo_pos_selected = halo_pos[self.valid_halo_inds]

      # Get the distances
      # Output is ordered such that dist[:,0] is the distance to the center of halo 0 for each particle
      self._dist_to_all_valid_halos = scipy.spatial.distance.cdist( self.particle_positions, halo_pos_selected )

    return self._dist_to_all_valid_halos

  ########################################################################

  @property
  def ahf_halos_length_scale_pkpc( self ):

    if not hasattr( self, '_ahf_halos_length_scale_pkpc' ):

      # Get the relevant length scale
      if self.length_scale == 'R_vir':
        length_scale = self.ahf_reader.ahf_halos['Rvir']
      elif self.length_scale == 'r_scale':
        # Get the files containing the concentration (counts on it being already calculated beforehand)
        self.ahf_reader.get_halos_add( self.ahf_reader.ahf_halos_snum )

        # Get the scale radius
        r_vir = self.ahf_reader.ahf_halos['Rvir']
        length_scale = r_vir/self.ahf_reader.ahf_halos['cAnalytic']
      else:
        raise KeyError( "Unspecified length scale" )
      self._ahf_halos_length_scale_pkpc = length_scale/( 1. + self.redshift )/self.hubble

    return self._ahf_halos_length_scale_pkpc

  ########################################################################

  @property
  def mass_inside_galaxy_cut( self ):
    '''
    Returns:
      mass_inside_galaxy_cut (np.ndarray) :
        Mass inside the galaxy_cut*length_scale for *.AHF_halos halos that containing a galaxy.
    '''

    if not hasattr( self, '_mass_inside_galaxy_cut' ):

      valid_radial_cut_pkpc = self.galaxy_cut*self.ahf_halos_length_scale_pkpc[self.valid_halo_inds]
      outside_radial_cut = self.dist_to_all_valid_halos > valid_radial_cut_pkpc

      mass_tiled = np.tile( self.particle_masses, ( self.valid_halo_inds.size, 1 ) ).transpose()

      mass_ma = np.ma.masked_array( mass_tiled, mask=outside_radial_cut )

      self._mass_inside_galaxy_cut = mass_ma.sum( axis=0 )

    return self._mass_inside_galaxy_cut

  ########################################################################
  # ID Finding Routines
  ########################################################################

  def find_ids( self ):
    '''Find relevant halo and galaxy IDs.

    Returns:
      galaxy_and_halo_ids (dict): Keys are...
      Parameters:
        halo_id (np.array of ints): ID of the least-massive halo the particle is part of.
        host_halo_id (np.array of ints): ID of the host halo the particle is part of.
        gal_id (np.array of ints): ID of the smallest galaxy the particle is part of.
        host_gal_id (np.array of ints): ID of the host galaxy the particle is part of.
        mt_halo_id (np.array of ints): Merger tree ID of the least-massive halo the particle is part of.
        mt_gal_id (np.array of ints): Merger tree ID of the smallest galaxy the particle is part of.
    '''

    # Dictionary to store the data in.
    galaxy_and_halo_ids = {}

    try:
      # Load the ahf data
      self.ahf_reader.get_halos( self.snum )

    # Typically halo files aren't created for the first snapshot.
    # Account for this.
    except NameError:
      if self.snum == 0:
        for id_type in self.ids_to_return:
          galaxy_and_halo_ids[id_type] = np.empty( self.n_particles )
          galaxy_and_halo_ids[id_type].fill( -2. )

        return galaxy_and_halo_ids

      else:
        raise KeyError( 'AHF data not found for snum {} in {}'.format( self.snum,
                                                                       self.ahf_data_dir ) )
    
    # Actually get the data
    for id_type in self.ids_to_return:
      if id_type == 'halo_id':
        galaxy_and_halo_ids['halo_id'] = self.find_halo_id()
      elif id_type == 'host_halo_id':
        galaxy_and_halo_ids['host_halo_id'] = self.find_host_id()
      elif id_type == 'gal_id':
        galaxy_and_halo_ids['gal_id'] = self.find_halo_id( self.galaxy_cut )
      elif id_type == 'host_gal_id':
        galaxy_and_halo_ids['host_gal_id'] = self.find_host_id( self.galaxy_cut )
      elif id_type == 'mt_halo_id':
        galaxy_and_halo_ids['mt_halo_id'] = self.find_halo_id( type_of_halo_id='mt_halo_id' )
      elif id_type == 'mt_gal_id':
        galaxy_and_halo_ids['mt_gal_id'] = self.find_halo_id( self.galaxy_cut, type_of_halo_id='mt_halo_id' )
      elif id_type == 'd_gal':
        galaxy_and_halo_ids['d_gal'] = self.find_d_gal()
      elif id_type == 'd_other_gal':
        galaxy_and_halo_ids['d_other_gal'] = self.find_d_other_gal()
      elif id_type == 'd_other_gal_scaled':
        galaxy_and_halo_ids['d_other_gal_scaled'] = self.find_d_other_gal( scaled=True )
      else:
        raise Exception( "Unrecognized id_type" )
    
    return galaxy_and_halo_ids

  ########################################################################

  def find_d_gal( self ):
    '''Find the distance to the center of the closest halo that contains a galaxy.

    Returns:
      d_gal (np.ndarray) : For particle i, d_gal[i] is the distance in pkpc to the center of the nearest galaxy.
    '''

    # Handle when no halos exist.
    if self.ahf_reader.ahf_halos.size == 0:
      return -2.*np.ones( (self.n_particles,) )
  
    return np.min( self.dist_to_all_valid_halos, axis=1 )

  ########################################################################

  def find_d_other_gal( self, scaled=False ):
    '''Find the distance to the center of the closest halo that contains a galaxy, other than the main galaxy.

    Returns:
      d_other_gal (np.ndarray) :
        For particle i, d_other_gal[i] is the distance in pkpc to the center of the nearest galaxy, besides the main galaxy.
    '''

    # Handle when no halos exist.
    if self.ahf_reader.ahf_halos.size == 0:
      return -2.*np.ones( (self.n_particles,) )

    # Handle when all the halos aren't massive enough
    if self.valid_halo_inds.size == 0:
      return -2.*np.ones( (self.n_particles,) )

    self.ahf_reader.get_mtree_halos( self.mtree_halos_index, self.halo_file_tag )

    mtree_halo = self.ahf_reader.mtree_halos[ self.main_mt_halo_id ]

    if self.snum < mtree_halo.index.min():
      # This mimics what would happen if ind_main_gal wasn't in self.valid_halo_inds
      ind_main_gal_in_valid_inds = np.array( [] )
    else:
      # The indice for the main galaxy is the same as the AHF_halos ID for it.
      ind_main_gal = mtree_halo['ID'][ self.snum ]

      valid_halo_ind_is_main_gal_ind = self.valid_halo_inds == ind_main_gal 
      ind_main_gal_in_valid_inds = np.where( valid_halo_ind_is_main_gal_ind )[0]

    if ind_main_gal_in_valid_inds.size == 0:
      dist_to_all_valid_other_gals = self.dist_to_all_valid_halos
      valid_halo_inds_sats = self.valid_halo_inds

    elif ind_main_gal_in_valid_inds.size == 1:
      dist_to_all_valid_other_gals = np.delete( self.dist_to_all_valid_halos, ind_main_gal_in_valid_inds[0], axis=1 )
      valid_halo_inds_sats = np.delete( self.valid_halo_inds, ind_main_gal_in_valid_inds[0] )

    else:
      raise Exception( "ind_main_gal_in_valid_inds too big, is size {}".format( valid_ind_main_gal.size ) )

    if not scaled:
      return np.min( dist_to_all_valid_other_gals, axis=1 )

    inds_sat = np.argmin( dist_to_all_valid_other_gals, axis=1 )

    # Now scale
    length_scale_sats = self.ahf_halos_length_scale_pkpc[ valid_halo_inds_sats ]

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

    ahf_host_id =  self.ahf_reader.ahf_halos['hostHalo']

    # Handle the case where we have an empty ahf_halos, because there are no halos at that redshift.
    # In this case, the ID will be -2 throughout
    if ahf_host_id.size == 0:
      return halo_id

    # Get the host halo ID
    host_id = ahf_host_id[ halo_id ]

    # Fix the invalid values (which come from not being associated with any halo)
    host_id_fixed = np.ma.fix_invalid( host_id, fill_value=-2 )

    return host_id_fixed.data.astype( int )

  ########################################################################

  def find_halo_id( self, radial_cut_fraction=1., type_of_halo_id='halo_id' ):
    '''Find the smallest halos our particles are inside of some radial cut of (we define this as the halo ID).
    In the case of using MT halo ID, we actually find the most massive our particles are inside some radial cut of.

    Args:
      radial_cut_fraction (float): A particle is in a halo if it's in radial_cut_fraction*length_scale from the center.
      type_of_halo_id (str): If 'halo_id' then this is the halo_id at a given snapshot.
                             If 'mt_halo_id' then this is the halo_id according to the merger tree.

    Returns:
      halo_id (np.array of ints): Shape ( n_particles, ). 
        The ID of the least massive substructure the particle's part of.
        In the case of using the 'mt_halo_id', this is the ID of the most massive merger tree halo the particle's part of.
        If it's -2, then that particle is not part of any halo, within radial_cut_fraction*length_scale .
    '''

    # Choose parameters of the rest of the function based on what type of halo ID we're using
    if type_of_halo_id == 'halo_id':

      # Get the virial masses. It's okay to leave in comoving, since we're just finding the minimum
      m_vir = self.ahf_reader.ahf_halos['Mvir']

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
      m_vir = self.ahf_reader.get_mtree_halo_quantity( quantity='Mvir', indice=self.snum,
                                                       index=self.mtree_halos_index, tag=self.halo_file_tag )

    else:
      raise Exception( "Unrecognized type_of_halo_id" )

    # Get the cut
    part_of_halo = find_containing_halos_fn( radial_cut_fraction=radial_cut_fraction )

    # Mask the data
    tiled_m_vir = np.tile( m_vir, ( self.n_particles, 1 ) )
    tiled_m_vir_ma = np.ma.masked_array( tiled_m_vir, mask=np.invert( part_of_halo ), )

    # Take the extremum of the masked data
    if type_of_halo_id == 'halo_id':
      halo_id = arg_extremum_fn( tiled_m_vir_ma, axis=1 )
    elif type_of_halo_id == 'mt_halo_id':
      halo_ind = arg_extremum_fn( tiled_m_vir_ma, axis=1 )
      halo_ids = np.array( sorted( self.ahf_reader.mtree_halos.keys() ) )
      halo_id = halo_ids[halo_ind]
    
    # Account for the fact that the argmin defaults to 0 when there's nothing there
    mask = extremum_fn( tiled_m_vir_ma, axis=1 ).mask
    halo_id = np.ma.filled( np.ma.masked_array(halo_id, mask=mask), fill_value=-2 )

    return halo_id

  ########################################################################

  def find_containing_halos( self, radial_cut_fraction=1. ):
    '''Find which halos our particles are inside of some radial cut of.

    Args:
      radial_cut_fraction (float): A particle is in a halo if it's in radial_cut_fraction*length_scale from the center.

    Returns:
      part_of_halo (np.array of bools): Shape (n_particles, n_halos). 
        If index [i, j] is True, then particle i is inside radial_cut_fraction*length_scale of the jth halo.
    '''

    # Get the radial cut
    radial_cut = radial_cut_fraction*self.ahf_halos_length_scale_pkpc[self.valid_halo_inds]

    # Find the halos that our particles are part of (provided they passed the minimum cut)
    part_of_halo_success = self.dist_to_all_valid_halos < radial_cut[np.newaxis,:]

    # Get the full array out
    part_of_halo = np.zeros( (self.n_particles, self.ahf_halos_length_scale_pkpc.size) ).astype( bool )
    part_of_halo[:,self.valid_halo_inds] = part_of_halo_success

    return part_of_halo

  ########################################################################

  def find_mt_containing_halos( self, radial_cut_fraction=1. ):
    '''Find which MergerTrace halos our particles are inside of some radial cut of.

    Args:
      radial_cut_fraction (float): A particle is in a halo if it's in radial_cut_fraction*length_scale from the center.

    Returns:
      part_of_halo (np.array of bools): Shape (n_particles, n_halos). 
        If index [i, j] is True, then particle i is inside radial_cut_fraction*length_scale of the jth halo, defined
          via the MergerTrace ID.
    '''

    # Load up the merger tree data
    self.ahf_reader.get_mtree_halos( self.mtree_halos_index, self.halo_file_tag )

    part_of_halo = []
    for halo_id in self.ahf_reader.mtree_halos.keys():
      mtree_halo = self.ahf_reader.mtree_halos[ halo_id ]

      # Only try to get the data if we're in the range we actually have the halos for.
      above_minimum_snap = self.snum >= mtree_halo.index.min()

      # Only try to get the data if we have the minimum stellar mass
      if above_minimum_snap:
        halo_value = mtree_halo[ self.minimum_criteria ][ self.snum ]/self.min_conversion_factor 
        has_minimum_value = halo_value >= self.minimum_value
      else:
        # If it's not at the point where it can be traced, it definitely doesn't have the minimum stellar mass.
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
        if self.length_scale == 'R_vir':
          length_scale = mtree_halo['Rvir'][ self.snum ]
        elif self.length_scale == 'r_scale':
          # Get the scale radius
          r_vir = mtree_halo['Rvir'][ self.snum ]
          length_scale = r_vir/mtree_halo['cAnalytic'][ self.snum ]
        else:
          raise KeyError( "Unspecified length scale" )
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
  # Radius Finding Routines
  ########################################################################

  def get_cumulative_mass_valid_halos( self ):
    '''Get the cumulative mass, going outwards from the center of each valid *AHF_halos halo.

    Returns:
      cumulative_mass (np.ndarray) : Shape (n_particles, n_halos)
        Index (i,j) is the cumulative mass in halo j at the location of particle i.
    '''

    sorted_inds = np.argsort( self.dist_to_all_valid_halos, axis=0 )

    sorted_mass = self.particle_masses[sorted_inds]

    cumulative_mass = np.cumsum( sorted_mass, axis=0 )

    return cumulative_mass




















