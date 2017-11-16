#!/usr/bin/env python
'''Tools for reading AHF output files.

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import copy
import glob
import numpy as np
import os
import pandas as pd
import string

import galaxy_diver.galaxy_finder.finder as galaxy_finder
import galaxy_diver.read_data.ahf as read_ahf
import galaxy_diver.read_data.metafile as read_metafile
import galaxy_diver.utils.astro as astro_utils
import galaxy_diver.utils.constants as constants
import galaxy_diver.utils.data_constants as data_constants
import galaxy_diver.utils.utilities as utilities

import generic_data
import particle_data

########################################################################
########################################################################

class HaloData( generic_data.GenericData ):

  @utilities.store_parameters
  def __init__( self, data_dir, tag=None, index=None, mt_kwargs={} ):
    '''Constructor for HaloData

    Args:
      data_dir (str) : Directory storing the data.
      tag (str) : If provided, what is an identifying tag for the halo data?
      index (int) : If provided, what is the final snapshot number for the halo data? Necessary for some AHF data.
      mt_kwargs (dict) : When loading merger tree halo files, additional arguments should be passed here.
    '''
    
    self.ahf_reader = read_ahf.AHFReader( data_dir )

    key_parser = HaloKeyParser()

    super( HaloData, self ).__init__( key_parser=key_parser )

  ########################################################################
  # Properties
  ########################################################################

  @property
  def mt_halos( self ):
    '''Attribute for accessing merger tree data.
    '''
    
    if not hasattr( self.ahf_reader, 'mtree_halos' ):

      self.ahf_reader.get_mtree_halos( index=self.index, tag=self.tag, **self.mt_kwargs )

    return self.ahf_reader.mtree_halos

  ########################################################################
  # Data Retrieval
  ########################################################################

  def get_data( self, data_key, snum ):
    '''Get halo data at a specific snapshot.

    Args:
      data_key (str) : What data to get.
      snum (int) : What snapshot to open.

    Returns:
      data (np.ndarray) : Requested data.
    '''

    self.ahf_reader.get_halos( snum )
    self.ahf_reader.get_halos_add( snum )
  
    return self.ahf_reader.ahf_halos[data_key].values

  ########################################################################

  def get_mt_data( self, data_key, mt_halo_id=0, a_power=None, ):
    '''Get halo data for a specific merger tree.

    Args:
      data_key (str) : What data to get.
      mt_halo_id (int) : What merger tree halo ID to select.
      a_power (float) : If given, multiply the result by the scale factor 1/(1 + redshift) to this power.

    Returns:
      mt_data (np.ndarray) : Requested data.
    '''

    mt_data = self.mt_halos[mt_halo_id][data_key].values

    # For converting coordinates
    if a_power is not None:
      mt_data *= ( 1. + self.get_mt_data( 'redshift', mt_halo_id ) )**-a_power

    return mt_data

  ########################################################################

  def get_masked_data( self, *args, **kwargs ):

    return super( HaloData, self ).get_masked_data( mask_multidim_data=False, *args, **kwargs )

########################################################################
########################################################################

class HaloKeyParser( generic_data.DataKeyParser ):

  def get_radius_key( self, multiplier, length_scale ):
    '''Get a key for Halo data, based on a length scale and a multiple of it.

    Args:
      multiplier (float) :
        multiplier*length_scale defines the radius around the center of the halo(s).

      length_scale (str) :
        multiplier*length_scale defines the radius around the center of the halo(s).

    Returns:
      radius_key (str) :
        Combination of length_scale and multiplier.
    '''

    if np.isclose( multiplier, 1.0 ):
      radius_key = length_scale
    else:
      radius_key = '{}{}'.format( multiplier, length_scale )

    return radius_key

  ########################################################################

  def get_enclosed_mass_key( self, ptype, multiplier, length_scale ):
    '''Get a key for Halo data, corresponding to a data column that records an enclosed mass.

    Args:
      ptype (str) :
        The particle type for the enclosed mass.

      multiplier (float) :
        multiplier*length_scale defines the radius around the center of the halo within which to get the mass.

      length_scale (str) :
        multiplier*length_scale defines the radius around the center of the halo within which to get the mass.

    Returns:
      enclosed_mass_key (str)
    '''

    return 'M{}({})'.format( ptype, self.get_radius_key( multiplier, length_scale ) )

  ########################################################################

  def get_average_quantity_key( self, data_key, ptype, multiplier, length_scale ):
    '''Get a key for Halo data, corresponding to a data column that records the average quantity inside a galaxy.

    Args:
      data_key (str) :
        What the enclosed quantity is.

      ptype (str) :
        The particle type for the enclosed mass.

      multiplier (float) :
        multiplier*length_scale defines the radius around the center of the halo within which to get the mass.

      length_scale (str) :
        multiplier*length_scale defines the radius around the center of the halo within which to get the mass.

    Returns:
      average_quantity_key (str)
    '''

    return '{}{}({})'.format( data_key, ptype, self.get_radius_key( multiplier, length_scale ) )


  ########################################################################

  def get_velocity_at_radius_key( self, velocity_key, multiplier, length_scale ):
    '''Get a key for Halo data, corresponding to a data column that records the velocity at at a specified radius

      velocity_key (str) :
        What velocity to get.

      multiplier (float) :
        multiplier*length_scale defines the radius around the center of the halo within which to get the mass.

      length_scale (str) :
        multiplier*length_scale defines the radius around the center of the halo within which to get the mass.

    Returns:
      velocity_at_radius_key (str)
    '''

    return '{}({})'.format( velocity_key, self.get_radius_key( multiplier, length_scale ) )

########################################################################
########################################################################

class HaloUpdater( read_ahf.AHFReader ):
  '''Class for updating Halo data (smoothing, adding in additional columns, etc)'''

  def __init__( self, *args, **kwargs ):

    self.key_parser = HaloKeyParser()

    super( HaloUpdater, self ).__init__( *args, **kwargs )

  ########################################################################
  # Get Data Values
  ########################################################################

  def get_accurate_redshift( self, metafile_dir ):
    '''Get a better values of the redshift than what's stored in the Halo filename, by loading them from an external file.

    Args:
      metafile_dir (str): The directory the snapshot_times are stored in.

    Modifies:
      self.mtree_halos (dict of pd.DataFrames): Updates the redshift column
    '''

    # Get the redshift data out
    metafile_reader = read_metafile.MetafileReader( metafile_dir )
    metafile_reader.get_snapshot_times()

    # Replace the old data
    for halo_id in self.mtree_halos.keys():

      mtree_halo = self.mtree_halos[ halo_id ]

      # Read and replace the new redshift
      new_redshift = metafile_reader.snapshot_times['redshift'][ mtree_halo.index ]
      mtree_halo['redshift'] = new_redshift

  ########################################################################

  def get_analytic_concentration( self, metafile_dir, type_of_halo_id='merger_tree' ):
    '''Get analytic values for the halo concentration, using colossus, Benedikt Diemer's cosmology code
    ( https://bitbucket.org/bdiemer/colossus ; http://www.benediktdiemer.com/code/colossus/ ).

    Assumptions:
      - We're using the default formula of Diemer&Kravtstov15
      - We're using the Bryan&Norman1998 version of the virial radius.

    Args:
      metafile_dir (str): The directory the snapshot_times are stored in.
      type_of_halo_id (str): 'merger_tree' if the halo id is a merger tree halo id.
                             'ahf_halos' if the halo id is a *.AHF_halos halo id.

    Returns
      c_vir (np.array of floats): The concentration, defined as R_vir/r_scale.
    '''

    # Include imports here, because this function may not in general work if colossus is not available,
    # and the rest of the module should still be made useable
    # There may be some warnings here about the version of scipy colossus uses, as opposed to the version galaxy_diver uses
    import colossus.cosmology.cosmology as co_cosmology
    import colossus.halo.concentration as co_concentration

    # Get simulation parameters, for use in creating a cosmology
    metafile_reader = read_metafile.MetafileReader( metafile_dir )
    metafile_reader.get_used_parameters()

    # Setup the cosmology used by the simulations
    sim_cosmo = {
      'flat': True,
      'H0' : float( metafile_reader.used_parameters['HubbleParam'] )*100.,
      'Om0' : float( metafile_reader.used_parameters['Omega0'] ),
      'Ob0' : float( metafile_reader.used_parameters['OmegaBaryon'] ),
      'sigma8' : co_cosmology.cosmologies['WMAP9']['sigma8'], # Use WMAP9 for values we don't store in our simulations explicitly.
      'ns' : co_cosmology.cosmologies['WMAP9']['ns'], # Use WMAP9 for values we don't store in our simulations explicitly.
    }
    cosmo = co_cosmology.setCosmology( 'sim_cosmo', sim_cosmo )

    if type_of_halo_id == 'merger_tree':

      # Loop over all mt halos
      for halo_id in self.mtree_halos.keys():

        # Load the data
        mtree_halo = self.mtree_halos[ halo_id ]

        # Get the concentration out
        c_vir = []
        for m_vir, z in zip( mtree_halo['Mvir'], mtree_halo['redshift'] ):
          c = co_concentration.concentration( m_vir, 'vir', z, model='diemer15', statistic='median')
          c_vir.append( c )

        # Turn the concentration into an array
        c_vir = np.array( c_vir )

        # Save the concentration
        mtree_halo['cAnalytic'] = c_vir

    elif type_of_halo_id == 'ahf_halos':

      # Get the redshift for the halo file.
      metafile_reader.get_snapshot_times()
      redshift = metafile_reader.snapshot_times['redshift'][self.ahf_halos_snum]

      # Get the concentration
      c = co_concentration.concentration( self.ahf_halos['Mvir'], 'vir', redshift, model='diemer15', statistic='median')

      return c

  ########################################################################

  def get_mass_radii( self,
    mass_fractions,
    simulation_data_dir,
    galaxy_cut,
    length_scale,
    ):
    '''Get radii that enclose a fraction (mass_fractions[i]) of a halo's stellar mass.

    Args:
      mass_fractions (list of floats) :
        Relevant mass fractions.

      simulation_data_dir (str) :
        Directory containing the raw particle data.

      galaxy_cut (float) :
        galaxy_cut*length_scale defines the radius around the center of the halo to look for stars.

      length_scale (str) :
        galaxy_cut*length_scale defines the radius around the center of the halo to look for stars.

    Returns:
      mass_radii (list of np.ndarrays) :
        If M_sum_j = all mass inside galaxy_cut*length_scale for halo j, then mass_radii[i][j] is the radius that
        contains a fraction mass_fractions[i] of M_sum_j.
    '''
        
    # Load the simulation data
    s_data = particle_data.ParticleData(
      simulation_data_dir,
      self.ahf_halos_snum,
      ptype = data_constants.PTYPES['star'],
    )

    try:
      particle_positions = s_data.data['P'].transpose()
    # Case where there are no star particles at this redshift.
    except KeyError:
      return [ np.array( [ np.nan, ]*self.ahf_halos.index.size ), ]*len( mass_fractions )

    # Find the mass radii
    galaxy_finder_kwargs = {
      'particle_positions' : particle_positions,
      'particle_masses' : s_data.data['M'],
      'snum' : self.ahf_halos_snum,
      'redshift' : s_data.redshift,
      'hubble' : s_data.data_attrs['hubble'],
      'galaxy_cut' : galaxy_cut,
      'length_scale' : length_scale,
      'ahf_reader' : self,
    }
    gal_finder = galaxy_finder.GalaxyFinder( **galaxy_finder_kwargs )
    
    mass_radii = [ gal_finder.get_mass_radius( mass_fraction ) for mass_fraction in mass_fractions ]
      
    return mass_radii

  ########################################################################

  def get_enclosed_mass( self,
    simulation_data_dir,
    ptype,
    galaxy_cut,
    length_scale,
    ):
    '''Get the mass inside galaxy_cut*length_scale for each Halo halo.

    Args:
      simulation_data_dir (str) :
        Directory containing the raw particle data.

      ptype (str) :
        What particle type to get the mass for.

      galaxy_cut (float) :
        galaxy_cut*length_scale defines the radius around the center of the halo within which to get the mass.

      length_scale (str) :
        galaxy_cut*length_scale defines the radius around the center of the halo within which to get the mass.

    Returns:
      mass_inside_all_halos (np.ndarray) :
        mass_inside_all_halos[i] is the mass of particle type ptype inside galaxy_cut*length scale around a galaxy.
    '''

    # Load the simulation data
    s_data = particle_data.ParticleData(
      simulation_data_dir,
      self.ahf_halos_snum,
      data_constants.PTYPES[ptype],
    )

    try:
      particle_positions = s_data.data['P'].transpose()
    # Case where there are no star particles at this redshift.
    except KeyError:
      return np.array( [ 0., ]*self.ahf_halos.index.size )

    # Find the mass radii
    galaxy_finder_kwargs = {
      'particle_positions' : particle_positions,
      'particle_masses' : s_data.data['M']*constants.UNITMASS_IN_MSUN,
      'snum' : self.ahf_halos_snum,
      'redshift' : s_data.redshift,
      'hubble' : s_data.data_attrs['hubble'],
      'galaxy_cut' : galaxy_cut,
      'length_scale' : length_scale,
      'ahf_reader' : self,
    }
    gal_finder = galaxy_finder.GalaxyFinder( **galaxy_finder_kwargs )

    mass_inside_all_halos = gal_finder.mass_inside_all_halos
    
    # Make sure to put hubble constant back in so we have consistent units.
    mass_inside_all_halos *= s_data.data_attrs['hubble']

    return mass_inside_all_halos

  ########################################################################

  def get_average_quantity_inside_galaxy( self,
    data_key,
    simulation_data_dir,
    ptype,
    galaxy_cut,
    length_scale,
    weight_data_key = 'M',
    fill_value = np.nan,
    ):
    '''Get the mass inside galaxy_cut*length_scale for each Halo halo.

    Args:
      data_key (str) :
        Data key for the quantity to get the average of.

      simulation_data_dir (str) :
        Directory containing the raw particle data.

      ptype (str) :
        What particle type to get the mass for.

      galaxy_cut (float) :
        galaxy_cut*length_scale defines the radius around the center of the halo within which to get the mass.

      length_scale (str) :
        galaxy_cut*length_scale defines the radius around the center of the halo within which to get the mass.

      weight_data_key (str) :
        Data key for the weight to use when averaging.

      fill_value (float) :
        What value to use when the average quantity inside the galaxy is not resolved.

    Returns:
      average_quantity_inside_galaxy (np.ndarray) :
        average_quantity_inside_galaxy[i] is the average value of the requested quantity for particle type ptype
        inside galaxy_cut*length scale around a galaxy.
    '''

    # Load the simulation data
    s_data = particle_data.ParticleData(
      simulation_data_dir,
      self.ahf_halos_snum,
      data_constants.PTYPES[ptype],

      # The following values need to be set, because they come into play when a galaxy is centered on halo finder
      # data. That's obviously not the case here...
      centered = True,
      vel_centered = True,
      hubble_corrected = True,
    )

    try:
      particle_positions = s_data.data['P'].transpose()
    # Case where there are no particles of the given ptype at this redshift.
    except KeyError:
      return np.array( [ fill_value, ]*self.ahf_halos.index.size )

    # Find the mass radii
    galaxy_finder_kwargs = {
      'particle_positions' : particle_positions,
      'snum' : self.ahf_halos_snum,
      'redshift' : s_data.redshift,
      'hubble' : s_data.data_attrs['hubble'],
      'galaxy_cut' : galaxy_cut,
      'length_scale' : length_scale,
      'ahf_reader' : self,
    }
    gal_finder = galaxy_finder.GalaxyFinder( low_memory_mode=False, **galaxy_finder_kwargs )

    average_quantity_inside_galaxy = gal_finder.weighted_summed_quantity_inside_galaxy(
      s_data.get_data( data_key ),
      s_data.get_data( weight_data_key ),
      fill_value,
    )

    return average_quantity_inside_galaxy

  ########################################################################
  
  def get_circular_velocity( self,
    galaxy_cut,
    length_scale,
    metafile_dir,
    ptypes = data_constants.STANDARD_PTYPES,
    ):
    '''Get the circular velocity at galaxy_cut*length_scale.

    Args:
      galaxy_cut (float) :
        galaxy_cut*length_scale defines the radius around the center of the halo within which to get the mass.

      length_scale (str) :
        galaxy_cut*length_scale defines the radius around the center of the halo within which to get the mass.

      metafile_dir (str) :
        Directory containing metafile data, for getting out the redshift given a snapshot.

      ptypes (list of strs) :
        Particle types to count the mass inside the halo of.

    Returns:
      v_circ (np.ndarray) 
        Circular velocity at galaxy_cut*length_scale using mass from the given ptypes.
    '''

    # Get the redshift, for converting the radius to pkpc/h.
    metafile_reader = read_metafile.MetafileReader( metafile_dir )
    metafile_reader.get_snapshot_times()
    redshift = metafile_reader.snapshot_times['redshift'][self.ahf_halos_snum]

    # Get the radius in pkpc/h
    try:
      radius = galaxy_cut*self.ahf_halos[length_scale]
    except KeyError:
      radius = galaxy_cut*self.ahf_halos_add[length_scale]
    radius /= ( 1. + redshift )

    # Get the mass in Msun/h
    masses = []
    for ptype in ptypes:
      mass_key = self.key_parser.get_enclosed_mass_key( ptype, galaxy_cut, length_scale )
      try:
        ptype_mass = self.ahf_halos_add[mass_key]
      except:
        ptype_mass = self.ahf_halos[mass_key]
      masses.append( ptype_mass )
    mass = np.array( masses ).sum( axis=0 )

    # Now get the circular velocity out
    # (note that we don't need to bother with converting out the 1/h's, because in this particular case they'll cancel)
    v_circ = astro_utils.circular_velocity( radius, mass )

    return v_circ

  ########################################################################
  # Alter Data
  ########################################################################

  def smooth_mtree_halos( self, metafile_dir ):
    '''Make Rvir and Mvir monotonically increasing, to help mitigate artifacts in the Halo-calculated merger tree.
    NOTE: This smooths in *physical* coordinates, so it may not be exactly smooth in comoving coordinates.

    Args:
      metafile_dir (str) :
        The directory the snapshot_times are stored in.

    Modifies:
      self.mtree_halos (dict of pd.DataFrames) :
        Changes self.mtree_halos[halo_id]['Rvir'] and self.mtree_halos[halo_id]['Mvir'] to be monotonically increasing.
    '''

    # We need to get an accurate redshift in order to smooth properly
    self.get_accurate_redshift( metafile_dir )

    for halo_id in self.mtree_halos.keys():

      # Load the data
      mtree_halo = self.mtree_halos[ halo_id ]

      # Convert into physical coords for smoothing (we'll still leave the 1/h in place)
      r_vir_phys = mtree_halo['Rvir']/( 1. + mtree_halo['redshift'] )

      # Smooth r_vir
      r_vir_phys_smooth = np.maximum.accumulate( r_vir_phys[::-1] )[::-1]

      # Convert back into comoving and save
      mtree_halo['Rvir'] = r_vir_phys_smooth*( 1. + mtree_halo['redshift'] )

      # Smooth Mvir
      mtree_halo['Mvir'] = np.maximum.accumulate( mtree_halo['Mvir'][::-1] )[::-1]

  ########################################################################

  def include_ahf_halos_to_mtree_halos( self ):
    '''While most of the halofile data are contained in *.AHF_halos files, some quantities are stored in
    *.AHF_halos files. These are usually computed manually, external to what's inherent in AHF. This routine adds
    on the the information from these files to the loaded merger tree data (which don't usually include them, because
    they're not inherent to AHF.

    Modifies:
      self.mtree_halos (dict of pd.DataFrames) :
        Adds additional columns contained in *.AHF_halos_add files.
    '''

    for mtree_id, mtree_halo in self.mtree_halos.items():

      print( "Looking at merger tree ID {}".format( mtree_id ) )

      halo_ids = mtree_halo['ID'].values
      snums = mtree_halo.index

      ahf_frames = []
      for snum, halo_id in zip( snums, halo_ids, ):

        print( "Getting data for snapshot {}".format( snum ) )

        self.get_halos( snum )
        self.get_halos_add( snum )

        # Get the columns we want to add on.
        halofile_columns = set( self.ahf_halos.columns )
        mtree_columns = set( mtree_halo.columns )
        columns_to_add = list( halofile_columns - mtree_columns )
        columns_to_add.sort()

        # Now get the values to add
        full_ahf_row = self.ahf_halos.loc[halo_id:halo_id] 
        ahf_row = full_ahf_row[columns_to_add]

        # Check for edge case, where there isn't an AHF row with specified halo number
        if ahf_row.size == 0:
          
          # Borrow a previous row for formatting
          ahf_row = copy.copy( ahf_frames[-1] )

          # Drop the previous row
          ahf_row = ahf_row.drop( ahf_row.index[0] )

          # Turn all the values to np.nan, so we know that they're invalid, but the format still works.
          [ ahf_row.set_value( halo_id, column_name, np.nan ) for column_name in columns_to_add ]

        ahf_frames.append( ahf_row )

      custom_mtree_halo = pd.concat( ahf_frames )

      # Add in the snapshots, and use them as the index
      custom_mtree_halo['snum'] = snums
      custom_mtree_halo = custom_mtree_halo.set_index( 'snum', )

      # Now merge onto the mtree_halo DataFram
      self.mtree_halos[mtree_id] = pd.concat( [ mtree_halo, custom_mtree_halo, ], axis=1 )

  ########################################################################
  # Save Data
  ########################################################################

  def save_mtree_halos( self, tag ):
    '''Save loaded mergertree halo files in a csv file.

    Args:
      tag (str) : If the previous file was for example '/path/to/file/halo_00000.dat',
                  the new file will be '/path/to/file/halo_00000_{}.dat'.format( tag )
    '''

    for halo_id in self.mtree_halos.keys():

      # Load the data
      mtree_halo = self.mtree_halos[ halo_id ]
      halo_filepath = self.mtree_halo_filepaths[ halo_id ]

      # Create the new filename
      filepath_base, file_ext = os.path.splitext( halo_filepath )
      save_filepath = '{}_{}{}'.format( filepath_base, tag, file_ext )

      mtree_halo.to_csv( save_filepath, sep='\t' )

  ########################################################################

  def save_smooth_mtree_halos( self,
    metafile_dir,
    index = None,
    include_ahf_halos_add = True,
    include_concentration = False,
    ):
    '''Load halo files, smooth them, and save as a new file e.g., halo_00000_smooth.dat

    Args:
      metafile_dir (str) :
        The directory the metafiles (snapshot_times and used_parameters) are stored in.

      index (str or int) :
        What type of index to use. Defaults to None, which raises an exception. You *must* choose an
        index, to avoid easy mistakes. See get_mtree_halos() for a full description.

      include_concentration (bool):
        Whether or not to add an additional column that gives an analytic value for the
        halo concentration.
    '''
    
    # Load the data
    self.get_mtree_halos( index=index )

    # Include data stored in *AHF_halos_add files.
    if include_ahf_halos_add:
      self.include_ahf_halos_to_mtree_halos()

    # Smooth the halos
    self.smooth_mtree_halos( metafile_dir )

    # Include the concentration, if chosen.
    if include_concentration:
      self.get_analytic_concentration( metafile_dir )

    # Save the halos
    self.save_mtree_halos( 'smooth' )

  ########################################################################

  def save_custom_mtree_halos( self, snums, halo_ids, metafile_dir, ):
    '''Save a custom merger tree.

    Args:
      snums (array-like or int) :
        What snapshots to generate the custom merger tree for.
        If a single integer, then snums will start at that integer and count backwards by single snapshots for the
        length of halo_ids

      halo_ids (array-like) :
        halo_ids[i] is the AHF_halos halo ID for the merger tree halo at snums[i].

      metafile_dir (str) :
        Directory for the metafile (used to get simulation redshift).

    Modifies:
      self.sdir/halo_00000_custom.dat (text file) : Saves the custom merger tree at this location.
    '''

    if isinstance( snums, int ):
      snums = np.arange( snums, snums - len( halo_ids ), -1 )

    # Concatenate the data
    ahf_frames = []
    for snum, halo_id in zip( snums, halo_ids, ):

      print( "Getting data for snapshot {}".format( snum ) )

      self.get_halos( snum )
      self.get_halos_add( snum )

      ahf_frames.append( self.ahf_halos.loc[halo_id:halo_id] )

    custom_mtree_halo = pd.concat( ahf_frames )

    # Make sure to store the IDs too
    custom_mtree_halo['ID'] = halo_ids

    # Add in the snapshots, and use them as the index
    custom_mtree_halo['snum'] = snums
    custom_mtree_halo = custom_mtree_halo.set_index( 'snum', )

    # Get and save the redshift
    metafile_reader = read_metafile.MetafileReader( metafile_dir )
    metafile_reader.get_snapshot_times()
    custom_mtree_halo['redshift'] = metafile_reader.snapshot_times['redshift'][snums]

    # Save the data
    save_filepath = os.path.join( self.sdir, 'halo_00000_custom.dat' )
    custom_mtree_halo.to_csv( save_filepath, sep='\t' )

  ########################################################################

  def save_ahf_halos_add( self,
    snum,
    include_analytic_concentration = True,
    include_mass_radii = True,
    include_enclosed_mass = True,
    include_average_quantity_inside_galaxy = False,
    include_v_circ = True,
    metafile_dir = None,
    simulation_data_dir = None,
    mass_radii_kwargs = {
      'mass_fractions' : [ 0.5, 0.75, 0.9, ],
      'galaxy_cut' : 0.15,
      'length_scale' : 'Rvir',
    },
    enclosed_mass_ptypes = data_constants.STANDARD_PTYPES,
    enclosed_mass_kwargs = {
      'galaxy_cut' : 2.0,
      'length_scale' : 'Rstar0.5',
    },
    average_quantity_data_keys = [ 'Vx', 'Vy', 'Vz', ],
    average_quantity_inside_galaxy_kwargs = {
      'ptype' : 'star',
      'galaxy_cut' : 2.0,
      'length_scale' : 'Rstar0.5',
    },
    v_circ_kwargs = {
      'galaxy_cut' : 2.0,
      'length_scale' : 'Rstar0.5',
    },
    verbose = False,
    ):
    '''Save additional columns that would be part of *.AHF_halos files, if that didn't break AHF.

    Args:
      snum (int) :
        Snapshot number to load.

      include_analytic_concentration (bool) :
        Include analytic concentration as one of the columns?

      include_mass_radii (bool) :
        Include radius that include some fraction of a particle's mass as one of the columns?

      include_enclosed_mass (bool) :
        Include the mass enclosed in some specified radii as one of the columns?

      include_average_quantity_inside_galaxy (bool) :
        Include the average value inside each galaxy for the quantities listed in average_quantity_data_keys?

      include_v_circ (bool) :
        Include the circular mass at some specified radii as one of the columns?

      metafile_dir (str) :
        The directory the metafiles (snapshot_times and used_parameters) are stored in.

      simulation_data_dir (str) :
        Directory containing the simulation data (used for getting the position and masses of the star particles).

      mass_radii_kwargs (dict) :
        Keyword args for self.get_mass_radii()

      enclosed_mass_ptypes (list of strs) :
        Particle types to get the mass inside a radii of.

      enclosed_mass_kwargs (dict) :
        Keyword args for self.get_enclosed_mass()

      average_quantity_data_keys (list of strs) :
        What data keys (to be passed to a standard ParticleData.get_data() function) to get the average quantity for?

      average_quantity_kwargs (dict) :
        Keyword args for self.get_average_quantity_inside_galaxy()

      v_circ_kwargs (dict) :
        Keyword args for self.get_circular_velocity()

      verbose (bool) :
        If True, print out additional information about how the steps are progressing.
    '''

    print 'Saving *.AHF_halos_add for snum {}'.format( snum )

    # Load the AHF_halos data
    self.get_halos( snum )

    # Figure out if there are any valid halos at this redshift if not, then a *lot* can be skipped.
    # TODO: Don't hard-code this in....
    valid_halos = self.ahf_halos['n_star'] >= 10
    no_valid_halos = valid_halos.sum() == 0
    blank_array = np.array( [ np.nan, ]*self.ahf_halos.index.size )

    # Create AHF_halos add
    self.ahf_halos_add = pd.DataFrame( {}, index=self.ahf_halos.index )
    self.ahf_halos_add.index.names = ['ID']

    # Get the analytic concentration
    if include_analytic_concentration:
      if verbose:
        print( "Including Analytic Concentration..." )
      self.ahf_halos_add['cAnalytic'] = self.get_analytic_concentration( metafile_dir, type_of_halo_id='ahf_halos' )

    # Get characteristic radii
    if include_mass_radii:
      if verbose:
        print( "Including Mass Radii..." )

      if no_valid_halos:
        mass_radii = [ blank_array, ]*len( mass_radii_kwargs['mass_fractions'] )
      else:
        mass_radii = self.get_mass_radii( simulation_data_dir = simulation_data_dir, **mass_radii_kwargs )

      for i, mass_fraction in enumerate( mass_radii_kwargs['mass_fractions'] ):
        label = 'Rstar{}'.format( mass_fraction )
        self.ahf_halos_add[label] = mass_radii[i]

    # Get mass enclosed in a particular radius
    if include_enclosed_mass:
      if verbose:
        print( "Including Enclosed Mass..." )
      for i, ptype in enumerate( enclosed_mass_ptypes ):

        if no_valid_halos:
          halo_masses = blank_array
        else:
          halo_masses = self.get_enclosed_mass( simulation_data_dir, ptype, **enclosed_mass_kwargs )

        label = self.key_parser.get_enclosed_mass_key( ptype, enclosed_mass_kwargs['galaxy_cut'], \
                                                       enclosed_mass_kwargs['length_scale'], )
        self.ahf_halos_add[label] = halo_masses

    # Get average quantity inside each galaxy (for halos that have galaxies)
    if include_average_quantity_inside_galaxy:
      if verbose:
        print( "Including Average Quantities..." )

      for i, data_key in enumerate( average_quantity_data_keys ):

        if verbose:
          print( "  Finding average {}...".format( data_key ) )

        if no_valid_halos:
          average_quantity = blank_array
        else:
          average_quantity = self.get_average_quantity_inside_galaxy(
            data_key,
            simulation_data_dir,
            **average_quantity_inside_galaxy_kwargs
          )

        label = self.key_parser.get_average_quantity_key(
          data_key,
          average_quantity_inside_galaxy_kwargs['ptype'],
          average_quantity_inside_galaxy_kwargs['galaxy_cut'],
          average_quantity_inside_galaxy_kwargs['length_scale'],
        )

        self.ahf_halos_add[label] = average_quantity

    # Get circular velocity at a particular radius
    if include_v_circ:
      if verbose:
        print( "Including Circular Velocity..." )
      v_circ = self.get_circular_velocity( metafile_dir=metafile_dir, **v_circ_kwargs )

      label = self.key_parser.get_velocity_at_radius_key(
        'Vc',
        v_circ_kwargs['galaxy_cut'],
        v_circ_kwargs['length_scale']
      ) 

      self.ahf_halos_add[label] = v_circ

    # Save AHF_halos add
    save_filepath = '{}_add'.format( self.ahf_halos_path )
    self.ahf_halos_add.to_csv( save_filepath, sep='\t' )

  ########################################################################

  def save_multiple_ahf_halos_adds( self, metafile_dir, snum_start, snum_end, snum_step ):
    '''Save additional columns that would be part of *.AHF_halos files, if that didn't break AHF.
    Do this for every *.AHF_halos file in self.sdir.

    Args:
      metafile_dir (str): The directory the metafiles (snapshot_times and used_parameters) are stored in.
      snum_start (int): Starting snapshot.
      snum_end (int): Ending snapshot.
      snum_step (int): Step between snapshots.
    '''

    # Save the halos
    for snum in range( snum_start, snum_end+snum_step, snum_step):

      # Save the data
      self.save_ahf_halos_add( snum, metafile_dir )

