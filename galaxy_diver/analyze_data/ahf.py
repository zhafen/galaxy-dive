#!/usr/bin/env python
'''Tools for reading AHF output files.

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import glob
import numpy as np
import os
import pandas as pd
import string

import galaxy_diver.galaxy_finder.finder as galaxy_finder
import galaxy_diver.read_data.ahf as read_ahf
import galaxy_diver.read_data.metafile as read_metafile

import particle_data

########################################################################
########################################################################

class AHFUpdater( read_ahf.AHFReader ):
  '''Class for updating AHF data (smoothing, adding in additional columns, etc)'''

  ########################################################################
  # Get Data Values
  ########################################################################

  def get_accurate_redshift( self, metafile_dir ):
    '''Get a better values of the redshift than what's stored in the AHF filename, by loading them from an external file.

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
      4,
    )

    # Find the mass radii
    galaxy_finder_kwargs = {
      'particle_positions' : s_data.data['P'].transpose(),
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
  # Alter Data
  ########################################################################

  def smooth_mtree_halos( self, metafile_dir ):
    '''Make Rvir and Mvir monotonically increasing, to help mitigate artifacts in the AHF-calculated merger tree.
    NOTE: This smooths in *physical* coordinates, so it may not be exactly smooth in comoving coordinates.

    Args:
      metafile_dir (str): The directory the snapshot_times are stored in.

    Modifies:
      self.mtree_halos (dict of pd.DataFrames) : Changes self.mtree_halos[halo_id]['Rvir'] and self.mtree_halos[halo_id]['Mvir']
                                                 to be monotonically increasing.
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

  def save_smooth_mtree_halos( self, metafile_dir, index=None, include_concentration=True ):
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
    metafile_dir,
    radii_mass_fractions = None,
    simulation_data_dir = None,
    galaxy_cut = None,
    length_scale = None,
    ):
    '''Save additional columns that would be part of *.AHF_halos files, if that didn't break AHF.

    Args:
      snum (int) :
        Snapshot number to load.

      metafile_dir (str) :
        The directory the metafiles (snapshot_times and used_parameters) are stored in.

      radii_mass_fractions (list of floats, optional) :
        The mass fractions for the characteristic stellar radii to obtain.

      simulation_data_dir (str, optional) :
        Directory containing the simulation data (used for getting the position and masses of the star particles).
        Only necessary if finding mass-based radii.

      galaxy_cut (float):
        galaxy_cut*length_scale is the radius within which the radii will be estimated.
        Only necessary if finding mass-based radii.

      length_scale (str):
        galaxy_cut*length_scale is the radius within which the radii will be estimated.
        Only necessary if finding mass-based radii.
    '''

    print 'Saving *.AHF_halos_add for snum {}'.format( snum )

    # Load the AHF_halos data
    self.get_halos( snum )

    # Create AHF_halos add
    self.ahf_halos_add = pd.DataFrame( {}, index=self.ahf_halos.index )
    self.ahf_halos_add.index.names = ['ID']

    # Get the analytic concentration
    self.ahf_halos_add['cAnalytic'] = self.get_analytic_concentration( metafile_dir, type_of_halo_id='ahf_halos' )

    # Get characteristic radii
    if radii_mass_fractions is not None:
      mass_radii = self.get_mass_radii(
        mass_fractions = radii_mass_fractions,
        simulation_data_dir = simulation_data_dir,
        galaxy_cut = galaxy_cut,
        length_scale = length_scale,
      )

      for i, mass_fraction in enumerate( radii_mass_fractions ):
        label = 'Rmass{}'.format( mass_fraction )
        self.ahf_halos_add[label] = mass_radii[i]

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

