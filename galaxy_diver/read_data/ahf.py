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

import galaxy_diver.read_data.metafile as read_metafile

########################################################################
########################################################################

class AHFReader( object ):
  '''Read AHF data.
  Note! All positions are in comoving coordinates, and everything has 1/h's sprinkled throughout.
  '''

  def __init__( self, sdir ):
    '''Initializes.

    Args:
      sdir (str): Simulation directory to load the AHF data from.
    '''

    self.sdir = sdir

  ########################################################################

  def get_mtree_halos( self, index=None, tag=None ):
    '''Get halo files (e.g. halo_00000.dat) in a dictionary of pandas DataFrames.

    Args:
      index (str or int) : What type of index to use. Defaults to None, which raises an exception. You *must* choose an index, to avoid easy mistakes. Options are...
        'snum' : Indexes by snapshot number, starting at 600 and counting down. Only viable with snapshot steps of 1!!
                 Identical to passing the integer 600. (See below.)
        'range' : Index by an increasing range.
        int : If an integer, then this integer should be the final snapshot number for the simulation.
              In this case, indexes by snapshot number, starting at the final snapshot number and counting down. Only viable with snapshot steps of 1!!
      tag (str) : Additional identifying tag for the files, e.g. 'smooth', means this function will look for 'halo_00000_smooth.dat', etc.

    Modifies:
      self.mtree_halos (dict of pd.DataFrames): DataFrames containing the requested data. The key for a given dataframe is that dataframe's Merger Tree Halo ID
      self.index (str): The users value for the index.
    '''

    def get_halo_filepaths( unexpanded_filename ):
      '''Function for getting a list of filepaths'''
      filepath_unexpanded = os.path.join( self.sdir, unexpanded_filename )
      halo_filepaths = glob.glob( filepath_unexpanded )
      return set( halo_filepaths )

    # Get the filename to search for
    if tag is not None:
      ahf_filename = 'halo_*_{}.dat'.format( tag )
      halo_filepaths = get_halo_filepaths( ahf_filename )

    else:
      ahf_filename = 'halo_*.dat'
      halo_filepaths = get_halo_filepaths( ahf_filename )

      # Find all files that are modified.
      ahf_modified_filename = 'halo_*_*.dat'
      halo_modified_filepaths = get_halo_filepaths( ahf_modified_filename )

      # Remove all the modified filepaths from the search list.
      halo_filepaths -= halo_modified_filepaths

    # Raise an exception if there are no files to load
    if len( halo_filepaths ) == 0:
      raise KeyError( 'No files to load in {}'.format( self.sdir ) )

    # Set up the data storage
    self.mtree_halos = {}
    self.mtree_halo_filepaths = {}

    # Loop over each file and load it
    for halo_filepath in halo_filepaths:

      # Load the data
      mtree_halo = pd.read_csv( halo_filepath, sep='\t', )

      # Extra tweaking to read the default AHF file format
      if tag is None:
        # Delete a column that shows up as a result of formatting
        del mtree_halo[ 'Unnamed: 93' ]

        # Remove the annoying parenthesis at the end of each label.
        mtree_halo.columns = [ string.split( label, '(' )[0] for label in list( mtree_halo ) ]

        # Remove the pound sign in front of the first column's name
        mtree_halo = mtree_halo.rename( columns = {'#redshift':'redshift', ' ID':'ID'} )

      # Get a good key
      base_filename = os.path.basename( halo_filepath )
      halo_num_str = base_filename[5:]
      if tag is not None:
        halo_num_str = string.split( halo_num_str, '_' )[0]
      halo_num = int( string.split( halo_num_str, '.' )[0] )

      if index == 'range':
        pass
      elif (index == 'snum') or (type(index) == int):
        if index == 'snum':
          final_snapshot_number = 600
        else:
          final_snapshot_number = index
        # Set the index, assuming we have steps of one snapshot
        n_rows = mtree_halo.shape[0]
        mtree_halo['snum'] = range( final_snapshot_number, final_snapshot_number - n_rows, -1)
        mtree_halo = mtree_halo.set_index( 'snum', )
      else:
        raise Exception( "index type not selected" )

      # Store the data
      self.mtree_halos[ halo_num ] = mtree_halo
      self.mtree_halo_filepaths[ halo_num ] = halo_filepath

    # Save the index and tag as an attribute
    self.index = index
    self.tag = tag

  ########################################################################

  def get_halos( self, snum ):
    '''Get *.AHF_halos file for a particular snapshot.

    Args:
      snum (int): Snapshot number to load.

    Modifies:
      self.ahf_halos (pd.DataFrame): Dataframe containing the requested data.
    '''

    # Load the data
    self.ahf_halos_path = self.get_filepath( snum, 'AHF_halos' )
    self.ahf_halos = pd.read_csv( self.ahf_halos_path, sep='\t', index_col=0 )

    # Delete a column that shows up as a result of formatting
    del self.ahf_halos[ 'Unnamed: 92' ]

    # Remove the annoying parenthesis at the end of each label.
    self.ahf_halos.columns = [ string.split( label, '(' )[0] for label in list( self.ahf_halos ) ]

    # Rename the index to a more suitable name, without the '#' and the (1)
    self.ahf_halos.index.names = ['ID']

    # Save the snapshot number of the ahf halos file.
    self.ahf_halos_snum = snum

  ########################################################################

  def get_halos_add( self, snum ):
    '''Get *.AHF_halos_add file for a particular snapshot.

    Args:
      snum (int): Snapshot number to load.

    Modifies:
      self.ahf_halos_add (pd.DataFrame): Dataframe containing the requested data.
    '''

    # Load the data
    self.ahf_halos_path = self.get_filepath( snum, 'AHF_halos_add' )
    self.ahf_halos = pd.read_csv( self.ahf_halos_path, sep='\t', index_col=0 )

  ########################################################################

  def get_mtree_idx( self, snum ):
    '''Get *.AHF_mtree_idx file for a particular snapshot.

    Args:
      snum (int): Snapshot number to load.

    Modifies:
      self.ahf_mtree_idx (pd.DataFrame): Dataframe containing the requested data.
    '''

    # If the data's already loaded, don't load it again.
    if hasattr( self, 'ahf_mtree_idx' ):
      return self.ahf_mtree_idx

    # Load the data
    ahf_mtree_idx_path = self.get_filepath( snum, 'AHF_mtree_idx' )
    self.ahf_mtree_idx = pd.read_csv( ahf_mtree_idx_path, delim_whitespace=True, names=['HaloID(1)', 'HaloID(2)'], skiprows=1  )

  ########################################################################

  def get_filepath( self, snum, ahf_file_type ):
    '''Get the filepath for a specified type of AHF file.

    Args:
      snum (int): Snapshot number to load.
      ahf_file_type (str): Can be AHF_halos or AHF_mtree_idx.

    Returns:
      ahf_filepath (str): The filepath to the specified file.
    '''

    # Load the data
    ahf_filename = 'snap{:03d}Rpep..z*.*.{}'.format( snum, ahf_file_type )
    ahf_filepath_unexpanded = os.path.join( self.sdir, ahf_filename )
    possible_filepaths = glob.glob( ahf_filepath_unexpanded )
    if len( possible_filepaths ) == 0:
      raise KeyError( 'No files to load in {}'.format( self.sdir ) )
    elif len( possible_filepaths ) > 1:
      raise Exception( 'Multiple possible *.{} files to load'.format( ahf_file_type ) )
    ahf_filepath = possible_filepaths[0]

    return ahf_filepath

  ########################################################################

  def get_mtree_halo_quantity( self, quantity, indice, index=None, tag=None ):
    '''Get a desired quantity for all halos at a particular snapshot.

    Args:
      quantity (str): mtree_halo key to load in the dataset
      indice (int): Indice of the quantity to load, as indicated by the index.
      index (str or int) : What type of index to use. Defaults to None, which raises an exception. You *must* choose an index, to avoid easy mistakes.
                           See get_mtree_halos() for a full description.
      tag (str) : Additional identifying tag for the files, e.g. 'smooth', means this function will look for 'halo_00000_smooth.dat', etc.

    Returns:
      mtree_halo_quantity (np.array): The ith index is the requested quantity for ith MT halo.
    '''

    # Load the data if it's not already loaded.
    if not hasattr( self, 'mtree_halos' ):
      self.get_mtree_halos( index=index, tag=tag )
    else:
      assert index == self.index
      assert tag == self.tag

    mtree_halo_quantity = [] 
    for halo_id in sorted( self.mtree_halos.keys() ):
      
      try:
        mtree_halo_quantity.append( self.mtree_halos[ halo_id ][ quantity ][ indice ] )

      # When we're past the point of galaxies being identified (i.e. high redshift, galaxies aren't formed yet), set the values by hand.
      except KeyError:

        if quantity == 'Mvir':
          mtree_halo_quantity.append( 0. )

        else:
          raise Exception( "Value of {} not specified before galaxies form".format( quantity ) )

    return np.array( mtree_halo_quantity )

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

  def get_analytic_concentration( self, metafile_dir, type_of_halo_id='merger_tree' ):
    '''Get analytic values for the halo concentration, using colossus, Benedikt Diemer's cosmology code.
    ( https://bitbucket.org/bdiemer/colossus ; http://www.benediktdiemer.com/code/colossus/ )

    Args:
      metafile_dir (str): The directory the snapshot_times are stored in.
      type_of_halo_id (str): 'merger_tree' if the halo id is a merger tree halo id.
                             'ahf_halos' if the halo id is a *.AHF_halos halo id.

    Assumptions:
      - We're using the default formula of Diemer&Kravtstov15
      - We're using the Bryan&Norman1998 version of the virial radius.

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

  def get_pos_or_vel( self, pos_or_vel, halo_id, inds, type_of_halo_id='merger_tree' ):
    '''Get the position or velocity of a mt halo (three dimensional).

    Args:
      pos_or_vel (str): Get position ('pos') or velocity ('vel').
      halo_id (int): Merger tree halo ID for the position or velocity you want.
      inds (int or np.array of ints): Indices you want the position or velocity for.
                                      If type_of_halo_id == 'merger_tree', uses same index as mtree_halos.
                                      Elif type_of_halo_id == 'ahf_halos', can only be a single int,
                                      which should be the snapshot number.
      type_of_halo_id (str): 'merger_tree' if the halo id is a merger tree halo id.
                             'ahf_halos' if the halo id is a *.AHF_halos halo id.

    Returns:
      p_or_v ( [len(inds), 3] np.array ): Position or velocity for the specified inds.
    '''

    # Choose the indices we'll access the data through
    if pos_or_vel == 'pos':
      keys = [ 'Xc', 'Yc', 'Zc' ]
    elif pos_or_vel == 'vel':
      keys = [ 'VXc', 'VYc', 'VZc' ]
    else:
      raise Exception( 'Unrecognized pos_or_vel, {}'.format( pos_or_vel ) )

    # Get the ahf_halo data, if requested.
    if type_of_halo_id == 'ahf_halos':
      self.get_halos( inds )

    # Get the data.
    p_or_v = []
    for key in keys:

      # Get the part
      if type_of_halo_id == 'merger_tree':
        p_or_v_part = self.mtree_halos[ halo_id ][ key ][ inds ] 
      elif type_of_halo_id == 'ahf_halos':
        p_or_v_part = self.ahf_halos[ key ][ halo_id ] 
      else:
        raise Exception( 'Unrecognized type_of_halo_id, {}'.format( type_of_halo_id ) )

      p_or_v.append( p_or_v_part )

    # Finish formatting.
    p_or_v = np.array( p_or_v ).transpose()

    return p_or_v

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

  def save_smooth_mtree_halos( self,  metafile_dir, index=None, include_concentration=False ):
    '''Load halo files, smooth them, and save as a new file e.g., halo_00000_smooth.dat

    Args:
      metafile_dir (str): The directory the metafiles (snapshot_times and used_parameters) are stored in.
      index (str or int) : What type of index to use. Defaults to None, which raises an exception. You *must* choose an index, to avoid easy mistakes.
                           See get_mtree_halos() for a full description.
      include_concentration (bool): Whether or not to add an additional column that gives an analytic value for the halo concentration.
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

  def save_ahf_halos_add( self, snum, metafile_dir ):
    '''Save additional columns that would be part of *.AHF_halos files, if that didn't break AHF.

    Args:
      snum (int): Snapshot number to load.
      metafile_dir (str): The directory the metafiles (snapshot_times and used_parameters) are stored in.
    '''

    # Load the AHF_halos data
    self.get_halos( snum )

    # Create AHF_halos add
    self.ahf_halos_add = pd.DataFrame( {}, index=self.ahf_halos.index )
    self.ahf_halos_add.index.names = ['ID']

    # Get the analytic concentration
    self.ahf_halos_add['cAnalytic'] = self.get_analytic_concentration( metafile_dir, type_of_halo_id='ahf_halos' )

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

      print 'Saving *.AHF_halos_add for snum {}'.format( snum )

      # Save the data
      self.save_ahf_halos_add( snum, metafile_dir )

