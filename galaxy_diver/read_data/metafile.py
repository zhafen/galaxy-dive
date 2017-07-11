#!/usr/bin/env python
'''Tools for reading simulation metafiles.

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import glob
import numpy as np
import os
import pandas as pd

########################################################################
########################################################################

class MetafileReader( object ):
  '''Read simulation metafiles, e.g. snapshot_times.txt
  '''

  def __init__( self, sdir ):
    '''Initializes.

    Args:
      sdir (str): Simulation directory to load the metafiles from.
    '''

    self.sdir = sdir

  ########################################################################

  def get_snapshot_times( self ):
    '''Load the snapshot_times.txt files that are in the simulation directories.

    Modifies:
      self.snapshot_times (pd.DataFrame): A dataframe containing information about the snapshot times.
    '''

    # FIRE-2 snapshot times
    # Assumes the file is named snapshot_times.txt
    try:
      filepath = os.path.join( self.sdir, 'snapshot_times.txt' )
      
      # Column names
      names = [ 'snum', 'scale-factor', 'redshift', 'time[Gyr]', 'time_width[Myr]' ]

      self.snapshot_times = pd.read_csv( filepath, delim_whitespace=True, skiprows=3, index_col=0, names=names ) 

    # FIRE-1 snapshot times (these are simpler)
    # Assumes the file is named output_times.txt
    except:

      filepath = os.path.join( self.sdir, 'output_times.txt' )

      # Column names
      names = [ 'scale-factor', ]

      # Load the data
      self.snapshot_times = pd.read_csv( filepath, names=names )

      # Rename the index
      self.snapshot_times.index.name = 'snum'

      # Get the redshift
      self.snapshot_times['redshift'] = 1./self.snapshot_times['scale-factor'] - 1.

  ########################################################################

  def get_used_parameters( self ):
    '''Load parameters used to run the simulation from, e.g. the gizmo_parameters.txt-usedvalues file.

    Modifies:
      self.used_parameters( pd.DataFrame): A dataframe containing the parameters.
    '''

    potential_filepaths = glob.glob( '{}/*usedvalues'.format( self.sdir ) )

    assert len( potential_filepaths ) < 2, 'Multiple options to choose the parameter file from.'
    assert len( potential_filepaths ) != 0, 'No used parameter file found (e.g. gizmo_parameters.txt-usedvalues).'

    parameter_filepath = potential_filepaths[0]

    self.used_parameters = {}
    with open(parameter_filepath,'r') as input_file:

      # Loop through the lines
      for i, line in enumerate( input_file ):

        # Split the line up
        split_line = line.split()
        
        # Check for lines that don't fit the expectations
        assert len( split_line ) == 2, 'Unexpected format in Line {}'.format( i )

        self.used_parameters[split_line[0]] = split_line[1]


