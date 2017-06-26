#!/usr/bin/env python
'''Tools for reading simulation metafiles.

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

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
    try:
      filepath = os.path.join( self.sdir, 'snapshot_times.txt' )
      
      # Column names
      names = [ 'snum', 'scale-factor', 'redshift', 'time[Gyr]', 'time_width[Myr]' ]

      self.snapshot_times = pd.read_csv( filepath, delim_whitespace=True, skiprows=3, index_col=0, names=names ) 

    # FIRE-1 snapshot times
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
