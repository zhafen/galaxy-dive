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
    '''

    filepath = os.path.join( self.sdir, 'snapshot_times.txt' )
    
    # Column names
    names = [ 'snum', 'scale-factor', 'redshift', 'time[Gyr]', 'time_width[Myr]' ]

    self.snapshot_times = pd.read_csv( filepath, delim_whitespace=True, skiprows=3, index_col=0, names=names ) 
