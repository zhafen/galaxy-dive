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
import string

########################################################################
########################################################################

class MetafileReader( object ):
  '''Read metafiles.
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
    
    pass
