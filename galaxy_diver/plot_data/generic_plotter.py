#!/usr/bin/env python
'''Class for plotting simulation data.

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

# Base python imports
import numpy as np

import galaxy_diver.utils.utilities as utilities

########################################################################
########################################################################

class GenericPlotter( object ):

  @utilities.store_parameters
  def __init__( self, data ):
    '''
    Args:
      data ( generic_data.data object or subclass of such ) : The data container to use.
    '''

    pass
    
  ########################################################################
