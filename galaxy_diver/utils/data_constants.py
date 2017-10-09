#!/usr/bin/env python
'''Non-physical constants for use in analyzing data.
For example, this contains the default fill values for invalid data.

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import numpy as np

########################################################################
########################################################################

INT_FILL_VALUE = -99999
FLOAT_FILL_VALUE = np.nan

PTYPES = {
  'gas' : 0,
  'DM' : 1,
  'lowresDM' : 2,
  'star' : 4,
}

STANDARD_PTYPES = [ 'gas', 'DM', 'lowresDM', 'star', ]
