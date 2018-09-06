#!/usr/bin/env python
'''Script for generating derived data documentation.

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import sys

import galaxy_dive.utils.utilities as utilities

utilities.gen_derived_data_doc( *sys.argv[1:] )
