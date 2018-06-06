#!/usr/bin/env python
'''Tools for running executables

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import sys

########################################################################

def choose_config_or_commandline( config_vals ):
    '''This is useful for running notebooks.
    It allows the user to use commandline arguments to replace defaults
    in the notebook.

    Args:
        config_vals (list-like) :
            Default values when no commandline args are used.

    Returns
        used_vals (list-like) :
            Values used, replacing default values with provided commandline
            values.
    '''

    if sys.argv[1] != 'use_commandline':
        return config_vals

    for i in range( len( sys.argv ) - 2 ):
        config_vals[i] = type( config_vals[i] )( sys.argv[i+2] )

    return config_vals
        
    
