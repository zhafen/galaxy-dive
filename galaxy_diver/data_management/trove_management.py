#!/usr/bin/env python
'''Code for managing data troves.

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import galaxy_diver.utils.utilities as utilities

########################################################################
########################################################################


class TroveManager( object ):
    '''Class for managing troves of data.'''

    @utilities.store_parameters
    def __init__( self, data_dir, file_format, *args ):
        '''Constructor.

        Args:
            data_dir (str) :
                Directory the data is stored in.

            file_format (str) :
                Format for data files.

            *args :
                Arguments to pass to file_format to get different data files.

        Returns:
            TroveManager object.
        '''

        pass

