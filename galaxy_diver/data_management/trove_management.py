#!/usr/bin/env python
'''Code for managing data troves.

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import itertools

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

    ########################################################################

    @property
    def combinations( self ):
        '''Returns:
            All combinations of arguments.
        '''

        if not hasattr( self, '_combinations' ):
            self._combinations = list( itertools.product( *self.args ) )

        return self._combinations

    ########################################################################

    @property
    def data_files( self ):
        '''Returns:
            All data files that should be part of the trove.
        '''

        if not hasattr( self, '_data_files' ):
            self._data_files = [
                self.file_format.format( *args ) for args in self.combinations
             ]

        return self._data_files
