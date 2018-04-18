#!/usr/bin/env python
'''Code for managing data troves.

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import itertools
import os

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

    ########################################################################

    def get_incomplete_combinations( self ):
        '''Returns:
            Combinations in the trove that have not yet been done.
        '''

        incomplete_combinations = []
        for i, data_file in enumerate( self.data_files ):

            data_path = os.path.join( self.data_dir, data_file )
            
            if not os.path.isfile( data_path ):
                incomplete_combinations.append( self.combinations[i] )

        return incomplete_combinations

    ########################################################################

    def get_incomplete_data_files( self ):
        '''Returns:
            Data files in the trove that have not yet been done.
        '''

        return [
            self.file_format.format( *args ) for args \
                in self.get_incomplete_combinations()
        ]

    ########################################################################

    def get_next_args_to_use( self ):
        '''Is this necessary? No. This function is really a wrapper that in
        essence provides documentation.

        Returns:
            Next set of arguments to use.
        '''

        return self.get_incomplete_combinations()[0]
