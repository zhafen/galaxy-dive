#!/usr/bin/env python
'''Code for managing data troves.

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import itertools
import os

import galaxy_dive.utils.utilities as utilities

########################################################################
########################################################################


class TroveManager( object ):
    '''Class for managing troves of data.'''

    @utilities.store_parameters
    def __init__( self, file_format, *args ):
        '''Constructor.

        Args:
            file_format (str) :
                Format for data files.

            *args :
                Arguments to pass to self.get_file() to get different data files.

        Returns:
            TroveManager object.
        '''

        pass

    ########################################################################

    def get_file( self, *args ):
        '''Default method for getting the data filename.
        
        Args:
            *args :
                Arguments provided. Assumes args[0] is the data dir.

        Returns:
            Filename for a given combination of args.
        '''

        filename = self.file_format.format( *args[1:] )

        return os.path.join( args[0], filename )

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
                self.get_file( *args ) for args in self.combinations
             ]

        return self._data_files

    ########################################################################

    def get_incomplete_combinations( self ):
        '''Returns:
            Combinations in the trove that have not yet been done.
        '''

        incomplete_combinations = []
        for i, data_file in enumerate( self.data_files ):

            if not os.path.isfile( data_file ):
                incomplete_combinations.append( self.combinations[i] )

        return incomplete_combinations

    ########################################################################

    def get_incomplete_data_files( self ):
        '''Returns:
            Data files in the trove that have not yet been done.
        '''

        return [
            self.get_file( *args ) for args \
                in self.get_incomplete_combinations()
        ]

    ########################################################################

    def get_next_args_to_use( self, when_done='return_last' ):
        '''Is this necessary? No. This function is really a wrapper that in
        essence provides documentation.


        Args:
            when_done (str) :
                What to do when there are no incomplete combinations? Defaults
                to returning the last of self.combinations.

        Returns:
            Next set of arguments to use.
        '''

        incomplete_combinations = self.get_incomplete_combinations()

        if len( incomplete_combinations ) == 0:
            if when_done == 'return_last':
                return self.combinations[-1]
            elif when_done == 'return_0':
                return 0

        return self.get_incomplete_combinations()[0]
