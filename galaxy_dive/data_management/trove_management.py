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

        self.combinations_to_skip = []
        self.combinations_order = None

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

    def set_order( self, value ):
        '''Change the order in which the trove manager loops through the files.

        Args:
            value (list of ints):
                New order, with one integer per argument provided to the file
                format. The integer corresponds to the order in which you wish
                to loop through the files, with lower values looping through
                first.
        '''

        self.combinations_order = value

        del self.combinations

    ########################################################################

    @property
    def combinations( self ):
        '''Returns:
            All combinations of arguments.
        '''

        if not hasattr( self, '_combinations' ):

            # Reorder
            if self.combinations_order is not None:
                order = [
                    len( self.combinations_order ) - i
                    for i in
                    self.combinations_order
                ]
                args = [
                    arg
                    for _, arg in
                    sorted( zip( order, self.args ) )
                ]
            else:
                args = self.args

            cs = list( itertools.product( *args ) )

            # Reorder back
            if self.combinations_order is not None:
                self._combinations = []
                for c in cs:
                    ordered_c = tuple( [
                        arg for _, arg in
                        sorted( zip( order, c ) )
                    ] )
                    self._combinations.append( ordered_c )
            else:
                self._combinations = cs

        return self._combinations

    @combinations.deleter
    def combinations( self ):
        if hasattr( self, '_combinations' ):
            del self._combinations

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
        '''Determines next combination to use, accounting for combinations to
        skip.

        Args:
            when_done (str) :
                What to do when there are no incomplete combinations? Defaults
                to returning the last of self.combinations.

        Returns:
            Next set of arguments to use.
        '''

        full_incomplete_combinations = self.get_incomplete_combinations()

        incomplete_combinations = []
        for c in full_incomplete_combinations:
            if not c in self.combinations_to_skip:
                incomplete_combinations.append( c )

        if len( incomplete_combinations ) == 0:
            if when_done == 'return_last':
                return self.combinations[-1]
            elif when_done == 'return_0':
                return 0

        return incomplete_combinations[0]
