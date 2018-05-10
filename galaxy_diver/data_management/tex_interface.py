#!/usr/bin/env python
'''Interface for saving data to TeX-readable files.

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import os
import warnings

import galaxy_diver.utils.utilities as utilities

########################################################################
########################################################################


class TeXVariableFile( object ):
    '''Class for creating, adding to, and deleting a TeX file of variables with values.'''

    @utilities.store_parameters
    def __init__( self, filename ):
        '''Constructor.

        Args:
            filename (str) :
                Location of file.

        Returns:
            TeXVariableFile object.
        '''

        pass

    ########################################################################

    @property
    def data_dict( self ):
        '''Actual data stored in the file.'''

        if not hasattr( self, '_data_dict' ):

            try:
                # Read the file
                with open( self.filename, 'r' ) as f:
                    lines = f.readlines()

                # Parse
                self._data_dict = {}
                for line in lines:
                    name = line.split( '{' )[1].split( '\\' )[-1][:-1]
                    value = line.split( '{' )[-1].split( '}' )[0]

                    self._data_dict[name] = value

            except IOError:
                print( "No pre-existing file found. Writing will be done to a fresh file." )

                self._data_dict = {}
            
        return self._data_dict

    ########################################################################

    def write( self ):
        '''Write the stored data_dict to the file.'''

        lines = []

        for key, item in self.data_dict.items():

            line = '\\newcommand{\\' + key + '}{' + item + '}\n'

            lines.append( line )

        with open ( self.filename, 'w' ) as f:

            f.writelines( lines )

    ########################################################################

    def save_variable( self, key, item ):
        '''Save a variable to the TeX file.'''

        print( "Saving {} as {}".format( item, key ) )

        if key in self.data_dict.keys():
            warnings.warn(
                "Overwriting variable {}. Previous value {}".format(
                    key,
                    self.data_dict[key],
                )
            )

        self.data_dict[key] = item

        self.write()

    ########################################################################

    def delete_variable( self, key ):

        print( "Deleting {}".format( key ) )

        del self.data_dict[key]

        self.write()

