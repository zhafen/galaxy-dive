#!/usr/bin/env python
'''Tools for reading Rockstar output files.

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import glob
import numpy as np
import os
import pandas as pd

import galaxy_diver.read_data.metafile as read_metafile

########################################################################

default = object()

########################################################################
########################################################################

class RockstarReader( object ):
    '''Read Rockstar data.
    Note! All positions are in comoving coordinates, and everything has 1/h's sprinkled throughout.
    '''

    def __init__( self, sdir ):
        '''Initializes.

        Args:
            sdir (str): Simulation directory to load the Rockstar data from.
        '''

        self.data_dir = sdir

    ########################################################################
    # Load Data
    ########################################################################

    def get_halos( self, snum, force_reload=False ):
        '''Get out_*.list file for a particular snapshot.

        Args:
            snum (int): Snapshot number to load.

            force_reload (bool): Force reloading, even if there's already an halos file loaded.

        Modifies:
            self.halos (pd.DataFrame): Dataframe containing the requested data.
        '''

        if hasattr( self, 'halos' ):
            if (self.halos_snum == snum) and not force_reload:
                return

        # Load the data
        self.halos_path = os.path.join(
            self.data_dir,
            'out_{:03d}.list'.format( snum ),
        )
        self.halos = pd.read_csv(
            self.halos_path,
            sep = ' ',
            header = 0,
            skiprows = range(1,16),
            index_col = 0
        )

        # Rename the index to a more suitable name, without the '#' and the (1)
        self.halos.index.names = ['ID']

        # Save the snapshot number of the rockstar halos file.
        self.halos_snum = snum

        # Note that we haven't added the additional halos data yet.
        self.halos_added = False

    ########################################################################
    # Utilities
    ########################################################################

    def get_filepath( self, snum, rockstar_file_type ):
        '''Get the filepath for a specified type of Rockstar file.

        Args:
            snum (int): Snapshot number to load.
            rockstar_file_type (str): Can be Rockstar_halos or Rockstar_mtree_idx.

        Returns:
            rockstar_filepath (str): The filepath to the specified file.
        '''

        # Load the data
        rockstar_filename = 'snap{:03d}Rpep..z*.*.{}'.format( snum, rockstar_file_type )
        rockstar_filepath_unexpanded = os.path.join( self.data_dir, rockstar_filename )
        possible_filepaths = glob.glob( rockstar_filepath_unexpanded )
        if len( possible_filepaths ) == 0:
            raise NameError( 'No files to load for snum {} in {}'.format( snum, self.data_dir ) )
        elif len( possible_filepaths ) > 1:
            raise Exception( 'Multiple possible *.{} files to load'.format( rockstar_file_type ) )
        rockstar_filepath = possible_filepaths[0]

        return rockstar_filepath

