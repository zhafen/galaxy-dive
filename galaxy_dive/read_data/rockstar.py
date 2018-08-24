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
import unyt

import galaxy_dive.read_data.metafile as read_metafile

########################################################################

default = object()

########################################################################
########################################################################

class RockstarReader( object ):
    '''Read Rockstar data.
    Note! All positions are in comoving coordinates,
    and everything has 1/h's sprinkled throughout.
    '''

    def __init__( self, sdir ):
        '''Initializes.

        Args:
            sdir (str): Simulation directory to load the Rockstar data from.
        '''

        self.data_dir = sdir

        # Common global names, and what columns those correspond to in the
        # data
        self.general_use_data_names = {
            'X' : 'X',
            'Y' : 'Y',
            'Z' : 'Z',
        }

        # Data units
        self.units = {
            'M200b' : unyt.Msun,
            'Vmax' : unyt.km / unyt.s,
            'Vmax' : unyt.km / unyt.s,
            'R200b' : unyt.Mpc,
            'Rs' : unyt.Mpc,
            'X' : unyt.Mpc,
            'Y' : unyt.Mpc,
            'Z' : unyt.Mpc,
            'VX' : unyt.km / unyt.s,
            'VY' : unyt.km / unyt.s,
            'VZ' : unyt.km / unyt.s,
            'JX' : unyt.Msun * unyt.Mpc * unyt.km / unyt.s,
            'JY' : unyt.Msun * unyt.Mpc * unyt.km / unyt.s,
            'JZ' : unyt.Msun * unyt.Mpc * unyt.km / unyt.s,
            'rs_klypin' : unyt.Mpc,
            'M200b_all' : unyt.Msun,
            'Mvir' : unyt.Msun,
            'M200c' : unyt.Msun,
            'M500c' : unyt.Msun,
            'M180b' : unyt.Msun,
        }

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

        # Rename the index to a more suitable name, without the '#'
        self.halos.index.names = ['ID']

        # Save the snapshot number of the rockstar halos file.
        self.halos_snum = snum

        # Note that we haven't added the additional halos data yet.
        self.halos_added = False

