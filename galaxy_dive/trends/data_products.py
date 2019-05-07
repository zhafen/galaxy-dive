#!/usr/bin/env python
'''Compilation of functions for interfacing with miscellanious data products.

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import copy
import numpy as np
import os
import pandas as pd

########################################################################

def tidal_tensor_data_grudic(
    snum,
    ids = None,
    data_dir = '/work/03532/mgrudic/tidal_tensor',
):
    '''Load data Mike Grudic processed that contains Tidal Tensor, velocity
    dispersion, and items used for calculating the aforementioned quantities.

    Args:
        snum (int): Snapshot to retrieve the data for.

        ids (array-like): IDs to retrieve. Defaults to all.

        data_dir (str): Path to directory containing data.

    Returns:
        pandas.DataFrame
            DataFrame containing quantities. When given an ID not in the data
            returns NaN values for that ID.
    '''

    def invalid_data_result():
        '''Results when the data is invalid in some form.'''
        base_arr = np.full( len( ids ), np.nan )
        standin_data = {}
        data_keys = [
            'ID',
            'Txx',
            'Tyy',
            'Tzz',
            'Txy',
            'Tyz',
            'Tzx',
            'sigma_v',
            'r_search',
            'cond_num',
        ]
        for key in data_keys:
            standin_data[key] = copy.deepcopy( base_arr )
        standin_data['ID'] = ids
        df = pd.DataFrame( standin_data )
        df = df.set_index( 'ID' )

        return df

    # Load the data
    filename = 'tidal_tensor_{}.npy'.format( snum )
    file_path = os.path.join( data_dir, filename )
    try:
        full_arr = np.load( file_path )
    except FileNotFoundError:
        return invalid_data_result()
    
    # Convert to a pandas data frame to get the selected IDs out.
    data = {
        'ID': full_arr[:,0].astype( int ),
        'Txx': full_arr[:,1],
        'Tyy': full_arr[:,2],
        'Tzz': full_arr[:,3],
        'Txy': full_arr[:,4],
        'Tyz': full_arr[:,5],
        'Tzx': full_arr[:,6],
        'sigma_v': full_arr[:,7],
        'r_search': full_arr[:,8],
        'cond_num': full_arr[:,9],
    }
    df = pd.DataFrame( data, )
    df = df.set_index( 'ID' )

    # Select on IDs
    if ids is not None:
        try:
            df = df.loc[ids]
        except KeyError:
            return invalid_data_result()

    return df
