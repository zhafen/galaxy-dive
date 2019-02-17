#!/usr/bin/env python
'''Compilation of observed and derived astrophysical trends for CGMs.

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import numpy as np
import os
import scipy.interpolate as interp

import galaxy_dive.config as gd_config

########################################################################

def cooling_time(
    r,
    z,
    sim_name,
    physics,
    data_dir = gd_config.DATA_DIR,
    filename = 'Rovertcool_res7100.npz',
):
    '''Extract the pre-calculated cooling time for a given snapshot
    from a look-up table.

    Args:
        r (float or array-like):
            Radii (in pkpc) at which to look up the cooling time.

        z (float or array-like):
            Redshifts at which to look up the cooling time.

        physics (str):
            What simulation variant to use (e.g. 'core' or 'metal_diffusion').

        data_dir (str):
            Directory containing the data.

        filename (str):
            Filename for lookup table.

    Returns:
        value or array-like (same shape as r and z):
            Interpolated values of t_cool from look-up table.
    '''

    assert sim_name[:3] == 'm12', \
        'Cooling time currently only calculated for m12 halos'

    # Load up the file
    filepath = os.path.join( data_dir, filename )
    data = np.load( filepath )

    # Translate the keys from bytes...
    keys = data['key'].astype( str )

    # Make sure the sim name exists in the sim
    assert sim_name[3] in keys[:,0]

    # Identify the relevant simulation snapshots
    matches_sim = keys[:,0] == sim_name[3]
    matches_physics = keys[:,1] == physics
    valid_snapshots = matches_sim & matches_physics

    # Get the redshifts stored in the file
    unsorted_redshifts = keys[:,2][valid_snapshots].astype( float )
    radii = data['Rs'][valid_snapshots]
    t_cools = data['tcools'][valid_snapshots]

    # Sort, tile, and mask
    sort_inds = unsorted_redshifts.argsort()
    redshift = unsorted_redshifts[sort_inds]
    redshift = np.tile( redshift, ( radii.shape[1], 1 ) ).transpose()
    radii = radii[sort_inds]
    t_cools = t_cools[sort_inds]

    # Interpolate
    return interp.griddata(
        ( redshift.flatten(), radii.flatten() ),
        t_cools.flatten(),
        ( z, r ),
    )
    
