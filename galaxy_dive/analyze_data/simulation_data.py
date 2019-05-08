#!/usr/bin/env python
'''Class for analyzing simulation data.

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

# Base python imports
import copy
import numpy as np
import numpy.testing as npt
import pandas as pd
import scipy
import scipy.signal as signal

# Imports from my own stuff
import galaxy_dive.analyze_data.ahf as analyze_ahf
import galaxy_dive.read_data.ahf as read_ahf
import galaxy_dive.utils.astro as astro
import galaxy_dive.utils.constants as constants
import galaxy_dive.utils.data_operations as data_operations
import galaxy_dive.analyze_data.generic_data as generic_data
import galaxy_dive.utils.utilities as utilities

########################################################################
########################################################################


class SimulationData( generic_data.GenericData ):
    '''Class for handling simulation data.

    Args:
        data_dir (str) :
             Directory the simulation is contained in.

        halo_data_dir (str) :
             Directory simulation halo catalogs are contained in.
            Defaults to data_dir

        ahf_index (str) :
             What to index the snapshots by.
             Should be the last snapshot in the simulation *if*
             AHF was run backwards from the last snapshot.
             Required to put in manually to avoid easy mistakes.

        averaging_frac (dict) :
            There are some averaged quantities (e.g. COM velocity, total
            angular momentum) that we consider. This is the fraction of
            length_scale_used within which we calculate these.

        length_scale_used (str) :
             What length scale to use for the simulation.
             Will be used to put lengths in fractions.

        z_sun (float) :
            Used mass fraction for solar metallicity.

        halo_data_retrieved (bool) :
            Whether or not we retrieved relevant values from the AHF halo data.

        centered (bool):
            Whether or not the coordinates are centered on the galaxy of
            choice at the start.

        vel_centered (bool) :
            Whether or not the velocities are relative to the galaxy of
            choice at the start.

        hubble_corrected (bool) :
            Whether or not the velocities have had the Hubble flow added
            (velocities must be centered).

        ahf_tag (str) :
            Identifying tag for the ahf merger tree halo files,
            looks for ahf files of type 'halo_00000_{}.dat'.format( tag ).

        main_halo_id (int) :
             What is the halo ID of the main galaxy in the simulation?

        center_method (str or np.array) :
            How to center the coordinates. Options are...

            'halo' (default) : Centers the dataset on the main halo
                (main_halo_id) using AHF halo data.

            np.array : Array of coordinates on which to center the data

        vel_center_method (str or np.array of size 3) :
            How to center the velocity coordinates, i.e. what the
            velocity is relative to. Options are...

            'halo' (default) : Sets velocity relative to the main halo
                (main_halo_id) using AHF halo data.

            np.array of size 3 :
                Centers the dataset on this coordinate.

    Keyword Args:
        function_args (dict):
            Dictionary of args used to specify an arbitrary function with
            which to generate data.
    '''

    @utilities.store_parameters
    def __init__(
        self,
        data_dir = None,
        halo_data_dir = None,
        ahf_index = None,
        averaging_frac = 4.,
        length_scale_used = 'Rstar0.5',
        z_sun = constants.Z_MASSFRAC_SUN,
        halo_data_retrieved = False,
        centered = False,
        vel_centered = False,
        hubble_corrected = False,
        ahf_tag = 'smooth',
        main_halo_id = 0,
        center_method = 'halo',
        vel_center_method = 'halo',
        store_ahf_reader = False,
        **kwargs
    ):

        # Make sure that all the arguments have been specified.
        for attr in vars( self ).keys():
            if attr == 'kwargs':
                continue
            if getattr( self, attr ) is None:

                # Set the analysis dir to data_dir if not given
                if attr == 'halo_data_dir':
                    self.halo_data_dir = self.data_dir

                elif attr == 'ahf_index':
                    continue

                else:
                    raise Exception( '{} not specified'.format( attr ) )

        # By definition, the halo data should not be retrieved when the class is first initiated.
        self.halo_data_retrieved = False

        super( SimulationData, self ).__init__( z_sun=z_sun, **kwargs )

    ########################################################################
    # Properties
    ########################################################################

    @property
    def length_scale( self ):
        '''Property for fiducial simulation length scale.'''

        if self.length_scale_used == 'R_vir':
            return self.r_vir
        elif self.length_scale_used == 'r_scale':
            return self.r_scale
        else:
            return self.halo_data.get_mt_data(
                self.length_scale_used,
                snums = self.snum,
                return_values_only = False, # This is because we're only getting one value out
                a_power = 1.,
            ) / self.data_attrs['hubble']

    ########################################################################

    @property
    def velocity_scale( self ):
        '''Property for fiducial simulation velocity scale.'''

        return self.v_c

    ########################################################################

    @property
    def base_data_shape( self ):
        '''Property for simulation redshift.'''

        if not hasattr( self, '_base_data_shape' ):

            # Use Density as the default data we assume will usually be there.
            if 'Den' in self.data:
                self._base_data_shape = self.data['Den'].shape
            # If it doesn't have density, it might have mass
            elif 'M' in self.data:
                self._base_data_shape = self.data['M'].shape
            else:
                raise Exception( "No data key to base shape off of." )

        return self._base_data_shape

    @base_data_shape.setter
    def base_data_shape( self, value ):
        '''Setting function for simulation redshift property.'''

        # If we try to set it, make sure that if it already exists we don't change it.
        if hasattr( self, '_base_data_shape' ):
            assert self._base_data_shape == value

        else:
            self._base_data_shape = value

    ########################################################################

    @property
    def halo_data( self ):
        '''Halo data used.
        '''

        if not hasattr( self, '_halo_data' ):

            self._halo_data = analyze_ahf.HaloData(
                self.halo_data_dir,
                tag = self.ahf_tag,
                index = self.ahf_index
            )

        return self._halo_data

    ########################################################################

    @property
    def redshift( self ):
        '''Property for simulation redshift.'''

        if not hasattr( self, '_redshift' ):

            # Try to get it from the attributes.
            if 'redshift' in self.data_attrs:
                self._redshift = self.data_attrs['redshift']
            elif hasattr( self, 'data' ):
                if 'redshift' in self.data:
                    self._redshift = self.data['redshift']
            # If not, retrieve halo data, which should set it.
            # In fact, if we call self.retrieve_halo_data() somewhere else and we already set redshift by getting it from
            # the attributes, it will check that it matches.
            else:
                self.retrieve_halo_data()

        return self._redshift

    @redshift.setter
    def redshift( self, value ):
        '''Setting function for simulation redshift property.'''

        # If we try to set it, make sure that if it already exists we don't change it.
        if hasattr( self, '_redshift' ):

            if isinstance( value, np.ndarray ) or isinstance( self._redshift, np.ndarray ):

                is_nan = np.any( [ np.isnan( value ), np.isnan( self._redshift ) ], axis=1 )
                not_nan_inds = np.where( np.invert( is_nan ) )[0]

                test_value = np.array(value)[not_nan_inds]  # Cast as np.ndarray because Pandas arrays can cause trouble.
                test_existing_value = np.array(self._redshift)[not_nan_inds]
                npt.assert_allclose( test_value, test_existing_value, atol=1e-5 )

                self._redshift = value

            else:
                npt.assert_allclose( value, self._redshift, atol=1e-5 )

        else:
            self._redshift = value

    ########################################################################

    @property
    def r_vir( self ):
        '''Property for virial radius.'''

        if not hasattr( self, '_r_vir' ):
            self.retrieve_halo_data()

        return self._r_vir

    @r_vir.setter
    def r_vir( self, value ):

        # If we try to set it, make sure that if it already exists we don't change it.
        if hasattr( self, '_r_vir' ):
            npt.assert_allclose( value, self._r_vir )

        else:
            self._r_vir = value

    ########################################################################

    @property
    def r_scale( self ):
        '''Property for scale radius.'''

        if not hasattr( self, '_r_scale' ):
            self.retrieve_halo_data()

        return self._r_scale

    @r_scale.setter
    def r_scale( self, value ):

        # If we try to set it, make sure that if it already exists we don't change it.
        if hasattr( self, '_r_scale' ):
            npt.assert_allclose( value, self._r_scale )

        else:
            self._r_scale = value

    ########################################################################

    @property
    def v_c( self ):
        '''Property for circular velocity.'''

        if not hasattr( self, '_v_c' ):
            self.retrieve_halo_data()

        return self._v_c

    @v_c.setter
    def v_c( self, value ):

        # If we try to set it, make sure that if it already exists we don't change it.
        if hasattr( self, '_v_c' ):
            npt.assert_allclose( value, self._v_c )

        else:
            self._v_c = value

    ########################################################################

    @property
    def hubble_z( self ):
        '''Property for the hubble function at specified redshift.'''

        if not hasattr( self, '_hubble_z' ):
            self._hubble_z = astro.hubble_parameter(
                self.redshift,
                h=self.data_attrs['hubble'],
                omega_matter=self.data_attrs['omega_matter'],
                omega_lambda=self.data_attrs['omega_lambda'],
                units='km/s/kpc'
            )

        return self._hubble_z

    ########################################################################
    # Overall changes to the data
    ########################################################################

    def center_coords( self ):
        '''Change the location of the origin, if the data isn't already centered.

        Modifies:
            self.data['P'] : Shifts the coordinates to the center.
        '''

        if self.centered:
            return

        if isinstance( self.center_method, np.ndarray ):
            self.origin = copy.copy( self.center_method )

        elif self.center_method == 'halo':
            self.retrieve_halo_data()
            self.origin = copy.copy( self.halo_coords )

        else:
            raise KeyError( "Unrecognized center_method, {}".format( self.center_method ) )

        # Do it like this because we don't know the shape of self.data['P'][0]
        for i in range( 3 ):
            self.data['P'][i] -= self.origin[i]

        # Note that we're now centered
        self.centered = True

    ########################################################################

    def center_vel_coords( self ):
        '''Get velocity coordinates to center on the main halo.

        Modifies:
            self.data['V'] : Makes all velocities relative to self.vel_origin
        '''

        if self.vel_centered:
            return

        if isinstance( self.vel_center_method, np.ndarray ):
            self.vel_origin = copy.copy( self.vel_center_method )

        elif self.vel_center_method == 'halo':
            self.retrieve_halo_data()
            self.vel_origin = copy.copy( self.halo_velocity )

        else:
            raise KeyError( "Unrecognized vel_center_method, {}".format( self.vel_center_method ) )

        # Do it like this because we don't know the shape of self.data['V'][0]
        for i in range( 3 ):
            self.data['V'][i] -= self.vel_origin[i]

        self.vel_centered = True

    ########################################################################

    def add_hubble_flow( self ):
        '''Correct for hubble flow movement.

        Modifies:
            self.data['V'] : Accounts for hubble flow, relative to origin
        '''

        if self.hubble_corrected:
            return

        self.center_vel_coords()

        # Handle weird formatting that happens when using data frames.
        if isinstance( self.hubble_z, pd.Series ):
            self.data['V'] += self.get_data( 'P' ) * self.hubble_z.values
        else:
            self.data['V'] += self.get_data( 'P' ) * self.hubble_z

        self.hubble_corrected = True

    ########################################################################
    # Get Data
    ########################################################################

    def get_data( self, data_key, sl=None ):
        '''Get the data from within the class. Only for getting the data. No post-processing or changing the data
        (putting it in particular units, etc.) The idea is to calculate necessary quantities as the need arises,
        hence a whole function for getting the data.

        Args:
            data_key (str) : Key in the data dictionary for the key we want to get
            sl (slice) : Slice of the data, if requested.

        Returns:
            data (np.ndarray) : Requested data.
        '''

        # Loop through, handling issues
        n_tries = 10
        for i in range( n_tries ):
            try:

                # Positions
                if self.key_parser.is_position_key( data_key ):
                    data = self.get_position_data( data_key )

                # Velocities
                elif self.key_parser.is_velocity_key( data_key ):

                    data = self.get_velocity_data( data_key )

                # Arbitrary functions of the data
                elif data_key == 'Function':

                    raise Exception( "TODO: Test this" )

                    # Use the keys to get the data
                    function_data_keys = self.kwargs['function_args']['function_data_keys']
                    input_data = [ self.get_data(function_data_key) for function_data_key in function_data_keys ]

                    # Apply the function
                    data = self.kwargs['function_args']['function']( input_data )

                # Other
                else:
                    data = self.data[data_key]

            # Calculate missing data
            except KeyError as e:
                self.handle_data_key_error( data_key )
                continue

            break

        if 'data' not in locals().keys():
            raise KeyError( "After {} tries, unable to find or create data_key, {}".format( i + 1, data_key ) )

        if sl is not None:
            return data[sl]

        return data

    ########################################################################

    def get_position_data( self, data_key ):
        '''Get position data (assuming the data starts with an 'R')

        Args:
            data_key (str) : Key in the data dictionary for the key we want to get

        Returns:
            data (np.ndarray) : Requested data.
        '''

        self.center_coords()

        # Transpose in order to account for when the data isn't regularly shaped
        if data_key == 'Rx':
            data = self.data['P'][0, :]
        elif data_key == 'Ry':
            data = self.data['P'][1, :]
        elif data_key == 'Rz':
            data = self.data['P'][2, :]
        else:
            data = self.data[data_key]

        return data

    ########################################################################

    def get_velocity_data( self, data_key ):
        '''Get position data (assuming the data starts with a 'V')

        Args:
            data_key (str) : Key in the data dictionary for the key we want to get

        Returns:
            data (np.ndarray) : Requested data.
        '''

        self.center_vel_coords()
        self.add_hubble_flow()

        # Get data
        if data_key == 'Vx':
            data = self.data['V'][0, :]
        elif data_key == 'Vy':
            data = self.data['V'][1, :]
        elif data_key == 'Vz':
            data = self.data['V'][2, :]
        else:
            data = self.data[data_key]

        return data

    ########################################################################

    def handle_data_key_error( self, data_key ):
        '''When get_data() fails to data_key in self.data, it passes the data_key to try and generate that data.

        Args:
            data_key (str) : Key to try and generate data for

        Modifies:
            self.data[data_key] (np.array) : If it finds a function to generate the data, it will do so
        '''

        if self.verbose:
            print( 'Data key {} not found in data. Attempting to calculate.'.format( data_key ) )

        method_str = 'calc_{}'.format( data_key )
        if hasattr( self, method_str ):
            getattr( self, method_str )()

        # SimulationData methods
        elif data_key == 'R':
            self.calc_radial_distance()
        elif data_key == 'Vmag':
            self.calc_velocity_magnitude()
        elif data_key == 'Vr':
            self.calc_radial_velocity()
        elif data_key == 'Vtan':
            self.calc_tangential_velocity()
        elif data_key == 'ind':
            self.calc_inds()
        elif data_key == 'L':
            self.calc_ang_momentum()
        elif data_key == 'Phi':
            self.calc_phi()
        elif data_key == 'AbsPhi':
            self.calc_abs_phi()
        elif data_key == 'NumDen':
            self.calc_num_den()
        elif data_key == 'HDen':
            self.calc_H_den()
        elif data_key == 'HIDen':
            self.calc_HI_den()

        # TODO: Move these to the subclasses somehow.
        # Subclass methods
        elif data_key == 'T':
            # TODO: This is a hacky fix, that should be changed...
            if hasattr( self, 'calc_temp' ):
                self.calc_temp()
        elif data_key == 'Pressure':
            self.calc_pressure()

        else:
            raise KeyError( 'NULL data_key, data_key = {}'.format( data_key ) )

    ########################################################################

    def get_distance_to_point( self, point ):
        '''This is *not* unit tested.

        Args:
            point (array-like of shape (3,)) : The point you want to find the distance to for each particle.
        '''

        d_to_point = scipy.spatial.distance.cdist( self.get_data( 'P' ).transpose(), point[np.newaxis, :] )

        return d_to_point.flatten()

    ########################################################################

    def get_potential( self, point ):
        '''This is *not* unit tested.

        Args:
            point (array-like of shape (3,)) : The point you want to find the potential at
        '''

        d_to_point = self.get_distance_to_point( point )

        potential_per_particle = -1. * constants.UNITG_UNIV * self.get_data( 'M' ) / d_to_point

        total_potential = potential_per_particle.sum()

        return total_potential

    ########################################################################
    # Full calculations based on the data
    ########################################################################

    def calc_radial_distance( self ):
        '''Calculate the distance from the origin for a given particle.'''

        self.data['R'] = np.sqrt(
            self.get_data( 'Rx' )**2. + \
            self.get_data( 'Ry' )**2. + \
            self.get_data( 'Rz' )**2.
        )

        return self.data['R']

    ########################################################################

    def calc_velocity_magnitude( self ):
        '''Calculate the velocity relative to an origin velocity
        for a given particle.'''

        self.data['Vmag'] = np.sqrt(
            self.get_data( 'Vx' )**2. + \
            self.get_data( 'Vy' )**2. + \
            self.get_data( 'Vz' )**2.
        )

        return self.data['Vmag']

    ########################################################################

    def calc_rho_xy( self ):
        '''Calculate impact parameter in the xy-plane.
        '''

        self.data['rho_xy'] = np.sqrt(
            self.get_data( 'Rx' )**2. + \
            self.get_data( 'Ry' )**2.
        )

        return self.data['rho_xy']

    ########################################################################

    def calc_rho_xz( self ):
        '''Calculate impact parameter in the xz-plane.
        '''

        self.data['rho_xz'] = np.sqrt(
            self.get_data( 'Rx' )**2. + \
            self.get_data( 'Rz' )**2.
        )

        return self.data['rho_xz']

    ########################################################################

    def calc_rho_yz( self ):
        '''Calculate impact parameter in the yz-plane.
        '''

        self.data['rho_yz'] = np.sqrt(
            self.get_data( 'Ry' )**2. + \
            self.get_data( 'Rz' )**2.
        )

        return self.data['rho_yz']

########################################################################
########################################################################


class SnapshotData( SimulationData ):
    '''Class for analysis of a single snapshot of data.'''

    def __init__( self, snum, *args, **kwargs ):
        '''
        Args:
            snum (int or array of ints) : Snapshot or snapshots to inspect.
        '''

        self.snum = snum

        super( SnapshotData, self ).__init__( *args, **kwargs )

    ########################################################################
    # Get Additional Data
    ########################################################################

    def retrieve_halo_data( self ):

        if self.halo_data_retrieved:
            return

        # Load the AHF data
        ahf_reader = read_ahf.AHFReader( self.halo_data_dir )
        ahf_reader.get_mtree_halos( index=self.ahf_index, tag=self.ahf_tag )

        # Select the main halo at the right redshift
        mtree_halo = ahf_reader.mtree_halos[self.main_halo_id].loc[self.snum]

        # Add the halo data to the class.
        self.redshift = mtree_halo['redshift']
        halo_coords_comoving = np.array( [ mtree_halo['Xc'], mtree_halo['Yc'], mtree_halo['Zc'] ] )
        self.halo_coords = halo_coords_comoving / (1. + self.redshift) / self.data_attrs['hubble']
        self.halo_velocity = np.array( [ mtree_halo['VXc'], mtree_halo['VYc'], mtree_halo['VZc'] ] )
        self.r_vir = mtree_halo['Rvir'] / (1. + self.redshift) / self.data_attrs['hubble']
        self.r_scale = self.r_vir / mtree_halo['cAnalytic']
        self.m_vir = mtree_halo['Mvir'] / self.data_attrs['hubble']
        self.m_gas = mtree_halo['M_gas'] / self.data_attrs['hubble']
        self.m_star = mtree_halo['M_star'] / self.data_attrs['hubble']

        # Calculate the circular velocity
        self.v_c = astro.circular_velocity( self.r_vir, self.m_vir )

        self.halo_data_retrieved = True

        if self.store_ahf_reader:
            self.ahf_reader = ahf_reader

    ########################################################################
    # Properties
    ########################################################################

    @property
    def central_mask( self ):
        '''This mask is used when, for example, finding the velocity of the
        center of mass.
        '''

        if not hasattr( self, '_central_mask' ):

            self._central_mask  = self.data_masker.mask_data(
                'Rf',
                0.,
                self.averaging_frac,
                return_or_store = 'return',
            )

        return self._central_mask

    ########################################################################

    @property
    def v_com( self ):
        '''Property for the velocity of the center of mass.'''

        if not hasattr( self, '_v_com' ):

            m_ma = self.get_selected_data( 'M', self.central_mask )
            v_ma = self.get_selected_data( 'V', self.central_mask )

            self._v_com = ( v_ma * m_ma ).sum( 1 ) / m_ma.sum()

        return self._v_com

    ########################################################################

    @property
    def total_ang_momentum(self):
        '''Calculate the total angular momentum vector.'''

        # Exit early if already calculated.
        if not hasattr( self, '_total_ang_momentum' ):

            # Adapt for application to 'l', which is a multidimensional array
            inner_mask = np.array([ self.central_mask ] * 3)

            # Apply masks
            ang_momentum = self.get_data('L')
            l_ma = np.ma.masked_array(ang_momentum, mask=inner_mask)

            # Get the total angular momentum
            self._total_ang_momentum = np.zeros(3)
            for i in range(3):
                self._total_ang_momentum[i] = l_ma[i].sum()

        return self._total_ang_momentum

    ########################################################################

    @property
    def dN_halo(self, R_vir=None, time_units='abs_length'):
        '''Calculate dN_halo/dX/dlog10Mh or dN_halo/dz/dlog10Mh. X is absorption path length (see for example Ribaudo+11)

        time_units: 'abs_length'- dN_halo/dX/dlog10Mh
                                'redshift' - dN_halo/dz/dlog10Mh
        R_vir:      None - Default, assumes given.
                                'BN' Calculates R_vir from the mass and redshift
        '''

        raise Exception( "TODO: Test this" )

        # Previous code that needs to be cleaned up below.

        # # Choose cosmology
        # cosmo = Cosmology.setCosmology('WMAP9')

        # # Calculate the virial radius, if necessary
        # if R_vir is not None:
        #     if R_vir == 'BN':
        #         # Calculate R_vir off of the cosmocode def, and convert to proper kpc.
        #         self.R_vir = cosmo.virialRadius(M, z) * 10.**3.

        # # Make h easier to use (don't have to write the whole thing out...)
        # h = self.data_attrs['hubble']

        # Mh = h * self.M_vir  # Convert the halo mass to Msun/h, so as to feed it into the HMF.
        # dldz = np.abs(cosmo.line_elt(self.redshift))  # Cosmological line element in Mpc.
        # dldz_kpc = dldz * 10.**3.
        # dndlog10M = cosmo.HMF(Mh, self.redshift) * self.M_vir * np.log(10) * h**4.  # HMF in 1/Mpc^3
        # dndlog10M_kpc = dndlog10M * 10.**-9.
        # dN_halo = dldz_kpc * dndlog10M_kpc * np.pi * self.R_vir**2.

        # # Convert from per redshift to per absorption path length.
        # if time_units == 'abs_length':
        #     dN_halo /= cosmo.dXdz(z)
        # elif time_units == 'redshift':
        #     pass

        # return dN_halo

    ########################################################################
    # Full calculations of the data
    ########################################################################

    def calc_velocity_magnitude(self):
        '''Calculate the radial velocity.'''

        # Center velocity and radius
        self.center_coords()
        self.center_vel_coords()

        # Calculate the radial velocity
        self.data['Vmag'] = np.sqrt(
            self.get_data( 'Vx' )**2. + \
            self.get_data( 'Vy' )**2. + \
            self.get_data( 'Vz' )**2.
        )

    ########################################################################

    def calc_radial_velocity(self):
        '''Calculate the radial velocity.'''

        # Center velocity and radius
        self.center_coords()
        self.center_vel_coords()

        # Calculate the radial velocity
        self.data['Vr'] = ( self.get_data( 'V' ) * self.get_data('P')).sum(0) / self.get_data('R')

    ########################################################################

    def calc_tangential_velocity(self):
        '''Calculate the radial velocity.'''

        # Center velocity and radius
        self.center_coords()
        self.center_vel_coords()

        # Calculate the radial velocity
        self.data['Vtan'] = np.sqrt( self.get_data( 'Vmag' )**2. - self.get_data( 'Vr' )**2. )

    ########################################################################

    def calc_inds(self):
        '''Calculate the indices the data are located at, prior to any masks.'''

        raise Exception( "TODO: Test this" )

        # Flattened index array
        flat_inds = np.arange(self.get_data('Den').size)

        # Put into a multidimensional array
        self.data['ind'] = flat_inds.reshape(self.get_data('Den').shape)

    ########################################################################

    def calc_ang_momentum( self ):
        '''Calculate the angular momentum.'''

        m_mult = np.array( [ self.get_data('M'), ] * 3 )

        p = self.get_data('P')
        v = self.get_data('V')

        l = np.cross( p, v, 0, 0).transpose()
        l *= m_mult

        self.data['L'] = l

    ########################################################################

    def calc_phi( self, normal_vector='total ang momentum' ):
        '''Calculate the angle (in degrees) from some vector.
        By default the vector is the total angular momentum.
        '''

        raise Exception( "TODO: Test this" )

        if vector == 'total ang momentum':
            # Calculate the total angular momentum vector, if it's not calculated yet
            self.normal_vector = self.calc_total_ang_momentum()
        else:
            self.normal_vector = normal_vector

        # Get the dot product
        P = self.get_data('P')
        dot_product = np.zeros(P[0, :].shape)
        for i in range(3):
            dot_product += self.v[i] * P[i, :]

        # Isolate for the cosine
        cos_phi = dot_product / self.get_data('R') / np.linalg.norm(self.v)

        # Get the angle (in degrees)
        self.data['Phi'] = np.arccos(cos_phi) * 180. / np.pi

    ########################################################################

    def calc_abs_phi(self, vector='total gas ang momentum'):
        '''The angle (in degrees) from some vector, but don't mirror
        it around 90 degrees (e.g. 135 -> 45 degrees, 180 -> 0 degrees).'''

        raise Exception( "TODO: Test this" )

        # Get the original Phi
        self.calc_phi(vector)

        self.data['AbsPhi'] = np.where(self.data['Phi'] < 90., self.data['Phi'], np.absolute(self.data['Phi'] - 180.))

    ########################################################################

    def calc_num_den(self):
        '''Calculate the number density (it's just a simple conversion...).'''

        self.data['NumDen'] = self.data['Den'] * constants.UNITDENSITY_IN_NUMDEN

    ########################################################################

    def calc_H_den(self):
        '''Calculate the H density in cgs (cm^-3). Assume the H fraction is ~0.75'''

        # Assume Hydrogen makes up 75% of the gas
        X_H = 0.75

        self.data['HDen'] = X_H * self.data['Den'] * constants.UNITDENSITY_IN_NUMDEN

    ########################################################################

    def calc_HI_den(self):
        '''Calculate the HI density in cgs (cm^-3).'''

        raise Exception( "TODO: Test this" )

        # Assume Hydrogen makes up 75% of the gas
        X_H = 0.75

        # Calculate the hydrogen density
        HDen = X_H * self.data['Den'] * constants.gas_den_to_nb

        self.data['HIDen'] = self.data['nHI'] * HDen

    ########################################################################
    # Non-Altering Calculations
    ########################################################################

    def dist_to_point(self, point, units='default'):
        '''Calculate the distance to a point for all particles.

        point : np array that gives the point
        '''

        raise Exception( "TODO: Test this/update to use scipy cdist" )

        # Calculate the distance to the point
        relative_positions = self.get_data('P').transpose() - np.array(point)
        d_mag = np.linalg.norm(relative_positions, axis=1)

        # Put in different units if necessary
        if units == 'default':
            pass
        elif units == 'h':
            d_mag /= self.get_data('h')
        else:
            raise Exception('Null units, units = {}'.format(units))

        return d_mag

    ########################################################################

    def calc_mu(self):
        '''Calculate the mean molecular weight. '''

        y_helium = self.data['Z_Species'][:, 0]  # Get the mass fraction of helium
        mu = 1. / (1. - 0.75 * y_helium + self.data['ne'])

        return mu

########################################################################
########################################################################

class TimeData( SimulationData ):
    '''Class for analysis of a time series data, e.g. the worldlines of a number of particles.'''

    def __init__(
        self,
        data_masker = None,
        *args, **kwargs
    ):

        if data_masker is None:
            data_masker = TimeDataMasker( self )

        super( TimeData, self ).__init__(
            data_masker = data_masker,
            *args,
            **kwargs
        )

    ########################################################################

    @property
    def n_particles( self ):
        '''Number of snapshots, i.e. data points on the time axis.'''

        if not hasattr( self, '_n_particles' ):
            self._n_particles = self.base_data_shape[0]

        return self._n_particles

    ########################################################################

    @property
    def n_snaps( self ):
        '''Number of snapshots, i.e. data points on the time axis.'''

        if not hasattr( self, '_n_snaps' ):
            self._n_snaps = self.base_data_shape[1]

        return self._n_snaps

    ########################################################################

    def retrieve_halo_data( self ):

        if self.halo_data_retrieved:
            return

        # Load the AHF data
        ahf_reader = read_ahf.AHFReader( self.halo_data_dir )
        ahf_reader.get_mtree_halos( index=self.ahf_index, tag=self.ahf_tag )

        # Select the main halo at the right redshift
        mtree_halo = ahf_reader.mtree_halos[self.main_halo_id].loc[self.snums]

        # Add the halo data to the class.
        self.redshift = mtree_halo['redshift']
        scale_factor_and_hinv = 1. / (1. + self.redshift) / self.hubble_param

        halo_coords_comoving = np.array( [ mtree_halo['Xc'], mtree_halo['Yc'], mtree_halo['Zc'] ] )
        self.halo_coords = halo_coords_comoving * scale_factor_and_hinv[np.newaxis, :]
        self.halo_velocity = np.array( [ mtree_halo['VXc'], mtree_halo['VYc'], mtree_halo['VZc'] ] )
        self.r_vir = mtree_halo['Rvir'] * scale_factor_and_hinv
        self.r_scale = self.r_vir / mtree_halo['cAnalytic']
        self.m_vir = mtree_halo['Mvir'] / self.hubble_param
        self.m_gas = mtree_halo['M_gas'] / self.hubble_param
        self.m_star = mtree_halo['M_star'] / self.hubble_param

        # Calculate the circular velocity
        self.v_c = astro.circular_velocity( self.r_vir, self.m_vir )

        self.halo_data_retrieved = True

        if self.store_ahf_reader:
            self.ahf_reader = ahf_reader

    ########################################################################

    def calc_radial_velocity( self ):
        '''Calculate the distance from the origin for a given particle.'''

        v_all = self.get_data( 'V' )
        p_all = self.get_data( 'P' )
        r_all = self.get_data( 'R' )

        # Note that we don't need to add the hubble flow, because we've
        # already done so in self.get_velocity_data()
        assert self.hubble_corrected

        v_r = []
        for i in range( self.n_snaps ):

            v = v_all[:,:,i]
            p = p_all[:,:,i]
            r = r_all[:,i]

            v_r_i = ( v * p ).sum( axis=0 )/r

            v_r.append( v_r_i )

        # Format the output
        self.data['Vr'] = np.array( v_r ).transpose()

    ########################################################################

    def handle_data_key_error( self, data_key ):

        method_str = 'calc_{}'.format( data_key )
        if hasattr( self, method_str ):
            getattr( self, method_str )()
        elif self.calc_time_as_classification( data_key ):
            return
        elif self.calc_time_until_not_classification( data_key ):
            return

        else:
            super( TimeData, self ).handle_data_key_error( data_key )
            # raise KeyError( 'NULL data_key, data_key = {}'.format( data_key ) )

    ########################################################################

    def get_processed_data(
        self,
        data_key,
        sl = None,
        smooth_data = False,
        smoothing_window_length = 9,
        smoothing_polyorder = 3,
        a_power = None,
        scale_key = None,
        scale_a_power = None,
        scale_h_power = None,
        tile_data = False,
        tile_dim = 'auto',
        *args, **kwargs
    ):
        '''Modified method for getting processed method. For the most part is
        equivalent to calling the method of the parent class, but is also
        capable of scaling the retrieved data by a column from the halo data.

        Args:
            data_key (str) : What to get out?

            sl (slice) : Slice of the data, if requested.

            smooth_data (bool) : If True, smooth the data.

            smoothing_window_length, smoothing_polyorder (int) :
                Arguments for how to smooth the data.

            scale_key (str) :
                Halo data entry by which to divide the data by.

            scale_a_power (float) :
                The halo data that we are scaling processed_data by will be
                multiplied by a to this power.
                Useful for data in cosmological units (as is often normal).

            scale_h_power (float) :
                The halo data that we are scaling processed_data by will be
                multiplied by the hubble parameter to this power.
                Useful for data in cosmological units (as is often normal).

            tile_data (bool) :
                If True, tile data along a given direction. This is usually for
                data formatting purposes.

            tile_dim (str) :
                If the data is tiled, what dimension of the data should match?
                Options:
                    'auto' :
                        Tiles according to data size.
                    'match_snaps' :
                        The data is tiled such that the new shape is
                        (self.n_snaps, data_size).
                    'match_particles' :
                        The data is tiled such that the new shape is
                        (data_size, self.n_particles).

            *args, **kwargs :
                Passed to SimulationData.get_processed_data.

        Returns:
            processed_data (array-like) : Requested data array.
        '''

        if ( sl is not None ) and tile_data:
            used_sl = None
        else:
            used_sl = sl

        processed_data = super( TimeData, self ).get_processed_data(
            data_key,
            sl = used_sl,
            *args, **kwargs
        )

        if smooth_data:
            processed_data = signal.savgol_filter(
                processed_data,
                window_length = smoothing_window_length,
                polyorder = smoothing_polyorder,
            )

        if scale_key is not None:

            # Get the data
            data_to_div_by = self.halo_data.get_mt_data(
                scale_key,
                mt_halo_id = self.main_halo_id,
                a_power = scale_a_power,
                snums = self.snums
            )

            if scale_h_power is not None:
                processed_data /= self.hubble_param**scale_h_power

            if sl is not None:
                try:
                    data_to_div_by = data_to_div_by[sl]
                # For when we're getting a slice of a single snapshot
                except IndexError:
                    assert sl[0] == slice(None)
                    data_to_div_by = data_to_div_by[sl[1]]

            processed_data /= data_to_div_by

        if tile_data:

            if tile_dim == 'auto':
                if processed_data.shape == ( self.n_particles, ):
                    tile_dim = 'match_snaps'
                elif processed_data.shape == ( self.n_snaps, ):
                    tile_dim = 'match_particles'
                elif processed_data.shape == self.base_data_shape:
                    tile_dim = None
                else:
                    raise Exception(
                        "Unrecognized data shape, {}".format(
                            processed_data.shape
                        )
                    )

            if tile_dim == 'match_snaps':
                processed_data = np.tile(
                    processed_data,
                    ( self.n_snaps, 1),
                ).transpose()

            elif tile_dim == 'match_particles':
                processed_data = np.tile(
                    processed_data,
                    ( self.n_particles, 1),
                )

            if sl is not None:
                processed_data = processed_data[sl]

        return processed_data

    ########################################################################

    def get_selected_data_over_time( self, *args, **kwargs ):
        '''Wrapper for geting masked data as a function of time.'''

        return self.data_masker.get_selected_data_over_time( *args, **kwargs )

    ########################################################################

    @property
    def snums( self ):

        # TODO: This is a workable structure for now, but it's not ideal. This may not always be how we get the snum
        return self.data['snum']

    ########################################################################

    @property
    def hubble_param( self ):

        # TODO: This is a workable structure for now, but it's not ideal. The hubble parameter may not always be here.
        return self.data_attrs['hubble']

    ########################################################################

    def calc_ang_momentum( self ):
        '''The angular momentum (in the standard coordinates).

        Modifies:
            self.data['L'] (array-like):
                Angular momentum of each resolution element.
        '''

        m_mult = np.array( [ self.get_data('M'), ] * 3 )

        p_all = self.get_data('P')
        v_all = self.get_data('V')

        l_all = np.zeros( ( 3, self.n_particles, self.n_snaps ) )
        # Calculate the angular momentum at each redshift
        for i in range( self.n_snaps ):                                         
                                                                                
            v = v_all[:,:,i]                                                    
            p = p_all[:,:,i]                                                    
            m = m_mult[:,:,i] 

            l = np.cross( p, v, 0, 0).transpose()
            l *= m

            l_all[:,:,i] = l

        self.data['L'] = l_all

    ########################################################################

    def calc_phi( self, normal_vector='total ang momentum' ):
        '''Calculate the angle (in degrees) from some vector.
        By default the vector is the total angular momentum.

        Args:
            normal_vector (str or array-like):
                Vector that represents the vertical.
        '''

        # Exit early if already calculated with the same normal vector
        if 'Phi' in self.data:
            same = np.allclose(
                self.normal_vector,
                normal_vector
            )
            if same:
                return

        # Set up the normal vector
        if normal_vector == 'total ang momentum':
            self.normal_vector = self.total_ang_momentum
        else:
            self.normal_vector = normal_vector

        # Get all the data we need
        p_all = self.get_data('P')
        r_all = self.get_data('R')

        # Loop over each snapshot and calculate
        phi_all = np.zeros( self.base_data_shape )
        for i in range( self.n_snaps ):

            p = p_all[:,:,i]                                                    
            r = r_all[:,i]

            # Get the dot product
            dot_product = np.zeros( p[0,:].shape )
            for j in range(3):
                dot_product += self.normal_vector[j] * p[j,:]

            # Isolate for the cosine
            cos_phi = dot_product / r / np.linalg.norm( self.normal_vector )

            # Get the angle (in degrees)
            phi_all[:,i] = np.arccos( cos_phi ) * 180. / np.pi

        self.data['Phi'] = phi_all

    ########################################################################

    def calc_abs_phi(self, normal_vector='total ang momentum' ):
        '''Calculate the angle (in degrees) from some vector, but mirror
        values past 90 degrees (e.g. 135 -> 45 degrees, 180 -> 0 degrees).
        This is useful when there's symmetry above and below 90.

        Args:
            normal_vector (str or array-like):
                Vector that represents the vertical. Defaults to using the
                total stellar angular momentum of the main galaxy.

        Modifies:
            self.data['AbsPhi'] (array-like):
                Angle from the vector, in many cases acting as the angle
                from the disk axis.
        '''

        # Get the original Phi
        self.calc_phi( normal_vector )

        phi = self.data['Phi']

        # Actual calculation
        self.data['AbsPhi'] = np.where(
            phi < 90.,
            phi,
            np.absolute( phi - 180.)
        )

    ########################################################################

    def calc_time_as_classification( self, data_key ):

        # Check if we should be running this function (does the provided
        # data_key even match the format we want to parse?)
        if data_key[:7] != 'time_as':
            return False

        # Get the data key for the classification
        classification_data_key = 'is_{}'.format( data_key[8:] )

        # Get out the classification data itself
        classification = self.get_data( classification_data_key )

        # Get out the time intervals, and tile them for formatting
        dt = self.get_data( 'dt' )

        # Fill in the array row by row
        time_as_classification = np.zeros( classification.shape )
        for i, row in enumerate( classification ):

            # Identify regions of contiguous classification
            contiguous_regions = data_operations.contiguous_regions( row )

            # Find the cumulative time in specified regions
            for start, end in contiguous_regions:

                dt_region = dt[start:end]

                # We need to flip when we sum because we want a reverse sum
                # (due to data formatting, t=0 is at j=-1)
                cumtime_region = np.cumsum( dt_region[::-1] )[::-1]

                # Store that time
                time_as_classification[i, start:end] = cumtime_region

        self.data[data_key] = time_as_classification

        return True

    ########################################################################

    def calc_time_until_not_classification( self, data_key ):

        # Check if we should be running this function (does the provided
        # data_key even match the format we want to parse?)
        if data_key[:14] != 'time_until_not':
            return False

        # Get the data key for the classification
        classification_data_key = 'is_{}'.format( data_key[15:] )

        # Get out the classification data itself
        classification = self.get_data( classification_data_key )

        # Get out the time intervals, and tile them for formatting
        dt = self.get_data( 'dt' )

        # dt[i] = time difference between i and i+1, so time until is 0
        # if the the particle changes the next snapshot
        dt =  np.insert( dt, 0, 0. )
        
        # Fill in the array row by row
        time_un_classification = np.zeros( classification.shape )
        for i, row in enumerate( classification ):

            # Identify regions of contiguous classification
            contiguous_regions = data_operations.contiguous_regions( row )

            # Find the cumulative time in specified regions
            for start, end in contiguous_regions:

                dt_region = dt[start:end]

                # We need to flip when we sum because we want a reverse sum
                # (due to data formatting, t=0 is at j=-1)
                cumtime_region = np.cumsum( dt_region )

                # Store that time
                try:
                    time_un_classification[i, start:end] = cumtime_region
                except:
                    #DEBUG
                    import pdb; pdb.set_trace()

        self.data[data_key] = time_un_classification

        return True

    ########################################################################

    def calc_metal_mass( self ):
        '''Calculate the metal mass held by each resolution element.

        Modifies:
            self.data['enriched_metal_mass'] (array-like):
                Metal mass from enrichment.
        '''

        metal_mass = self.get_data( 'M' ) * self.get_data( 'Z' ) * self.z_sun

        self.data['metal_mass'] = metal_mass

    ########################################################################

    def calc_enriched_metal_mass( self ):
        '''Calculate the metal mass that comes from enrichment for
        each resolution element, not counting mass that's at the metallicity
        floor. Assumes that the there will always be at least one resolution
        element in the simulation that's at the metallicity floor.

        Modifies:
            self.data['enriched_metal_mass'] (array-like):
                Metal mass from enrichment.
        '''

        enrichment_fraction = (
            self.get_data( 'Z' ) - np.nanmin( self.get_data( 'Z' ) )
        )

        enrichment_fraction *= self.z_sun

        enriched_metal_mass = self.get_data( 'M' ) * enrichment_fraction

        self.data['enriched_metal_mass'] = enriched_metal_mass

########################################################################
########################################################################

class TimeDataMasker( generic_data.DataMasker ):
    '''Data masker for worldline data.'''

    def __init__( self, time_data ):

        super( TimeDataMasker, self ).__init__( time_data )

    ########################################################################

    def get_selected_data_over_time(
        self,
        data_key,
        snum,
        mask = 'total',
        optional_masks = None,
        n_samples = None,
        seed = None,
        *args, **kwargs
    ):
        '''Get data over the full time history, based on its mask at
        one time.
    
        Args:
            data_key (str): Data to get.
            
            snum (int): Snapshot to get the data corresponding to.
        '''

        # Make sure we don't try to pass a slice to any keyword arguments
        assert 'sl' not in kwargs, 'Taking slices of the original data' + \
                ' should not be done when using get_selected_data_over_time'

        # Get the appropriate mask
        if isinstance( mask, np.ndarray ):
            used_mask = mask
        elif isinstance( mask, bool ) or isinstance( mask, np.bool_ ):
            if not mask:
                used_mask = np.zeros(
                    self.data_object.base_data_shape
                ).astype( bool )
            else:
                raise Exception( "All data is masked." )
        elif mask == 'total':
            used_mask = self.get_total_mask( optional_masks=optional_masks )
        else:
            raise KeyError( "Unrecognized type of mask, {}".format( mask ) )

        # Get the relevant ind
        correct_snum = self.data_object.snums == snum
        ind = np.arange( self.data_object.n_snaps )[correct_snum][0]

        # Get the boolean for the selected data
        sl = np.invert( used_mask[:,ind] )

        # If we want to get the particle ind, we do something different
        if data_key == 'particle_ind':

            n_particles_mask = sl.sum()

            if n_samples is None:
                n_particles_selected = n_particles_mask
            else:
                n_particles_selected = n_samples

            if n_samples >= n_particles_mask:
                print( "n_samples > n_particles_selected, not sampling." )
                n_particles_selected = n_particles_mask

            return np.tile(
                np.arange( n_particles_selected ),
                ( self.data_object.n_snaps, 1 ),
            ).transpose()
        
        # Get the masked data (masked via the slice)
        masked_data = self.data_object.get_processed_data(
            data_key,
            sl = sl,
            *args, **kwargs
        )

        # Sample a subset of the data
        if n_samples is not None:

            # If given a seed for sampling, use it
            if seed is not None:
                np.random.seed( seed )

            n_particles_selected = masked_data.shape[0]
            if n_samples >= n_particles_selected:
                print( "n_samples > n_particles_selected, not sampling." )
                return masked_data

            sampled_inds = np.random.choice(
                np.arange( n_particles_selected ),
                n_samples,
                replace = False,
            )
            masked_data = masked_data[sampled_inds,:]

        return masked_data





