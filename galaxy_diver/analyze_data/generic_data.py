#!/usr/bin/env python
'''Subclass for analyzing particle data.

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

# Base python imports
import copy
from functools import wraps
import h5py
import numpy as np
import numpy.testing as npt
import string
import warnings


# Imports from my own stuff
import galaxy_diver.read_data.ahf as read_ahf
import galaxy_diver.utils.astro as astro
import galaxy_diver.utils.constants as constants

########################################################################

class GenericData( object ):

  def __init__( self,
                data_dir = None,
                ahf_data_dir = None,
                snum = None,
                ahf_index = None,

                averaging_frac = 0.5,
                length_scale_used = 'r_scale',
                halo_data_retrieved = False,
                centered = False,
                vel_centered = False,
                hubble_corrected = False,

                z_sun = constants.Z_MASSFRAC_SUN,

                ahf_tag = 'smooth',
                main_halo_id = 0,
                center_method = 'halo',
                vel_center_method = 'halo',

                verbose = False,
                store_ahf_reader = False,

                **kwargs ):
    '''Initialize.

    Args:
      data_dir (str) : Directory the simulation is contained in.
      ahf_data_dir (str) : Directory simulation analysis is contained in. Defaults to data_dir
      snum (int or array of ints) : Snapshot or snapshots to inspect.
      ahf_index (str) : What to index the snapshots by. Should be the last snapshot in the simulation *if*
                                  AHF was run backwards from the last snapshot.
                                  Required to put in manually to avoid easy mistakes.

      averaging_frac (float): What fraction of the radius to average over when calculating velocity and
        similar properties? (centered on the origin)
      length_scale_used (str) : What length scale to use for the simulation. Will be used to put lengths in fractions.
        Options...
        'r_scale' : Scale radius.
        'R_vir' : Virial radius.
      halo_data_retrieved (bool) : Whether or not we retrieved relevant values from the AHF halo data.
      centered (bool): Whether or not the coordinates are centered on the galaxy of choice at the start.
      vel_centered (bool) : Whether or not the velocities are relative to the galaxy of choice at the start.
      hubble_corrected (bool) : Whether or not the velocities have had the Hubble flow added (velocities
                                          must be centered).

      z_sun (float) : Used mass fraction for solar metallicity.

      ahf_tag (str) : Identifying tag for the ahf merger tree halo files, looks for ahf files of type
                                'halo_00000_{}.dat'.format( tag ).
      main_halo_id (int) : What is the halo ID of the main galaxy in the simulation?
      center_method (str or np.array) : How to center the coordinates. Options...
        'halo' (default) : Centers the dataset on the main halo (main_halo_id) using AHF halo data.
        np.array : Array of coordinates on which to center the data
      vel_center_method (str or np.array of size 3) : How to center the velocity coordinates, i.e. what the
                                                                velocity is relative to. Options are...
        'halo' (default) : Sets velocity relative to the main halo (main_halo_id) using AHF halo data.
        np.array of size 3 : Centers the dataset on this coordinate.

      verbose (bool) : Print out additional information.

    Keyword Args:
      function_args (dict): Dictionary of args used to specify an arbitrary function with which to
        generate data.
    '''

    # Store the arguments
    for arg in locals().keys():
      setattr( self, arg, locals()[arg] )

    # Make sure that all the arguments have been specified.
    for attr in vars( self ).keys():
      if attr == 'kwargs':
        continue
      if getattr( self, attr ) is None:

        # Set the analysis dir to data_dir if not given
        if attr == 'ahf_data_dir':
          self.ahf_data_dir = self.data_dir

        elif attr == 'ahf_index':
          warnings.warn( "AHF index not specified. Will be unable to use halo finding data." )

        else:
          raise Exception( '{} not specified'.format( attr ) )

    # For storing and creating masks to pass the data
    self.data_masker = DataMasker( self )

    # By definition, the halo data should not be retrieved when the class is first initiated.
    self.halo_data_retrieved = False

    # Setup a data key parser
    self.key_parser = DataKeyParser()

  ########################################################################
  # Properties
  ########################################################################

  @property
  def length_scale( self ):
    '''Property for fiducial simulation length scale.'''

    if self.length_scale_used == 'R_vir':
      return self.r_vir
    else:
      return self.r_scale

  ########################################################################

  @property
  def velocity_scale( self ):
    '''Property for fiducial simulation velocity scale.'''

    return self.v_c

  ########################################################################

  @property
  def metallicity_scale( self ):
    '''Property for fiducial simulation metallicity scale.'''

    return self.z_sun

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

        test_value = np.array(value)[not_nan_inds] # Cast as np.ndarray because Pandas arrays can cause trouble.
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
      self._hubble_z = astro.hubble_parameter( self.redshift, h=self.data_attrs['hubble'],
        omega_matter=self.data_attrs['omega_matter'], omega_lambda=self.data_attrs['omega_lambda'], units='km/s/kpc' )

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

    self.data['V'] += self.get_data( 'P' )*self.hubble_z

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
      except KeyError, e:
        self.handle_data_key_error( data_key )
        continue

      break

    if 'data' not in locals().keys():
      raise KeyError( "After {} tries, unable to find or create data_key, {}".format( i+1, data_key ) )

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
      data = self.data['P'][0,:]
    elif data_key == 'Ry':
      data = self.data['P'][1,:]
    elif data_key == 'Rz':
      data = self.data['P'][2,:]
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
      data = self.data['V'][0,:]
    elif data_key == 'Vy':
      data = self.data['V'][1,:]
    elif data_key == 'Vz':
      data = self.data['V'][2,:]
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

    # GenericData methods
    if data_key == 'R':
      self.calc_radial_distance()
    elif data_key == 'Vr':
      self.calc_radial_velocity()
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
    elif data_key ==  'Rx' or data_key ==  'Ry' or data_key ==  'Rz' or data_key == 'P':
      self.calc_positions()
    elif data_key[:-3] ==  'Rx_face' or data_key[:-3] ==  'Ry_face' or data_key[:-3] ==  'Rz_face':
      self.calc_face_positions( data_key )
    elif data_key == 'T':
      self.calc_temp()
    elif data_key == 'M':
      self.calc_mass()

    else:
      raise KeyError( 'NULL data_key, data_key = {}'.format( data_key ) )

  ########################################################################

  def get_processed_data( self, data_key, sl=None ):
    '''Get post-processed data. (Accounting for fractions, log-space, etc.).'''

    # Account for fractional data keys
    data_key, fraction_flag = self.key_parser.is_fraction_key( data_key )

    # Account for logarithmic data
    data_key, log_flag = self.key_parser.is_log_key( data_key )

    # Get the data and make a copy to avoid altering
    data_original = self.get_data( data_key, sl=sl )
    data = copy.deepcopy( data_original )

    # Actually calculate the fractional data
    if fraction_flag:

      # Put distances in units of the virial radius
      if self.key_parser.is_position_key( data_key ):
        data /= self.length_scale

      # Put velocities in units of the circular velocity
      elif self.key_parser.is_velocity_key( data_key ):
        data /= self.velocity_scale

      # Put the metallicity in solar units
      elif data_key == 'Z':
        data /= self.metallicity_scale

      else:
        raise Exception('Fraction type not recognized')

    # Make appropriate units into log
    if log_flag:
      data =  np.log10( data )

    return data

  ########################################################################

  def shift( self, data, data_key ):
    '''Shift or multiply the data by some amount. Note that this is applied after logarithms are applied.

    data : data to be shifted
    data_key : Data key for the parameters to be shifted
    Parameters are a subdictionary of self.kwargs
    'data_key' : What data key the data is shifted for
    '''

    raise Exception( "TODO: Test this" )

    shift_p = self.kwargs['shift']

    # Exit early if not the right data key
    if shift_p['data_key'] != data_key:
      return 0

    # Shift by the mass metallicity relation.
    if 'MZR' in shift_p:

      # Gas-phase ISM MZR fit from Ma+2015.
      # Set to be no shift for gas with the mean metallicity of LLS at z=0 (This has a value of -0.45, as I last calculated)
      log_shift = 0.93*(np.exp(-0.43*self.redshift) - 1.) - 0.45

      data -= log_shift

  ########################################################################
  # Full calculations based on the data
  ########################################################################
  
  def calc_radial_distance(self):
    '''Calculate the distance from the origin for a given particle.'''

    self.data['R'] = np.sqrt( self.get_data( 'Rx' )**2. + self.get_data( 'Ry' )**2. + self.get_data( 'Rz' )**2. )

########################################################################
########################################################################

class SnapshotData( GenericData ):
  '''Class for analysis of a single snapshot of data.'''

  def __init__( self, *args, **kwargs ):

    super( SnapshotData, self ).__init__( *args, **kwargs )

  ########################################################################
  # Get Additional Data
  ########################################################################

  def retrieve_halo_data( self ):

    if self.halo_data_retrieved:
      return

    # Load the AHF data
    ahf_reader = read_ahf.AHFReader( self.ahf_data_dir )
    ahf_reader.get_mtree_halos( index=self.ahf_index, tag=self.ahf_tag )

    # Select the main halo at the right redshift
    mtree_halo = ahf_reader.mtree_halos[self.main_halo_id].loc[self.snum]

    # Add the halo data to the class.
    self.redshift = mtree_halo['redshift']
    halo_coords_comoving = np.array( [ mtree_halo['Xc'], mtree_halo['Yc'], mtree_halo['Zc'] ] )
    self.halo_coords = halo_coords_comoving/(1. + self.redshift)/self.data_attrs['hubble']
    self.halo_velocity = np.array( [ mtree_halo['VXc'], mtree_halo['VYc'], mtree_halo['VZc'] ] )
    self.r_vir = mtree_halo['Rvir']/(1. + self.redshift)/self.data_attrs['hubble']
    self.r_scale = self.r_vir/mtree_halo['cAnalytic']
    self.m_vir = mtree_halo['Mvir']/self.data_attrs['hubble']
    self.m_gas = mtree_halo['M_gas']/self.data_attrs['hubble']
    self.m_star = mtree_halo['M_star']/self.data_attrs['hubble']

    # Calculate the circular velocity
    self.v_c = astro.circular_velocity( self.r_vir, self.m_vir )

    self.halo_data_retrieved = True

    if self.store_ahf_reader:
      self.ahf_reader = ahf_reader

  ########################################################################
  # Properties
  ########################################################################

  @property
  def v_com( self ):
    '''Property for the velocity of the center of mass.'''

    if not hasattr( self, '_v_com' ):
      
      radial_mask = self.data_masker.mask_data( 'Rf', 0., self.averaging_frac, 'return' )
      
      m_ma = self.data_masker.get_masked_data( 'M', radial_mask )
      v_ma = self.data_masker.get_masked_data( 'V', radial_mask )

      self._v_com = ( v_ma*m_ma ).sum( 1 )/m_ma.sum()

    return self._v_com

  ########################################################################

  @property
  def total_ang_momentum(self):
    '''Calculate the total angular momentum vector.'''

    raise Exception( "TODO: Test and memoize this, and other attributes" )

    # Exit early if already calculated.
    try:
      self.total_ang_momentum
      return self.total_ang_momentum

    # Calculate the total angular momentum
    except AttributeError:

      # Make sure necessary ingredients are calculated
      if not self.halo_data_retrieved:
        self.retrieve_halo_data()
      if not self.hubble_corrected:
        self.correct_hubble_flow()

      # Get mask for only inner components
      r_mask = self.add_mask('R', 0., self.averaging_frac*self.R_vir, return_or_store='return')

      # Adapt for application to 'l', which is a multidimensional array
      inner_mask = np.array([r_mask]*3)

      # Apply masks
      ang_momentum = self.get_data('L')
      l_ma = np.ma.masked_array(ang_momentum, mask=inner_mask)

      # Get the total angular momentum
      self.total_ang_momentum = np.zeros(3)
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

    # Choose cosmology
    cosmo = Cosmology.setCosmology('WMAP9')

    # Calculate the virial radius, if necessary
    if R_vir is not None:
      if R_vir=='BN':
        # Calculate R_vir off of the cosmocode def, and convert to proper kpc.
        self.R_vir = cosmo.virialRadius(M, z)*10.**3. 

    # Make h easier to use (don't have to write the whole thing out...)
    h = self.data_attrs['hubble']

    Mh = h*self.M_vir # Convert the halo mass to Msun/h, so as to feed it into the HMF.
    dldz = np.abs(cosmo.line_elt(self.redshift)) # Cosmological line element in Mpc.
    dldz_kpc = dldz*10.**3.
    dndlog10M = cosmo.HMF(Mh, self.redshift)*self.M_vir*np.log(10)*h**4. # HMF in 1/Mpc^3
    dndlog10M_kpc = dndlog10M*10.**-9.
    dN_halo =  dldz_kpc*dndlog10M_kpc*np.pi*self.R_vir**2.

    # Convert from per redshift to per absorption path length.
    if time_units == 'abs_length':
      dN_halo /= cosmo.dXdz(z)
    elif time_units == 'redshift':
      pass

    return dN_halo

  ########################################################################
  # Full calculations of the data
  ########################################################################

  def calc_radial_velocity(self):
    '''Calculate the radial velocity.'''

    raise Exception( "TODO: Test this" )

    # Center velocity and radius
    self.change_coords_center()
    self.change_vel_coords_center()
    
    # Calculate the radial velocity
    self.data['Vr'] = (self.data['V']*self.get_data('P')).sum(0)/self.get_data('R')

  ########################################################################

  def calc_ang_momentum(self):
    '''Calculate the angular momentum.'''

    raise Exception( "TODO: Test this" )

    # Make sure we're centered.
    self.change_coords_center()
    self.change_vel_coords_center()

    # Calculate the angular momentum for each particle
    m_mult = np.array([self.get_data('M'),]*3)
    self.data['L'] = m_mult*np.cross(self.get_data('P'), self.get_data('V'), 0, 0).transpose()

  ########################################################################

  def calc_inds(self):
    '''Calculate the indices the data are located at, prior to any masks.'''

    raise Exception( "TODO: Test this" )

    # Flattened index array
    flat_inds = np.arange(self.get_data('Den').size)

    # Put into a multidimensional array
    self.data['ind'] = flat_inds.reshape(self.get_data('Den').shape)

  ########################################################################

  def calc_phi(self, vector='total gas ang momentum'):
    '''Calculate the angle (in degrees) from some vector.'''

    raise Exception( "TODO: Test this" )

    if vector == 'total ang momentum':
      # Calculate the total angular momentum vector, if it's not calculated yet
      self.v = self.calc_total_ang_momentum()
    elif vector == 'total gas ang momentum':
      p_d = ParticleData(self.kwargs)
      self.v = p_d.calc_total_ang_momentum()
    else:
      self.v = vector

    # Get the dot product
    P = self.get_data('P')
    dot_product = np.zeros(P[0,:].shape)
    for i in range(3):
      dot_product += self.v[i]*P[i,:]

    # Isolate for the cosine
    cos_phi = dot_product/self.get_data('R')/np.linalg.norm(self.v)

    # Get the angle (in degrees)
    self.data['Phi'] = np.arccos(cos_phi)*180./np.pi

  ########################################################################

  def calc_abs_phi(self, vector='total gas ang momentum'):
    '''Calculate the angle (in degrees) from some vector, but don't mirror it around 90 degrees (e.g. 135 -> 45 degrees, 180 -> 0 degrees).'''

    raise Exception( "TODO: Test this" )

    # Get the original Phi
    self.calc_phi(vector)

    self.data['AbsPhi'] = np.where(self.data['Phi'] < 90., self.data['Phi'], np.absolute(self.data['Phi'] - 180.))

  ########################################################################

  def calc_num_den(self):
    '''Calculate the number density (it's just a simple conversion...).'''

    raise Exception( "TODO: Test this" )

    self.data['NumDen'] = self.data['Den']*constants.gas_den_to_nb

  ########################################################################

  def calc_H_den(self):
    '''Calculate the H density in cgs (cm^-3). Assume the H fraction is ~0.75'''

    raise Exception( "TODO: Test this" )

    # Assume Hydrogen makes up 75% of the gas
    X_H = 0.75

    self.data['HDen'] = X_H*self.data['Den']*constants.gas_den_to_nb

  ########################################################################

  def calc_HI_den(self):
    '''Calculate the HI density in cgs (cm^-3).'''

    raise Exception( "TODO: Test this" )

    # Assume Hydrogen makes up 75% of the gas
    X_H = 0.75

    # Calculate the hydrogen density
    HDen = X_H*self.data['Den']*constants.gas_den_to_nb 

    self.data['HIDen'] = self.data['nHI']*HDen

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

    y_helium = self.data['Z_Species'][:,0] # Get the mass fraction of helium
    mu = 1./(1. - 0.75*y_helium + self.data['ne'])

    return mu


########################################################################
########################################################################

class TimeData( GenericData ):
  '''Class for analysis of a time series data, e.g. the worldlines of a number of particles.'''

  def __init__( self, *args, **kwargs ):
    '''
    Args:
      snums (array-like of ints) : Snapshots for the time series.
    '''

    super( TimeData, self ).__init__( *args, **kwargs )

  ########################################################################

  def retrieve_halo_data( self ):

    if self.halo_data_retrieved:
      return

    # Load the AHF data
    ahf_reader = read_ahf.AHFReader( self.ahf_data_dir )
    ahf_reader.get_mtree_halos( index=self.ahf_index, tag=self.ahf_tag )

    # Select the main halo at the right redshift
    mtree_halo = ahf_reader.mtree_halos[self.main_halo_id].loc[self.snum]

    # Add the halo data to the class.
    self.redshift = mtree_halo['redshift']
    scale_factor_and_hinv = 1./(1. + self.redshift)/self.data_attrs['hubble']

    halo_coords_comoving = np.array( [ mtree_halo['Xc'], mtree_halo['Yc'], mtree_halo['Zc'] ] )
    self.halo_coords = halo_coords_comoving*scale_factor_and_hinv[np.newaxis,:]
    self.halo_velocity = np.array( [ mtree_halo['VXc'], mtree_halo['VYc'], mtree_halo['VZc'] ] )
    self.r_vir = mtree_halo['Rvir']*scale_factor_and_hinv
    self.r_scale = self.r_vir/mtree_halo['cAnalytic']
    self.m_vir = mtree_halo['Mvir']/self.data_attrs['hubble']
    self.m_gas = mtree_halo['M_gas']/self.data_attrs['hubble']
    self.m_star = mtree_halo['M_star']/self.data_attrs['hubble']

    # Calculate the circular velocity
    self.v_c = astro.circular_velocity( self.r_vir, self.m_vir )

    self.halo_data_retrieved = True

    if self.store_ahf_reader:
      self.ahf_reader = ahf_reader

########################################################################
########################################################################

class DataKeyParser( object ):
  '''Class for parsing data_keys provided to GenericData.'''

  ########################################################################

  def is_position_key( self, data_key ):
    '''Checks if the data key deals with position primarily.

    Args:
      data_key (str) : Data key to check.
  
    Returns:
      is_position_key (bool) : True if deals with position.
    '''

    return ( data_key[0] == 'R' ) | ( data_key == 'P' )

  ########################################################################

  def is_velocity_key( self, data_key ):
    '''Checks if the data key deals with velocity primarily.

    Args:
      data_key (str) : Data key to check.
  
    Returns:
      is_velocity_key (bool) : True if deals with velocity.
    '''

    return ( data_key[0] == 'V' )


  ########################################################################

  def is_fraction_key( self, data_key ):
    '''Check if the data should be some fraction of its relevant scale.

    Args:
      data_key (str) : Data key to check.
  
    Returns:
      is_fraction_key (bool) : True if the data should be scaled as a fraction of the relevant scale.
    '''

    fraction_flag = False
    if data_key[-1] == 'f':
      data_key = data_key[:-1]
      fraction_flag = True

    return data_key, fraction_flag

  ########################################################################

  def is_log_key( self, data_key ):
    '''Check if the data key indicates the data should be put in log scale.

    Args:
      data_key (str) : Data key to check.
  
    Returns:
      is_log_key (bool) : True if the data should be taken log10 of
    '''

    log_flag = False
    if data_key[0:3] == 'log':
      data_key = data_key[3:]
      log_flag = True

    return data_key, log_flag

########################################################################
########################################################################

class DataMasker( object ):

  def __init__( self, generic_data ):
    '''Class for masking data.

    Args:
      generic_data (GenericData object) : Used for getting data to find mask ranges.
    '''

    self.generic_data = generic_data

    self.masks = []

  ########################################################################

  def mask_data( self, data_key, data_min, data_max, return_or_store='store' ):
    '''Get only the particle data within a certain range. Note that it retrieves the processed data.

    Args:
      data_key (str) : Data key to base the mask off of.
      data_min (float) : Everything below data_min will be masked.
      data_max (float) : Everything above data_max will be masked.
      return_or_store (str) : Whether to store the mask as part of the masks dictionary, or to return it.

    Returns:
      data_mask (np.array of bools) : If requested, the mask for data in that range.

    Modifies:
      self.masks (list of dicts) : Appends a dictionary describing the mask.
    '''

    # Get the mask
    data = self.generic_data.get_processed_data( data_key )
    data_ma = np.ma.masked_outside( data, data_min, data_max )

    # Handle the case where an entire array is masked or none of it is masked
    # (Make into an array for easier combination with other masks)
    if data_ma.mask.size == 1:
      data_ma.mask = data_ma.mask*np.ones( shape=data.shape, dtype=bool )

    if return_or_store == 'store':
      self.masks.append( {'data_key': data_key, 'data_min': data_min, 'data_max': data_max, 'mask': data_ma.mask} )

    elif return_or_store == 'return':
      return data_ma.mask

    else:
      raise Exception('NULL return_or_store')

  ########################################################################

  def get_total_mask( self ):
    '''Get the result of combining all masks in the data.

    Returns:
      total_mask (np.array of bools) : Result of all masks.
    '''

    # Compile masks
    all_masks = []
    for mask_dict in self.masks:
      all_masks.append( mask_dict['mask'] )

    # Combine masks
    return np.any( all_masks, axis=0, keepdims=True )[0]

  ########################################################################

  def get_masked_data( self, data_key, mask='total', sl=None, apply_slice_to_mask=True ):
    '''Get all the data that doesn't have some sort of mask applied to it. Use the processed data.

    Args:
      data_key (str) : Data key to get the data for.
      mask (str or np.array of bools) : Mask to apply. If none, use the total mask.
      sl (slice) : Slice to apply to the data
      apply_slice_to_mask (bool) : Whether or not to apply the same slice you applied to the data to the mask.

    Returns:
      data_ma (np.array) : Compressed masked data. Because it's compressed it may not have the same shape as the
        original data.
    '''
    data = self.generic_data.get_processed_data( data_key, sl=sl )

    # Get the appropriate mask
    if isinstance( mask, np.ndarray ):
      used_mask = mask
    elif isinstance( mask, bool ) or isinstance( mask, np.bool_ ):
      if not mask:
        return data
      raise Exception( "All data is masked." )
    elif mask == 'total':
      used_mask = self.get_total_mask()
    else:
      raise KeyError( "Unrecognized type of mask, {}".format( mask ) )

    if ( sl is not None ) and apply_slice_to_mask:
      used_mask = used_mask[sl]

    # Test for if the data fits the mask, or if it's multi-dimensional
    if len( data.shape ) > len( self.generic_data.base_data_shape ):
      data_ma = [ np.ma.array( data_part, mask=used_mask ) for data_part in data ]
      data_ma = [ data_ma_part.compressed() for data_ma_part in data_ma ]
      data_ma = np.array( data_ma )

    else:
      data_ma = np.ma.array( data, mask=used_mask )
      data_ma = data_ma.compressed()

    return data_ma

  ########################################################################

  def clear_masks( self ):
    '''Reset the masks in total to nothing.
    
    Modifies:
      self.masks (lists) : Sets to empty
    '''

    self.masks = []
