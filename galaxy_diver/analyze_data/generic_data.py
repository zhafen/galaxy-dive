#!/usr/bin/env python
'''Subclass for analyzing particle data.

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

# Base python imports
import copy
import h5py
import numpy as np
import numpy.testing as npt
import string


# Imports from my own stuff
import galaxy_diver.read_data.ahf as read_ahf
import galaxy_diver.utils.constants as constants
import galaxy_diver.utils.io as io

########################################################################

class GenericData( object ):

  def __init__( self,
                averaging_frac = 0.05,
                centered = False,
                vel_centered = False,
                hubble_corrected = False,
                z_sun = constants.Z_MASSFRAC_SUN,
                ahf_tag = 'smooth',
                main_mt_halo_id = 0,
                **kwargs ):
    '''Initialize.

    Args:
      averaging_frac (float, optional): What fraction of the radius to average over when calculating velocity and similar properties? (centered on the origin)
      centered (bool, optional): Whether or not the coordinates are centered on the galaxy of choice at the start.
      vel_centered (bool, optional) : Whether or not the velocities are relative to the galaxy of choice at the start.
      hubble_corrected (bool, optional) : Whether or not the velocities have had the Hubble flow added (velocities must be centered).
      z_sun (float, optional) : Used mass fraction for solar metallicity.
      ahf_tag (str, optional) : Identifying tag for the ahf merger tree halo files, looks for ahf files of type 'halo_00000_{}.dat'.format( tag ).
      main_mt_halo_id (int, optional) : What is the halo ID of the main galaxy in the simulation?

    Keyword Args:
      sdir (str, required) : Directory the simulation is contained in.
      snum (int, required) : Snapshot to inspect.
      ahf_index (str, required) : What to index the snapshots by. Should be the last snapshot in the simulation *if* AHF was run backwards from the last snapshot.
                                  Required to put in manually to avoid easy mistakes.

      analysis_dir (str, optional) : Directory simulation analysis is contained in. Defaults to sdir
      function_args (dict, optional): Dictionary of args used to specify an arbitrary function with which to generate data.
    '''

    # Store the arguments
    for arg in locals().keys():
      setattr( self, arg, locals()[arg] )

    # Set the analysis dir to sdir, if not given
    if 'analysis_dir' not in self.kwargs:
      self.kwargs['analysis_dir'] = self.kwargs['sdir']

    # For storing masks to look at the data through
    self.masks = []

    # Array for containing units
    self.units = {}

  ########################################################################

  def handle_data_key_error(self, data_key):
    '''
    data_key : key that produced the data key error
    '''

    raise Exception( "TODO: Test this" )

    print 'Data key not found in data. Attempting to calculate.'

    if data_key ==  'Rx' or data_key ==  'Ry' or data_key ==  'Rz' or data_key == 'P':
      self.calc_positions()
    elif data_key[:-3] ==  'Rx_face' or data_key[:-3] ==  'Ry_face' or data_key[:-3] ==  'Rz_face':
      self.calc_face_positions( data_key )
    elif data_key == 'R':
      self.calc_radial_distance()
    elif data_key == 'Vr':
      self.calc_radial_velocity()
    elif data_key == 'T':
      self.calc_temp()
    elif data_key == 'ind':
      self.calc_inds()
    elif data_key == 'L':
      self.calc_ang_momentum()
    elif data_key == 'Phi':
      self.calc_phi()
    elif data_key == 'AbsPhi':
      self.calc_abs_phi()
    elif data_key == 'M':
      self.calc_mass()
    elif data_key == 'NumDen':
      self.calc_num_den()
    elif data_key == 'HDen':
      self.calc_H_den()
    elif data_key == 'HIDen':
      self.calc_HI_den()
    elif data_key == 'Cl':
      self.calc_classifications()
    else:
      raise Exception('NULL data_key, data_key = {}'.format(data_key))

  ########################################################################
  # Get Additional Data
  ########################################################################

  # Get the halo data out
  def retrieve_halo_data( self ):

    # Load the AHF data
    ahf_reader = read_ahf.AHFReader( self.kwargs['analysis_dir'] )
    ahf_reader.get_mtree_halos( index=self.kwargs['ahf_index'], tag=self.ahf_tag )

    # Select the main halo at the right redshift
    mtree_halo = ahf_reader.mtree_halos[self.main_mt_halo_id].loc[self.kwargs['snum']]

    ## Get the halo number
    #if 'halo_number' in self.kwargs:
    #  self.halo_number = self.kwargs['halo_number']
    ## Assume that the halo files are as default if the halo number isn't given explicitely
    #else: 
    #  sim_name = string.split(self.kwargs['sdir'], '/')[-1]
    #  sims_w_halo_1 = ['B1_hr_Dec5_2013_11',]
    #  if sim_name in sims_w_halo_1:
    #    self.halo_number = 1
    #  else:
    #    self.halo_number = 0

    ## Get the halo data
    #halo_data = io.getHaloDataRedshift(self.kwargs['sdir'], self.halo_number, self.data_attrs['redshift'])

    ## Get rid of 1/h factors in the halo data
    #vals_to_be_converted = range(3, 13) 
    #for val_index in vals_to_be_converted:
    #  halo_data[val_index] /= self.data_attrs['hubble']

    # Add the halo data to the class.
    self.redshift = mtree_halo['redshift']
    self.halo_coords = np.array( [ mtree_halo['Xc'], mtree_halo['Yc'], mtree_halo['Zc'] ] )/(1. + self.redshift)/self.data_attrs['hubble']
    self.R_vir = mtree_halo['Rvir']/(1. + self.redshift)/self.data_attrs['hubble']
    self.M_vir = mtree_halo['Mvir']/self.data_attrs['hubble']
    self.M_gas = mtree_halo['M_gas']/self.data_attrs['hubble']
    self.M_star = mtree_halo['M_star']/self.data_attrs['hubble']

    if 'redshift' in self.data_attrs:
      npt.assert_allclose( self.redshift, self.data_attrs['redshift'] )

    # Calculate the circular velocity
    self.v_c = np.sqrt(constants.G*self.M_vir*constants.Msun_to_kg / (self.R_vir*constants.kpc_to_km*1.e9))

  ########################################################################

  def get_header_values(self):
    '''Get some overall values for the snapshot.'''

    raise Exception( "TODO: Test this/get rid of it" )

    header = read_snapshot.readsnap(self.kwargs['sdir'], self.kwargs['snum'], 0, cosmological=1, header_only=1)

    self.k = header['k']
    self.redshift = header['redshift']
    self.time = header['time']
    self.hubble = header['hubble']

  ########################################################################
  # Calculate simple net values of the simulation
  ########################################################################

  def calc_total_ang_momentum(self):
    '''Calculate the total angular momentum vector.'''

    raise Exception( "TODO: Test this" )

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
      r_mask = self.add_mask('R', 0., self.averaging_frac*self.R_vir, return_or_add='return')

      # Adapt for application to 'l', which is a multidimensional array
      inner_mask = np.array([r_mask]*3)

      # Apply masks
      ang_momentum = self.get_data('L')
      l_ma = np.ma.masked_array(ang_momentum, mask=inner_mask)

      # Get the total angular momentum
      self.total_ang_momentum = np.zeros(3)
      for i in range(3):
        self.total_ang_momentum[i] = l_ma[i].sum()

      return self.total_ang_momentum

  ########################################################################

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
        self.R_vir = cosmo.virialRadius(M, z)*10.**3. # Calculate R_vir off of the cosmocode def, and convert to proper kpc.

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
  # Decorators for returning the data
  ########################################################################

  def correct_hubble_flow(self):
    '''Correct for hubble flow movement.'''

    raise Exception( "TODO: Test this" )

    # Make sure we're centered
    self.change_coords_center()

    # Hubble constant
    H0 = self.data_attrs['hubble']*100.
    Hz = H0*np.sqrt(self.data_attrs['omega_matter'] * (1.0 + self.data_attrs['redshift'])**3. + self.data_attrs['omega_lambda'])
    Hz /= 1000. # Convert from km/s / Mpc to km/s / kpc

    self.data['V'] += self.get_data('P')*Hz

    self.hubble_corrected = True

  ########################################################################

  def change_coords_center( self, center_method='halo' ):
    '''Change the location of the origin.
    '''

    raise Exception( "TODO: Test this, turn into a decorator" )

    # Make sure not to change the coords multiple times
    if self.centered:
      return 0

    if center_method == 'halo':
      if not self.halo_data_retrieved:
        self.retrieve_halo_data()
    else:
      self.peak_coords = center_method

    self.data['P'][0] -= self.peak_coords[0]
    self.data['P'][1] -= self.peak_coords[1]
    self.data['P'][2] -= self.peak_coords[2]

    self.centered = True

  ########################################################################

  def change_vel_coords_center(self):
    '''Get velocity coordinates to center on the main halo.'''

    raise Exception( "TODO: Test this" )

    if self.vel_centered:
      return 0

    # Make sure necessary ingredients are calculated
    if self.halo_data_retrieved == False:
      self.retrieve_halo_data()
    if self.hubble_corrected == False:
      self.correct_hubble_flow()

    # Get mask for only inner components
    r_mask = self.add_mask('R', 0., self.averaging_frac*self.R_vir, return_or_add='return')

    # Adapt for application to 'v', which is a multidimensional array
    inner_mask = np.array([r_mask]*3)

    # Get mass array to multiply by
    m_ma = np.ma.masked_array(self.get_data('M'), mask=r_mask)
    m_ma_mult = np.array([m_ma]*3)

    # Apply masks
    v_ma = np.ma.masked_array(self.data['V'], mask=inner_mask)

    # Get the average velocity
    weighted_v = m_ma_mult*v_ma
    self.v_center = np.zeros(3)
    for i in range(3):
      self.v_center[i] = weighted_v[i,:].sum()
    self.v_center /= m_ma.sum()

    # Apply the average velocity
    self.data['V'][0] -= self.v_center[0]
    self.data['V'][1] -= self.v_center[1]
    self.data['V'][2] -= self.v_center[2]

    self.vel_centered = True

  ########################################################################
  # Complicated results of data
  ########################################################################

  def calc_radial_distance(self):
    '''Calculate the distance from the origin for a given particle.'''

    raise Exception( "TODO: Test this" )

    # Make sure we're centered
    self.change_coords_center()

    self.data['R'] = np.sqrt(self.get_data('Rx')**2. + self.get_data('Ry')**2. + self.get_data('Rz')**2.)

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

    raise Exception( "TODO: Test this" )

    y_helium = self.data['Z_Species'][:,0] # Get the mass fraction of helium
    mu = 1./(1. - 0.75*y_helium + self.data['ne'])

    return mu

  ########################################################################
  # Get Data
  ########################################################################

  def get_data(self, data_key):
    '''Get the data from within the class. Only for getting the data. No post-processing or changing the data (putting it in particular units, etc.)
    The idea is to calculate necessary quantities as the need arises, hence a whole function for getting the data.

    Args:
      data_key (str) : Key in the data dictionary for the key we want to get

    Returns:
      data
    '''

    raise Exception( "TODO: Test this" )

    # Loop through, handling issues
    n_tries = 10
    for i in range( n_tries ):
      try:

        if data_key[0] == 'R':

          # Center our position
          self.change_coords_center()

          # Transpose in order to account for when the data isn't regularly shaped
          if data_key == 'Rx':
            data = self.data['P'][0,:]
          elif data_key == 'Ry':
            data = self.data['P'][1,:]
          elif data_key == 'Rz':
            data = self.data['P'][2,:]
          else:
            data = self.data[data_key]

        # Velocities
        elif data_key[0] == 'V':

          # Center velocity
          self.change_vel_coords_center()

          # Get data
          if data_key == 'Vx':
            data = self.data['V'][0,:]
          if data_key == 'Vy':
            data = self.data['V'][1,:]
          if data_key == 'Vz':
            data = self.data['V'][2,:]
          else:
            data = self.data[data_key]

        # Arbitrary functions of the data
        elif data_key == 'Function':

          raise Exception( "TODO: Test this" )

          # Use the keys to get the data
          input_data = [ self.get_data(function_data_key) for function_data_key in self.kwargs['function_args']['function_data_keys'] ]
          
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
  
    return data

  ########################################################################

  def get_processed_data(self, data_key):
    '''Get post-processed data. (Accounting for fractions, log-space, etc.).'''

    raise Exception( "TODO: Test this" )

    # Account for fractional data keys
    fraction_flag = False
    if data_key[-1] == 'f':
      data_key = data_key[:-1]
      fraction_flag = True

    # Account for logarithmic data
    log_flag = False
    if data_key[0:3] == 'log':
      data_key = data_key[3:]
      log_flag = True

    # Get the data and make a copy to avoid altering
    data_original = self.get_data(data_key)
    data = copy.deepcopy(data_original)

    # Actually calculate the fractional data
    if fraction_flag:

      # Get halo data, if not retrieved
      if not self.halo_data_retrieved:
        self.retrieve_halo_data()

      # Put distances in units of the virial radius
      if data_key[0] == 'R':
        data /= self.R_vir

      # Put velocities in units of the circular velocity
      elif data_key[0] == 'V':
        data /= self.v_c

      else:
        raise Exception('Fraction type not recognized')

    # Put the metallicity in solar units
    if data_key == 'Z':
      if 'Z' in self.units:
        if self.units['Z'] == 'solar':
          pass
        else:
          raise Exception ("NULL units for data_key={}".format(data_key))
      else:
        data /= self.z_sun

    # Make appropriate units into log
    non_log_keys = ['P', 'R', 'Rx', 'Ry', 'Rz', 'Rho', 'V', 'Vr', 'Vx', 'Vy', 'Vz', 'h', 'Phi', 'AbsPhi', 'Cl']
    if data_key in non_log_keys:
      if log_flag:
        data =  np.log10(data)
      else:
        pass
    else:
      data =  np.log10(data)

    # Shift or multiply the data by some amount
    if 'shift' in self.kwargs:
      self.shift(data, data_key)

    return data

  ########################################################################

  def shift(self, data, data_key):
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
  # For Masking Data
  ########################################################################

  def add_mask(self, data_key, data_min, data_max, return_or_add='add'):
    '''Get only the particle data within a certain range. Note that it retrieves the processed data.'''

    raise Exception( "TODO: Test this" )

    # Get the mask
    data = self.get_processed_data(data_key)
    data_ma = np.ma.masked_outside(data, data_min, data_max)

    # Handle the case where an entire array is masked or none of it is masked
    # (Make into an array for easier combination with other masks)
    if data_ma.mask.size == 1:
      data_ma.mask = data_ma.mask*np.ones(shape=data.shape, dtype=bool)

    if return_or_add == 'add':
      self.masks.append({'data_key': data_key, 'data_min': data_min, 'data_max': data_max, 'mask': data_ma.mask})
    elif return_or_add == 'return':
      return data_ma.mask
    else:
      raise Exception('NULL return_or_add')

  ########################################################################

  def get_total_mask(self):

    raise Exception( "TODO: Test this" )

    # Compile masks
    all_masks = []
    for mask_dict in self.masks:
      all_masks.append(mask_dict['mask'])

    # Combine masks
    return np.any(all_masks, axis=0, keepdims=True)[0]

  ########################################################################

  def get_masked_data_for_all_masks(self, data_key):
    '''Wrapper for compliance w/ old scripts.'''

    raise Exception( "TODO: Test this" )

    return self.get_masked_data(data_key, mask='total')

  ########################################################################

  def get_masked_data(self, data_key, mask='total'):
    '''Get all the data that doesn't have some sort of mask applied to it. Use the processed data.'''

    raise Exception( "TODO: Test this" )

    # Get the appropriate mask
    if mask == 'total':
      tot_mask = self.get_total_mask()
    else:
      tot_mask = mask

    data = self.get_processed_data(data_key)
    data_ma = np.ma.array(data, mask=tot_mask)

    return data_ma.compressed()

