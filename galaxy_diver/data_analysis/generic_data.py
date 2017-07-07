'''
Name: gas_data.py
Author: Zachary Hafen (zachary.h.hafen@gmail.com)
Purpose: Base class for dealing with gas data from a zoom-simulation
          Note: A number of previously written scripts don't really use this...
'''

# Base python imports
import copy
import h5py
import numpy as np
import string


# Imports from my own stuff
import galaxy_diver.utils.constants as constants
import galaxy_diver.utils.dataio as dataio
import galaxy_diver.read_data.read_snapshot as read_snapshot

########################################################################

class GasData(object):

  def __init__(self, data_p, **kw_args):
    '''
    data_p : Parameters specifying the gridded snapshot file. Depends on the particle type
    '''

    self.data_p = data_p

    # For storing masks to look at the data through
    self.masks = []

    # What fraction of the radius to average over when calculating velocity and similar properties? (centered on the origin)
    self.averaging_frac=0.05

    # State that we assume the data isn't centered at the start
    self.centered = False

    # State that we don't center the particles on the target galaxy in position or velocity space
    self.vel_centered = False

    # State that the halo data isn't stored
    self.halo_data_retrieved = False

    # Note whether or not the data has been corrected for the hubble flow
    self.hubble_corrected = False

    # Default solar metallicity (Aspund+09)
    self.Z_sun = 0.014

    # Array for containing units
    self.units = {}

  ########################################################################

  def handle_data_key_error(self, data_key):
    '''
    data_key : key that produced the data key error
    '''

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

  def retrieve_halo_data(self):

    # Get the halo number
    if 'halo_number' in self.data_p:
      self.halo_number = self.data_p['halo_number']
    # Assume that the halo files are as default if the halo number isn't given explicitely
    else: 
      sim_name = string.split(self.data_p['sdir'], '/')[-1]
      sims_w_halo_1 = ['B1_hr_Dec5_2013_11',]
      if sim_name in sims_w_halo_1:
        self.halo_number = 1
      else:
        self.halo_number = 0

    # Get the halo data
    halo_data = dataio.getHaloDataRedshift(self.data_p['sdir'], self.halo_number, self.data_attrs['redshift'])

    # Get rid of 1/h factors in the halo data
    vals_to_be_converted = range(3, 13) 
    for val_index in vals_to_be_converted:
      halo_data[val_index] /= self.data_attrs['hubble']

    # Add the halo data to the class.
    self.redshift = halo_data[0]
    self.halo_ID = halo_data[1]
    self.host_ID = halo_data[2]
    self.peak_coords = (halo_data[3], halo_data[4], halo_data[5])
    self.R_vir = halo_data[6]
    self.M_vir = halo_data[7]
    self.M_gas = halo_data[8]
    self.M_star = halo_data[9]
    self.CoM_coords = (halo_data[10], halo_data[11], halo_data[12]) 

    # Calculate the circular velocity
    self.v_c = np.sqrt(constants.G*self.M_vir*constants.Msun_to_kg / (self.R_vir*constants.kpc_to_km*1.e9))

    self.halo_data_retrieved = True

  ########################################################################

  def get_header_values(self):
    '''Get some overall values for the snapshot.'''

    header = read_snapshot.readsnap(self.data_p['sdir'], self.data_p['snum'], 0, cosmological=1, header_only=1)

    self.k = header['k']
    self.redshift = header['redshift']
    self.time = header['time']
    self.hubble = header['hubble']

  ########################################################################
  # Calculate simple net values of the simulation
  ########################################################################


  def calc_total_ang_momentum(self):
    '''Calculate the total angular momentum vector.'''

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
  # Internal changes to the data
  ########################################################################

  def correct_hubble_flow(self):
    '''Correct for hubble flow movement.'''

    # Make sure we're centered
    self.change_coords_center()

    # Hubble constant
    H0 = self.data_attrs['hubble']*100.
    Hz = H0*np.sqrt(self.data_attrs['omega_matter'] * (1.0 + self.data_attrs['redshift'])**3. + self.data_attrs['omega_lambda'])
    Hz /= 1000. # Convert from km/s / Mpc to km/s / kpc

    self.data['V'] += self.get_data('P')*Hz

    self.hubble_corrected = True

  ########################################################################

  def change_coords_center(self, center_method='halo'):
    '''Change the location of the origin to center on the main halo.'''

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

  def calc_radial_distance(self):
    '''Calculate the distance from the origin for a given particle.'''

    # Make sure we're centered
    self.change_coords_center()

    self.data['R'] = np.sqrt(self.get_data('Rx')**2. + self.get_data('Ry')**2. + self.get_data('Rz')**2.)

  ########################################################################

  def calc_radial_velocity(self):
    '''Calculate the radial velocity.'''

    # Center velocity and radius
    self.change_coords_center()
    self.change_vel_coords_center()
    
    # Calculate the radial velocity
    self.data['Vr'] = (self.data['V']*self.get_data('P')).sum(0)/self.get_data('R')

  ########################################################################

  def calc_ang_momentum(self):
    '''Calculate the angular momentum.'''

    # Make sure we're centered.
    self.change_coords_center()
    self.change_vel_coords_center()

    # Calculate the angular momentum for each particle
    m_mult = np.array([self.get_data('M'),]*3)
    self.data['L'] = m_mult*np.cross(self.get_data('P'), self.get_data('V'), 0, 0).transpose()

  ########################################################################

  def calc_inds(self):
    '''Calculate the indices the data are located at, prior to any masks.'''

    # Flattened index array
    flat_inds = np.arange(self.get_data('Den').size)

    # Put into a multidimensional array
    self.data['ind'] = flat_inds.reshape(self.get_data('Den').shape)

  ########################################################################

  def calc_phi(self, vector='total gas ang momentum'):
    '''Calculate the angle (in degrees) from some vector.'''

    if vector == 'total ang momentum':
      # Calculate the total angular momentum vector, if it's not calculated yet
      self.v = self.calc_total_ang_momentum()
    elif vector == 'total gas ang momentum':
      p_d = ParticleData(self.data_p)
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

    # Get the original Phi
    self.calc_phi(vector)

    self.data['AbsPhi'] = np.where(self.data['Phi'] < 90., self.data['Phi'], np.absolute(self.data['Phi'] - 180.))

  ########################################################################

  def calc_num_den(self):
    '''Calculate the number density (it's just a simple conversion...).'''

    self.data['NumDen'] = self.data['Den']*constants.gas_den_to_nb

  ########################################################################

  def calc_H_den(self):
    '''Calculate the H density in cgs (cm^-3). Assume the H fraction is ~0.75'''

    # Assume Hydrogen makes up 75% of the gas
    X_H = 0.75

    self.data['HDen'] = X_H*self.data['Den']*constants.gas_den_to_nb

  ########################################################################

  def calc_HI_den(self):
    '''Calculate the HI density in cgs (cm^-3).'''

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
  # Get Data
  ########################################################################

  def get_data(self, data_key):
    '''Get the data from within the class. Only for getting the data. No post-processing or changing the data (putting it in particular units, etc.)
    The idea is to calculate necessary quantities as the need arises, hence a whole function for getting the data.'''

    # Loop through, handling issues
    for i in range(50):
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

          # Use the keys to get the data
          input_data = [self.get_data(function_data_key) for function_data_key in self.data_p['function_p']['function_data_keys']]
          
          # Apply the function
          data = self.data_p['function_p']['function'](input_data)

        # Other
        else:
          data = self.data[data_key]

      # Calculate missing data
      except KeyError, e:
        self.handle_data_key_error(data_key)
        continue

      break
  
    return data

  ########################################################################

  def get_processed_data(self, data_key):
    '''Get post-processed data. (Accounting for fractions, log-space, etc.).'''

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
        data /= self.Z_sun

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
    if 'shift' in self.data_p:
      self.shift(data, data_key)

    return data

  ########################################################################

  def shift(self, data, data_key):
    '''Shift or multiply the data by some amount. Note that this is applied after logarithms are applied.

    data : data to be shifted
    data_key : Data key for the parameters to be shifted
    Parameters are a subdictionariy of data_p
    'data_key' : What data key the data is shifted for
    '''

    shift_p = self.data_p['shift']

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

    # Compile masks
    all_masks = []
    for mask_dict in self.masks:
      all_masks.append(mask_dict['mask'])

    # Combine masks
    return np.any(all_masks, axis=0, keepdims=True)[0]

  ########################################################################

  def get_masked_data_for_all_masks(self, data_key):
    '''Wrapper for compliance w/ old scripts.'''

    return self.get_masked_data(data_key, mask='total')

  ########################################################################

  def get_masked_data(self, data_key, mask='total'):
    '''Get all the data that doesn't have some sort of mask applied to it. Use the processed data.'''

    # Get the appropriate mask
    if mask == 'total':
      tot_mask = self.get_total_mask()
    else:
      tot_mask = mask

    data = self.get_processed_data(data_key)
    data_ma = np.ma.array(data, mask=tot_mask)

    return data_ma.compressed()

########################################################################
# SubClasses
########################################################################

class LOSData(GasData):

  def __init__(self, data_p, **kw_args):
    '''Class for data that's a number of lines of sight in a grid shap.

    data_p : Parameters specifying the gridded snapshot file. Includes...
    "    "['sdir'] : Directory the snapshot is stored in
    "    "['snum'] : Snapshot number
    "    "['Nx'] : Number of grid cells on a side
    "    "['gridsize'] : How large the grid is
    "    "['ionized'] : How the ionization state is calculated.
                        'R13' : Ionization state calculated using Rahmati+13 fitting function
         "['comp_method'] : How the absorber component is calculated.
                            'all' : Entire LOS
                            'vshape' : Strongest absorber component in velocity space
                            'shape' : Strongest absorber component in physical space
    '''

    super(LOSData, self).__init__(data_p, **kw_args)

    self.retrieve_data()

    # State that we assume the data is centered at the start
    self.centered = True

    # State that we assume the data is centered at the start
    self.vel_centered = True

    # State that the data has already had the hubble flow accounted for
    self.hubble_corrected = True

    # Note units as necessary
    self.units['Z'] = 'solar'

  ########################################################################

  def retrieve_data(self):

    # Open file
    self.LOS_data_file_name = dataio.getLOSDataFilename(self.data_p['sdir'], self.data_p['Nx'], self.data_p['gridsize'], self.data_p['face'], \
                                                            self.data_p['comp_method'], self.data_p['ionized'], self.data_p['den_weight'])
    f = h5py.File(self.LOS_data_file_name, 'r')
    snapshot_data = f[str(self.data_p['snum'])]

    # Get the grid attributes
    self.data_attrs = {}
    for key in snapshot_data.attrs.keys():
      self.data_attrs[key] = snapshot_data.attrs[key]

    # Add the line of sight width (or cell length) to the data attributes
    self.data_attrs['cell_length'] = self.data_attrs['gridsize']/self.data_p['Nx']

    # Get the data
    self.data = {}
    for key in snapshot_data.keys():

      # Parse the keys and put in a more general format
      # All units should be the standard GIZMO output units
      # Gas Density
      if key == 'LOSDen':
        self.data['Den'] = snapshot_data[key][...]
      # Gas Neutral Hydrogen density
      elif key == 'LOSNHI':
        self.data['HIDen'] = snapshot_data[key][...]
      # Gas Neutral Hydrogen column density
      elif key == 'LOSHI':
        self.data['HI'] = snapshot_data[key][...]
      # Gas Hydrogen column density
      elif key == 'LOSH':
        self.data['H'] = snapshot_data[key][...]
      # Star Formation Rate
      elif key == 'LOSSFR':
        self.data['SFR'] = snapshot_data[key][...]
      # Temperature
      elif key == 'LOST':
        self.data['T'] = snapshot_data[key][...]
      # X Distance
      elif key == 'LOSRx':
        x = snapshot_data[key][...]
      # Y Distance
      elif key == 'LOSRy':
        y = snapshot_data[key][...]
      # Z Distance
      elif key == 'LOSRz':
        z = snapshot_data[key][...]
      # Radial Distance
      elif key == 'LOSR':
        self.data['R'] = snapshot_data[key][...]
      # Projected Distance
      elif key == 'LOSRr':
        self.data['Rho'] = snapshot_data[key][...]
      # X Velocity
      elif key == 'LOSVx':
        vx = snapshot_data[key][...]
      # Y Velocity
      elif key == 'LOSVy':
        vy = snapshot_data[key][...]
      # Z Velocity
      elif key == 'LOSVz':
        vz = snapshot_data[key][...]
      # Radial velocity
      elif key == 'LOSVr':
        self.data['Vr'] = snapshot_data[key][...]
      # Metallicity. Note that it's already in solar units in its main form, so we need to note that.
      elif key == 'LOSZ':
        self.data['Z'] = snapshot_data[key][...]
      # Standard deviation of the Metallicity (of the grid cells making up the LOS)
      elif key == 'LOSstdZ':
        self.data['StdZ'] = snapshot_data[key][...]
      # Log Metallicity (and averaged that way)
      elif key == 'LOSlogZ':
        self.data['LogZ'] = snapshot_data[key][...]
      # Standard deviation of Log Metallicity (of the grid cells making up the LOS)
      elif key == 'LOSstdlogZ':
        self.data['StdLogZ'] = snapshot_data[key][...]
      else:
        raise Exception('NULL key, key={}'.format(key))

    # Finish organizing velocity and position
    self.data['P'] = np.array([x, y, z])
    self.data['V'] = np.array([vx, vy, vz])

  ########################################################################

  def covering_fraction(self):
    '''Calculate the covering fraction for data satisfying all criteria.'''

    # Get the number of data fulfilling all criteria
    full_criteria_mask = self.get_total_mask()
    num_valid_data = float(np.invert(full_criteria_mask).sum())
  
    # Get the number of data fulfilling just the radius requirements
    num_impact_parameter_masks = 0
    for mask in self.masks:

      if mask['data_key'] == 'Rho':
        tot_mask = mask['mask']
        num_all_data = float(np.invert(tot_mask).sum())
        num_impact_parameter_masks += 1

      elif mask['data_key'] == 'Rhof':
        tot_mask = mask['mask']
        num_all_data = float(np.invert(tot_mask).sum())
        num_impact_parameter_masks += 1

    # Check that something didn't go wrong when getting the impact parameter masks
    if num_impact_parameter_masks != 1:
      raise Exception('num_impact_parameter_masks != 1, num_impact_parameter_masks = {}'.format(num_impact_parameter_masks))

    # Return the covering fraction
    self.f_cov = num_valid_data/num_all_data
    return self.f_cov

  ########################################################################

  def calc_mass(self):
    '''Calculate the mass for LOS cells.'''

    # Use the total H column density, and convert to total gas mass
    self.data['M'] = self.get_data('H')/0.75*constants.M_P
    self.data['M'] *= (self.data_attrs['cell_length']*constants.kpc_to_cm)**2.
    self.data['M'] /= constants.Msun_to_kg*1000. # Finish converting

########################################################################
########################################################################

class GriddedData(GasData):

  def __init__(self, data_p, **kw_args):
    '''
    data_p : Parameters specifying the gridded snapshot file. Must include...
    "    "['sdir'] : Directory the snaphost is stored in
    "    "['snum'] : Snapshot number
    "    "['Nx'] : Number of grid cells on a side
    "    "['gridsize'] : How large the grid is
    "    "['ionized'] : Legacy name. Originally how the ionization state is calculated. 
                        Currently any identifying string added to the end of the filename.
                        'R13' : Ionization state calculated using Rahmati+13 fitting function
    "    "['ion_grid'] : Whether or not this is a grid containing ion information.
    '''

    super(GriddedData, self).__init__(data_p, **kw_args)

    # Check if we have an ion grid
    if 'ion_grid' in data_p:
      self.is_ion_grid = data_p['ion_grid']
    else:
      self.is_ion_grid = False

    self.retrieve_data()

    # State that we assume the grid is centered at the start
    self.centered = True

  ########################################################################

  def retrieve_data(self):

    # Open file
    self.grid_file_name = dataio.getGridFilename(self.data_p['sdir'], self.data_p['snum'], self.data_p['Nx'], self.data_p['gridsize'], self.data_p['ionized'], \
                          ion_grid=self.is_ion_grid)
    f = h5py.File(self.grid_file_name, 'r')

    # Get the grid attributes
    self.data_attrs = {}
    for key in f.attrs.keys():
      self.data_attrs[key] = f.attrs[key]

    def load_data(f):

      # Get the data
      self.data = {}
      for key in f.keys():

        # Parse the keys and put in a more general format
        # All units should be the standard GIZMO output units
        # Gas Density
        if key == 'GasDen':
          self.data['Den'] = f[key][...]
        # Gas Neutral Hydrogen fraction
        elif key == 'GasNHI':
          self.data['nHI'] = f[key][...]
        # Should be the mean free electron number per proton
        elif key == 'GasNe':
          self.data['ne'] = f[key][...]
        # Same as above, but due to helium
        elif key == 'GasNe_He':
          self.data['ne_He'] = f[key][...]
        # Star Formation Rate
        elif key == 'GasSFR':
          self.data['SFR'] = f[key][...]
        # Temperature
        elif key == 'GasT':
          self.data['T'] = f[key][...]
        # X Velocity
        elif key == 'GasVx':
          vx = f[key][...]
        # Y Velocity
        elif key == 'GasVy':
          vy = f[key][...]
        # Z Velocity
        elif key == 'GasVz':
          vz = f[key][...]
        # Metallicity
        elif key == 'GasZ':
          self.data['Z'] = f[key][...]
        else:
          raise Exception('NULL key')

      # Finish organizing velocity
      self.data['V'] = np.array([vx, vy, vz])

    def load_ion_data(f):

      self.data = {}
      for key in f.keys():
        self.data[key] = f[key][...]

    # Load the data
    if not self.is_ion_grid:
      load_data(f)
    elif self.is_ion_grid:
      load_ion_data(f)
    else:
      raise Exception('Unrecognized ion_grid')

    f.close()

  ########################################################################

  def calc_positions(self):
    '''Calculate the positions of gridcells.'''

    # Get how the spacing on one side
    Nx = self.data.values()[0].shape[0]
    gridsize = self.data_attrs['gridsize']
    pos_coords = np.linspace(-gridsize/2., gridsize/2., Nx)

    # Mesh the spacing together for all the grid cells
    x, y, z = np.meshgrid(pos_coords, pos_coords, pos_coords, indexing='ij')

    self.data['P'] = np.array([x, y, z])

  ########################################################################

  def calc_face_positions(self, data_key):
    '''Calculate positions if you're just looking at one face of a grid.'''

    # Figure out which face to calculate for
    target_face = string.split( data_key, '_' )[-1]

    if target_face == 'xy':
      self.data['Rx_face_xy'] = self.get_data( 'Rx' )[:, :, 0]
      self.data['Ry_face_xy'] = self.get_data( 'Ry' )[:, :, 0]
    if target_face == 'xz':
      self.data['Rx_face_xz'] = self.get_data( 'Rx' )[:, 0, :]
      self.data['Rz_face_xz'] = self.get_data( 'Rz' )[:, 0, :]
    if target_face == 'yz':
      self.data['Ry_face_yz'] = self.get_data( 'Ry' )[0, :, :]
      self.data['Rz_face_yz'] = self.get_data( 'Rz' )[0, :, :]


  ########################################################################

  def calc_mass(self):
    '''Calculate the mass per grid cell.'''

    # Calculate the mass per grid cell, for easy use later
    self.data['M'] = self.data['Den']*(self.data_attrs['gridsize']/self.data_attrs['Nx'])**3.

  ########################################################################

  def calc_column_density(self, key, face):
    '''Calculate the column density for a targeted key.

    Args --
    key (str) : The data to project.
    face (int) : The face to use. 0=yz, 1=xz, 2=xy.

    Returns --
    key_col_den (np.array) : The column density in {key units}/cm^2. Assumes the grid uses fiducial GIZMO units.
    '''

    key_data = self.get_data( key )

    # Get the projected data
    summed_key_data = key_data.sum(face)

    # Get the column density
    self.data_attrs['cell_length'] = self.data_attrs['gridsize']/float( self.data_attrs['Nx'] - 1 )
    dx = self.data_attrs['cell_length']*3.086e21
    key_col_den = dx*summed_key_data

    return key_col_den

########################################################################
########################################################################

class ParticleData(GasData):
  '''Subclass for particle data.

  Args --
  data_p : Parameters specifying the gridded snapshot file. Includes...
  "    "['sdir'] : Directory the snaphost is stored in
  "    "['snum'] : Snapshot number
  "    "['ptype'] : Type of particle we're gridding. Usually 0 for gas
  '''

  def __init__(self, data_p, **kw_args):

    super(ParticleData, self).__init__(data_p, **kw_args)

    self.retrieve_data(**kw_args)

  ########################################################################

  def retrieve_data(self, **kw_args):

    if 'load_additional_ids' in self.data_p:
      load_additional_ids = self.data_p['load_additional_ids']
    else:
      load_additional_ids = False

    # Assume we convert from cosmological units
    P = read_snapshot.readsnap(self.data_p['sdir'], self.data_p['snum'], self.data_p['ptype'], load_additional_ids=load_additional_ids, cosmological=1, **kw_args)

    # Parse the keys and put in a more general format
    # All units should be the standard GIZMO output units
    self.data = {}
    self.data_attrs = {}
    for key in P.keys():

      # Get the attributes
      attrs_keys = ['redshift', 'omega_lambda', 'flag_metals', 'flag_cooling', 'omega_matter', 'flag_feedbacktp', 'time', 'boxsize', 'hubble', 'flag_sfr', 'flag_stellarage', 'k']
      if key in attrs_keys:
        self.data_attrs[key] = P[key]

      # Get the data
      # Gas Density
      elif key == 'rho':
        self.data['Den'] = P[key]
      # Gas Neutral Hydrogen fraction
      elif key == 'nh':
        self.data['nHI'] = P[key]
      # Should be the mean free electron number per proton
      elif key == 'ne':
        self.data['ne'] = P[key]
      # Star Formation Rate
      elif key == 'sfr':
        self.data['SFR'] = P[key]
      # Position
      elif key == 'p':
        self.data['P'] = P[key].transpose()
      # Velocity
      elif key == 'v':
        self.data['V'] = P[key].transpose()
      # Metal mass fraction
      elif key == 'z':
        self.data['Z'] = P[key][:,0] # Total metallicity (everything not H or He)
        self.data['Z_Species'] = P[key][:,1:] # Details per species, [He, C, N, O, Ne, Mg, Si, S, Ca, Fe], in order
      # Particle IDs
      elif key == 'id':
        self.data['ID'] = P[key]
      elif key == 'child_id':
        self.data['ChildID'] = P[key]
      elif key == 'id_gen':
        self.data['IDGen'] = P[key]
      # Particle Masses
      elif key == 'm':
        self.data['M'] = P[key]
      # Internal energy
      elif key == 'u':
        self.data['U'] = P[key]
      # Smoothing lengths
      elif key == 'h':
        self.data['h'] = P[key]
      elif key == 'age':
        self.data['Age'] = P[key]
      else:
        raise Exception('NULL key, key={}'.format(key))

  ########################################################################

  def calc_temp(self, gamma=5./3.):
    '''Calculate the temperature from the internal energy. '''

    mu = self.calc_mu()
    u_cgs = self.data['U']*1.e10
    self.data['T'] = constants.m_p*mu*(gamma - 1)*u_cgs/constants.k_b

  ########################################################################

  def calc_classifications(self):
    '''Get the classification for each particle, using data from the Angles-Alcazar+16 pipeline.
    Uses classes from the tracked_particle_data_handling.py module.

    Parameters (include in data_p)
    'tracked_p_data_dir' : Directory containing the tracked-particle data.
    'tracked_p_file_tag' : Identifying tag for the tracked-particle data.
    '''

    sim_name = string.split(self.data_p['sdir'], '/')[-1]
    full_data_dir = '{}/{}'.format(self.data_p['tracked_p_data_dir'], sim_name)

    # Load the actual tracked particle data
    tracked_p_data = tracked_particle_data_handling.TrackedParticleDataHandler(full_data_dir, self.data_p['tracked_p_file_tag'])

    # Get the classifications
    self.data['Cl'] = tracked_p_data.classify_dataset(self.data['ID'], self.data_attrs['redshift'])
