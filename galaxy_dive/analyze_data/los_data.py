#!/usr/bin/env python
'''Subclass for analyzing particle data.

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import galaxy_dive.analyze_data.generic_data as generic_data

########################################################################

class LOSData( generic_data.GenericData ):

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

