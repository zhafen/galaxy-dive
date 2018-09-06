#!/usr/bin/env python
'''Subclass for analyzing particle data.

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import numpy as np

import galaxy_dive.analyze_data.simulation_data as simulation_data
import galaxy_dive.read_data.snapshot as read_snapshot
import galaxy_dive.utils.constants as constants
import galaxy_dive.utils.utilities as utilities

########################################################################

class ParticleData( simulation_data.SnapshotData ):
  '''Subclass for particle data.
  '''

  @utilities.store_parameters
  def __init__( self, sdir=None, snum=None, ptype=None, load_additional_ids=False, **kwargs ):
    '''Initialize.

    Args:
      sdir (str) : Directory the snapshot is stored in
      snum (int) : Snapshot number to open
      ptype (int) : Particle type to load.
      load_additional_ids (bool, optional) : Whether or not to load child_ids, etc, if they exist.
    '''

    super( ParticleData, self ).__init__( data_dir=sdir, snum=snum, **kwargs )

    self.retrieve_data()

  ########################################################################

  def retrieve_data( self ):

    # Assume we convert from cosmological units
    P = read_snapshot.readsnap( self.sdir, self.snum, self.ptype, load_additional_ids=self.load_additional_ids,
                                cosmological=True, )

    # Parse the keys and put in a more general format
    # All units should be the standard GIZMO output units
    self.data = {}
    self.data_attrs = {}
    for key in P.keys():

      # Get the attributes
      attrs_keys = ['redshift', 'omega_lambda', 'flag_metals', 'flag_cooling', 'omega_matter', 'flag_feedbacktp',
                    'time', 'boxsize', 'hubble', 'flag_sfr', 'flag_stellarage', 'k']
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
      elif key == 'potential':
        self.data['Potential'] = P[key]
      else:
        raise Exception('NULL key, key={}'.format(key))

  ########################################################################

  def calc_temp( self, gamma=5./3. ):
    '''Calculate the temperature from the internal energy. '''

    #raise Exception( TODO: Test this )

    mu = self.calc_mu()
    u_cgs = self.data['U']*1.e10
    self.data['T'] = constants.PROTON_MASS*mu*(gamma - 1)*u_cgs/constants.K_B

  ########################################################################

  def calc_mu( self ):
    '''we track this with metal species now, could do better...
    '''

    XH=0.76

    yhelium = ( 1. - XH ) / ( 4. * XH )

    return ( 1. + 4. * yhelium ) / ( 1. + yhelium + self.data['ne'] )

  ########################################################################

  def calc_pressure( self ):
    '''Calculate the pressure in units of K/cm^3.'''

    self.data['Pressure'] = self.get_data( 'NumDen' )*self.get_data( 'T' )

  ########################################################################

  def calc_classifications( self ):
    '''Get the classification for each particle, using data from the Angles-Alcazar+16 pipeline.
    Uses classes from the tracked_particle_data_handling.py module.

    Parameters (include in data_p)
    'tracked_p_data_dir' : Directory containing the tracked-particle data.
    'tracked_p_file_tag' : Identifying tag for the tracked-particle data.
    '''

    sim_name = self.data_p['sdir'].split( '/' )[-1]
    full_data_dir = os.path.join( self.data_p['tracked_p_data_dir'], sim_name )

    # Load the actual tracked particle data
    tracked_p_data = tracked_particle_data_handling.TrackedParticleDataHandler(full_data_dir,
                                                                               self.data_p['tracked_p_file_tag'])

    # Get the classifications
    self.data['Cl'] = tracked_p_data.classify_dataset(self.data['ID'], self.data_attrs['redshift'])

  ########################################################################

  def calc_metal_mass( self ):
      
    self.data['metal_mass'] = self.get_data( 'M' ) * self.get_data( 'Z' )

  ########################################################################

  def calc_enriched_metal_mass( self ):
    '''Calculate the metal mass that comes from enrichment, not counting mass
    that's at the metallicity floor. Assumes that the there will always be at
    least one particle in the simulation that's at the metallicity floor.'''

    enrichment_fraction = self.get_data( 'Z' ) - self.get_data( 'Z' ).min()

    self.data['enriched_metal_mass'] = self.get_data( 'M' ) * enrichment_fraction

  ########################################################################

  def find_duplicate_ids( self ):
    '''Find all the IDs in the particle data set that have duplicates.

    Returns:
      duplicate_ids ( np.ndarray ) : Array of IDs that have duplicates.
    '''

    unique_ids, id_counts = np.unique( self.get_data( 'ID' ), return_counts=True )

    duplicate_inds = np.where( id_counts > 1 )[0]

    return unique_ids[duplicate_inds]





























