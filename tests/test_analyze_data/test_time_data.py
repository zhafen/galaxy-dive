'''Testing for particle_data.py
'''

import copy
import mock
import numpy as np
import numpy.testing as npt
import pandas as pd
import pdb
import unittest

import galaxy_diver.analyze_data.simulation_data as simulation_data
import galaxy_diver.read_data.snapshot as read_snapshot
import galaxy_diver.utils.astro as astro

########################################################################

default_kwargs = {
  'data_dir' : './tests/data/sdir',
  'ahf_data_dir' : './tests/data/analysis_dir',
  'snum' : [ 600, 550, 500 ],
  'ahf_index' : 600,
  'store_ahf_reader' : True,
}

########################################################################

class TestTimeData( unittest.TestCase ):

  def setUp( self ):

    self.t_data = simulation_data.TimeData( **default_kwargs )

    self.t_data.data = {
      'P' : np.zeros( ( 3, 4, 5 ) ),
      'V' : np.zeros( ( 3, 4, 5 ) ),
    }

    self.t_data.data_attrs = {
      'hubble' : 0.70199999999999996,
      'omega_matter' : 0.272,
      'omega_lambda' : 0.728,
    }

  ########################################################################

  def test_retrieve_halo_data( self ):

    self.t_data.retrieve_halo_data()

    # Make sure we have the right redshift
    expected = np.array([ 0.       ,  0.0698467,  0.16946  ])
    actual = self.t_data.redshift
    npt.assert_allclose( expected, actual )

    # Make sure we have the position right for mt halo 0, snap 500
    actual = self.t_data.halo_coords
    expected = np.array([
      [ 41792.16330006,  44131.23097357,  46267.67030715],
      [ 39109.186634  ,  41182.20415729,  43161.67681807],
      [ 35829.92061045,  37586.13756083,  39375.69771064]
    ]).transpose()
    npt.assert_allclose( expected, actual )

    actual = self.t_data.halo_velocity
    expected = np.array([
      [-48.53,  72.1 ,  96.12],
      [-49.05,  72.73,  96.86],
      [-48.89,  73.77,  97.25]
    ]).transpose()
    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_center_coords( self ):

    self.t_data.center_method = np.array( [ [ 1., 2., 3., 4., 5. ], ]*3 )

    self.t_data.center_coords()
    expected = np.array( [ [ [-1., -2., -3., -4., -5.], ]*4, ]*3 )
    actual = self.t_data.data['P']
    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_center_vel_coords( self ):

    self.t_data.vel_center_method = np.array( [ [ 1., 2., 3., 4., 5. ], ]*3 )

    self.t_data.center_vel_coords()
    expected = np.array( [ [ [-1., -2., -3., -4., -5.], ]*4, ]*3 )
    actual = self.t_data.data['V']
    npt.assert_allclose( expected, actual )

  ########################################################################

  @mock.patch( 'galaxy_diver.analyze_data.simulation_data.SimulationData.get_data' )
  @mock.patch( 'galaxy_diver.analyze_data.simulation_data.SimulationData.hubble_z', new_callable=mock.PropertyMock )
  def test_add_hubble_flow( self, mock_hubble_z, mock_get_data ):

    # Change the mock data we're using
    self.t_data.vel_centered = True
    self.t_data.data['P'] = np.array( [ [ [1., 2., 3., 4., 5.], ]*4, ]*3 )

    # Create side effects for the mock functions
    mock_hubble_z.side_effect = [ pd.Series( [ 1., ]*5 ), ]*2
    mock_get_data.side_effect = [ self.t_data.data['P'], ]

    self.t_data.add_hubble_flow()

    actual = self.t_data.data['V']
    expected = np.array( [ [ [1., 2., 3., 4., 5.], ]*4, ]*3 )
    npt.assert_allclose( expected, actual )

    mock_hubble_z.assert_called_once()

  ########################################################################

  def test_get_data( self ):

    # Change the mock data
    self.t_data.data['P'] = np.array( [ [ [1., 2., 3., 4., 5.], ]*4, ]*3 )

    self.t_data.centered = True
    self.t_data.vel_centered = True

    expected = np.array( [ [1., 2., 3., 4., 5.], ]*4 )
    actual = self.t_data.get_data( 'Rx' )
    npt.assert_allclose( expected, actual )

  ########################################################################

  @mock.patch( 'galaxy_diver.analyze_data.ahf.HaloData.get_mt_data' )
  def test_get_processed_data( self, mock_get_mt_data ):
    '''Test that our processed data method works, especially when scaling data.'''
  
    # Setup the mock data
    self.t_data.data['P'] = np.array( [ [ [1., 2., 3., 4., 5.], ]*4, ]*3 )
    mock_get_mt_data.side_effect = [ np.array([1., 2., 3., 4., 5.]) ]

    # So we don't have to do extra calculations that change things
    self.t_data.centered = True
    self.t_data.vel_centered = True

    expected = np.array( [ [1., 1., 1., 1., 1.], ]*4 )/self.t_data.data_attrs['hubble']
    actual = self.t_data.get_processed_data( 'Rx', scale_key='Rvir', scale_a_power=1., scale_h_power=-1., )
    npt.assert_allclose( expected, actual )

    mock_get_mt_data.assert_called_once_with( 'Rvir', mt_halo_id=0, a_power=1., )

  ########################################################################

  def test_hubble_z( self ):

    self.t_data.redshift = np.array([ 0.       ,  0.0698467,  0.16946  ])

    expected = np.array( [ astro.hubble_parameter( redshift, units='km/s/kpc' ) for redshift in self.t_data.redshift ] )
    actual = self.t_data.hubble_z
    npt.assert_allclose( expected, actual )

  ########################################################################

  def test_redshift_pd_df( self ):
    '''Sometimes we have a pandas DataFrame with NaNs in it as the redshift, and that causes trouble.
    '''
    
    data = {
      'redshift' : np.array( [1., 2., np.nan ] ),
    }
    df = pd.DataFrame( data, index=np.array( [ 1, 2, 3 ] ) )

    self.t_data._redshift = df['redshift']
    self.t_data.redshift = np.array( [1., 2., 3.])

  ########################################################################

  def test_redshift_pd_df_nan_alternate( self ):
    '''Having nans in either array (due to how things are defined when redshift is too low) can cause problems.
    '''
    
    data = {
      'redshift' : np.array( [1., 2., 3. ] ),
    }
    df = pd.DataFrame( data, index=np.array( [ 1, 2, 3 ] ) )

    self.t_data._redshift = df['redshift']
    self.t_data.redshift = np.array( [1., 2., np.nan ] )

  ########################################################################

  def test_calc_radial_distance( self ):

    # Change the mock data
    self.t_data.data['P'] = np.array( [ [ [1., 2., 3., 4., 5.], ]*4, ]*3 )

    self.t_data.centered = True
    self.t_data.vel_centered = True

    expected = np.array( [ [1., 2., 3., 4., 5.], ]*4 )*np.sqrt( 3 )
    actual = self.t_data.get_data( 'R' )
    npt.assert_allclose( expected, actual )


########################################################################
########################################################################


