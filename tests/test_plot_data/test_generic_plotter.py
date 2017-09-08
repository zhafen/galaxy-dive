'''Testing for particle_data.py
'''

import mock
import numpy as np
import numpy.testing as npt
import pdb
import unittest

import galaxy_diver.analyze_data.simulation_data as simulation_data
import galaxy_diver.plot_data.generic_plotter as generic_plotter

########################################################################

default_kwargs = {
  'data_dir' : './tests/data/sdir',
  'ahf_data_dir' : './tests/data/analysis_dir',
  'snum' : 500,
  'ahf_index' : 600,
}

########################################################################

class TestStartup( unittest.TestCase ):

  @mock.patch( 'galaxy_diver.plot_data.generic_plotter.GenericPlotter.plot_something', create=True )
  @mock.patch( 'galaxy_diver.analyze_data.simulation_data.SnapshotData.retrieve_halo_data' )
  @mock.patch( 'galaxy_diver.analyze_data.simulation_data.SnapshotData.__init__' )
  def test_basic( self, mock_init, mock_retrieve_halo_data, mock_plot_something ):

    mock_init.side_effect = [ None, ]

    snapshot_data = simulation_data.SnapshotData()
    data_plotter = generic_plotter.GenericPlotter( snapshot_data )

    data_plotter.data_object.retrieve_halo_data()
    mock_retrieve_halo_data.assert_called_once()

    data_plotter.plot_something()
    mock_plot_something.assert_called_once()


    

