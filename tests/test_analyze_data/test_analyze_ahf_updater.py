#!/usr/bin/env python
'''Testing for analyze_data.ahf.py

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import glob
from mock import call, patch, PropertyMock
import numpy as np
import numpy.testing as npt
import os
import pytest
import unittest

import galaxy_diver.analyze_data.ahf_updater as ahf_updater
import galaxy_diver.read_data.ahf as read_ahf
import galaxy_diver.analyze_data.particle_data as particle_data
import galaxy_diver.galaxy_finder.finder as gal_finder

sdir = './tests/data/analysis_dir'
sdir2 = './tests/data/analysis_dir2'
data_sdir = './tests/data/sdir'
data_sdir2 = './tests/data/sdir2'

########################################################################

# Decorator for skipping slow tests
slow = pytest.mark.skipif(
    not pytest.config.getoption("--runslow"),
    reason="need --runslow option to run"
)


class SaneEqualityArray(np.ndarray):
    '''Numpy array subclass that allows you to test if two arrays are equal.'''

    def __eq__(self, other):
        return ( isinstance(other, np.ndarray) and self.shape == other.shape and
                 np.allclose(self, other) )


def sane_eq_array(list_in):
    '''Wrapper for SaneEqualityArray, that takes in a list.'''

    arr = np.array(list_in)

    return arr.view(SaneEqualityArray)

########################################################################
########################################################################


class TestHaloUpdater( unittest.TestCase ):

    def setUp( self ):

        self.ahf_updater = ahf_updater.HaloUpdater( sdir )

        # Remove smooth halo files that are generated
        halo_filepaths = glob.glob(
            './tests/data/ahf_test_data/halo_*_smooth.dat' )
        for halo_filepath in halo_filepaths:
            if os.path.isfile( halo_filepath ):
                os.system( 'rm {}'.format( halo_filepath ) )

    ########################################################################

    def tearDown( self ):

        # Remove extra files that are generated
        halo_filepaths = glob.glob(
            './tests/data/ahf_test_data/halo_*_test.dat' )
        for halo_filepath in halo_filepaths:
            if os.path.isfile( halo_filepath ):
                os.system( 'rm {}'.format( halo_filepath ) )

    ########################################################################

    def test_save_and_load_mtree_halo_files( self ):

        self.ahf_updater.get_mtree_halos( 'snum' )
        expected = self.ahf_updater.mtree_halos[10]['ID']
        expected_detail = self.ahf_updater.mtree_halos[0]['ID'][500]

        # Save
        save_tag = 'test'
        self.ahf_updater.save_mtree_halos( save_tag )

        # Load
        new_ahf_reader = ahf_updater.HaloUpdater( sdir )
        new_ahf_reader.get_mtree_halos( 'snum', tag=save_tag )
        actual = new_ahf_reader.mtree_halos[10]['ID']
        actual_detail = new_ahf_reader.mtree_halos[0]['ID'][500]

        npt.assert_allclose( expected, actual )
        npt.assert_allclose( expected_detail, actual_detail )

    ########################################################################

    def test_include_ahf_halos_to_mtree_halos( self ):

        # Load the mtree halos data
        self.ahf_updater.get_mtree_halos( 'snum', 'sparse', True )

        # Function itself.
        self.ahf_updater.include_ahf_halos_to_mtree_halos()

        # Test for snapshot 600
        snum = 600
        del self.ahf_updater.ahf_halos
        self.ahf_updater.get_halos( snum )
        self.ahf_updater.get_halos_add( snum )
        for halo_id in self.ahf_updater.mtree_halos.keys():
            test_keys = [ 'Rvir', 'cAnalytic', ]
            for test_key in test_keys:
                expected = self.ahf_updater.ahf_halos[test_key][halo_id]
                actual = self.ahf_updater.mtree_halos[halo_id][test_key][snum]
                npt.assert_allclose( expected, actual )

        # Test for snapshot 599
        snum = 599
        for halo_id in self.ahf_updater.mtree_halos.keys():
            test_keys = [ 'Rstar0.99', 'cAnalytic', 'Rstar0.5', ]
            for test_key in test_keys:
                expected = np.nan
                actual = self.ahf_updater.mtree_halos[halo_id][test_key][snum]
                npt.assert_allclose( expected, actual )

    ########################################################################

    def test_get_accurate_redshift( self ):

        expected = 0.00015930

        self.ahf_updater.get_mtree_halos( 'snum' )

        self.ahf_updater.get_accurate_redshift( './tests/data/sdir' )

        actual = self.ahf_updater.mtree_halos[0]['redshift'][599]

        npt.assert_allclose( expected, actual, )

    ########################################################################

    def test_smooth_mtree_halos( self ):

        self.ahf_updater.get_mtree_halos( 'snum' )

        self.ahf_updater.smooth_mtree_halos( data_sdir )

        # Test that the redshift worked.
        redshift_expected_598 = 0.00031860
        redshift_actual = self.ahf_updater.mtree_halos[0]['redshift'][598]
        npt.assert_allclose( redshift_expected_598, redshift_actual )

        # Test that r_vir worked
        # (that we never make a dip down in the early snapshot)
        r_vir_min = self.ahf_updater.mtree_halos[0]['Rvir'][210].min()
        assert r_vir_min > 30.

        # Test that m_vir worked
        # (that we never make a dip down in the early snapshot)
        m_vir_min = self.ahf_updater.mtree_halos[0]['Mvir'][210].min()
        assert m_vir_min > 1e11

    ########################################################################

    def test_smooth_mtree_halos_fire1( self ):

        # Get the right directory
        self.ahf_updater.sdir = sdir2
        self.ahf_updater.get_mtree_halos( 440 )

        self.ahf_updater.smooth_mtree_halos( data_sdir2 )

        # Test that the redshift worked.
        redshift_expected_439 = 0.0049998743750157
        redshift_actual = self.ahf_updater.mtree_halos[0]['redshift'][439]
        npt.assert_allclose( redshift_expected_439, redshift_actual )

        # Test that r_vir worked
        # (that we never make a dip down in the early snapshot)
        r_vir_min = self.ahf_updater.mtree_halos[0]['Rvir'][326].min()
        assert r_vir_min > 100.

        # Test that m_vir worked
        # (that we never make a dip down in the early snapshot)
        m_vir_min = self.ahf_updater.mtree_halos[0]['Mvir'][326].min()
        assert m_vir_min > 1e11

    ########################################################################

    @slow
    def test_save_smooth_mtree_halos( self ):

        # Get the results
        self.ahf_updater.save_smooth_mtree_halos(
            data_sdir,
            'snum',
            include_ahf_halos_add=False,
            include_concentration=True,
            smooth_keys = [],
        )

        # Load the saved files
        self.ahf_updater.get_mtree_halos( 'snum', 'smooth' )

        # Test that the redshift worked.
        redshift_expected_598 = 0.00031860
        redshift_actual = self.ahf_updater.mtree_halos[0]['redshift'][598]
        npt.assert_allclose( redshift_expected_598, redshift_actual )

        # Test that r_vir worked
        # (that we never make a dip down in the early snapshot)
        r_vir_min = self.ahf_updater.mtree_halos[0]['Rvir'][210].min()
        assert r_vir_min > 30.

        # Test that m_vir worked (that we never make a dip down in the early snapshot)
        m_vir_min = self.ahf_updater.mtree_halos[0]['Mvir'][210].min()
        assert m_vir_min > 1e11

    ########################################################################

    @slow
    def test_save_smooth_mtree_halos_different_snum( self ):

        # Pass it the right directory
        self.ahf_updater.sdir = sdir2

        # Get the results
        self.ahf_updater.save_smooth_mtree_halos(
            data_sdir2,
            440,
            False,
            smooth_keys = [],
        )

        # Load the saved files
        self.ahf_updater.get_mtree_halos( 440, 'smooth' )

        # Test that the redshift worked.
        redshift_expected_439 = 0.0049998743750157
        redshift_actual = self.ahf_updater.mtree_halos[0]['redshift'][439]
        npt.assert_allclose( redshift_expected_439, redshift_actual )

        # Test that r_vir worked (that we never make a dip down in the early snapshot)
        r_vir_min = self.ahf_updater.mtree_halos[0]['Rvir'][210].min()
        assert r_vir_min > 30.

        # Test that m_vir worked (that we never make a dip down in the early snapshot)
        m_vir_min = self.ahf_updater.mtree_halos[0]['Mvir'][210].min()
        assert m_vir_min > 1e11

        # Make sure there aren't NaN's
        assert self.ahf_updater.mtree_halos[0]['Rvir'][440] > 0

    ########################################################################

    def test_save_smooth_mtree_halos_including_ahf_adds( self ):

        self.ahf_updater.get_mtree_halos( 'snum', 'sparse', True )

        #DEBUG
        import pdb; pdb.set_trace()

        # Get the results
        self.ahf_updater.save_smooth_mtree_halos(
            metafile_dir = data_sdir,
            index = 'snum',
            include_ahf_halos_add = True,
            smooth_keys = [ 'Rstar0.5' ],
            tag = 'sparse',
            adjust_default_labels = True,
        )

        #DEBUG
        import pdb; pdb.set_trace()

    ########################################################################

    @patch( 'galaxy_diver.read_data.ahf.AHFReader.get_halos_add', )
    def test_save_custom_mtree_halos( self, mock_get_halos_add ):

        # Run the test
        halo_ids = np.array( [ 0, 3, 5, ] )
        self.ahf_updater.save_custom_mtree_halos(
            snums=[600,550,500],
            halo_ids=halo_ids,
            metafile_dir=data_sdir,
        )

        # Load in new file
        self.ahf_updater.get_mtree_halos( tag='custom' )
        mtree_halo = self.ahf_updater.mtree_halos[0]

        # Compare IDs
        npt.assert_allclose( halo_ids, mtree_halo['ID'] )

        # Compare number of substructures (nice easy integers to compare)
        expected = np.array([ 792, 48, 55 ])
        actual = mtree_halo['numSubStruct']
        npt.assert_allclose( expected, actual )

        # Compare the snapshot number
        expected = np.array([ 600, 550, 500, ])
        actual = mtree_halo.index
        npt.assert_allclose( expected, actual )

        # Compare the redshift
        expected = np.array([ 0., 0.069847, 0.169460, ])
        actual = mtree_halo['redshift']
        npt.assert_allclose( expected, actual, atol=1e-3 )

    ########################################################################

    def test_save_custom_mtree_halos_including_ahf_halos_add( self ):

        # Run the test
        halo_ids = np.array( [ 3, ] )
        self.ahf_updater.save_custom_mtree_halos( snums=[600,], halo_ids=halo_ids, metafile_dir=data_sdir, )

        # Load in new file
        self.ahf_updater.get_mtree_halos( tag='custom' )
        mtree_halo = self.ahf_updater.mtree_halos[0]

        del self.ahf_updater.ahf_halos
        self.ahf_updater.get_halos( 600 )
        self.ahf_updater.get_halos_add( 600 )
        expected = np.array( [ self.ahf_updater.ahf_halos['cAnalytic'][3], ] )
        actual = mtree_halo['cAnalytic']
        npt.assert_allclose( expected, actual )

    ########################################################################

    def test_save_custom_mtree_halos_default_snums( self ):

        # Run the test
        halo_ids = np.array( [ 3, ] )
        self.ahf_updater.save_custom_mtree_halos( snums=600, halo_ids=halo_ids, metafile_dir=data_sdir, )

        # Load in new file
        self.ahf_updater.get_mtree_halos( tag='custom' )
        mtree_halo = self.ahf_updater.mtree_halos[0]

        expected = np.array([ 600 ])
        actual = mtree_halo.index
        npt.assert_allclose( expected, actual )

    ########################################################################

    @slow
    def test_get_analytic_concentration_mtree_halos( self ):

        # Load the saved files
        self.ahf_updater.get_mtree_halos( 'snum', )
        self.ahf_updater.get_analytic_concentration( data_sdir )

        c_vir_z0_expected = 10.66567139
        c_vir_z0_actual = self.ahf_updater.mtree_halos[0]['cAnalytic'][600]

        npt.assert_allclose( c_vir_z0_expected, c_vir_z0_actual, rtol=1e-3 )

    ########################################################################

    @slow
    def test_save_and_load_ahf_halos_add( self ):

        # Save halos_add
        self.ahf_updater.save_ahf_halos_add(
            600,
            metafile_dir = data_sdir,
            include_mass_radii = False,
            include_enclosed_mass = False,
            include_v_circ = False,
        )

        # Load halos_add
        self.ahf_updater.get_halos_add( 600 )

        # Check that we calculated the concentration correctly.
        c_vir_z0_expected = 10.66567139
        c_vir_z0_actual = self.ahf_updater.ahf_halos_add['cAnalytic'][0]
        npt.assert_allclose( c_vir_z0_expected, c_vir_z0_actual, rtol=1e-3 )

    ########################################################################

    @patch( 'galaxy_diver.analyze_data.ahf_updater.HaloUpdater.save_ahf_halos_add' )
    def test_save_multiple_ahf_halos_adds( self, mock_save_ahf_halos_add ):

        self.ahf_updater.save_multiple_ahf_halos_adds( data_sdir, 500, 600, 50 )

        calls = [
            call(500, data_sdir),
            call(550, data_sdir),
            call(600, data_sdir),
            ]
        mock_save_ahf_halos_add.assert_has_calls( calls, any_order=True )

########################################################################
########################################################################


class TestMassRadii( unittest.TestCase ):

    def setUp( self ):

        self.ahf_updater = ahf_updater.HaloUpdater( sdir )

        snum = 600
        self.ahf_updater.get_halos( snum )

    ########################################################################

    @patch( 'galaxy_diver.galaxy_finder.finder.GalaxyFinder.get_mass_radius', )
    @patch( 'galaxy_diver.galaxy_finder.finder.GalaxyFinder.__init__', )
    def test_get_mass_radii( self, mock_init, mock_get_mass_radius ):

        mock_init.side_effect = [ None, ]
        mock_get_mass_radius.side_effect = [ np.arange(10)*10., np.linspace( 50., 200., 10 ), ]

        mass_fractions = [ 0.5, 0.99, ]

        expected = [ np.arange(10)*10., np.linspace( 50., 200., 10 ), ]
        data_dir = os.path.join( data_sdir, 'output' )
        actual = self.ahf_updater.get_mass_radii( mass_fractions, data_dir, 0.15, 'Rvir', )

        npt.assert_allclose( expected, actual )

    ########################################################################

    @patch( 'galaxy_diver.analyze_data.particle_data.ParticleData.__init__', )
    def test_get_mass_radii_no_stars( self, mock_init, ):
        '''Test that, when there are no stars in the snapshot, we return nan values.'''

        mock_init.side_effect = [ None, ]

        # I mock data here. Note that I had to mock __init__ too, to prevent overwriting.
        with patch.object( particle_data.ParticleData, 'data', new_callable=PropertyMock, create=True ) as mock_data:
            mock_data.return_value = {}

            mass_fractions = [ 0.5, 0.99, ]

            expected = [ np.array( [np.nan, ]*9 ), ]*2
            data_dir = os.path.join( data_sdir, 'output' )
            actual = self.ahf_updater.get_mass_radii( mass_fractions, data_dir, 0.15, 'Rvir', )

            npt.assert_allclose( expected, actual )

    ########################################################################

    def test_get_mass_radii_from_subsampled_data( self,):
        '''Now test, using actual simulation data (even if it's subsampled).'''

        mass_fractions = [ 0.5,  ]
        data_dir = os.path.join( data_sdir, 'output' )

        actual = self.ahf_updater.get_mass_radii( mass_fractions, data_dir, 0.15, 'Rvir', )
        expected = np.array( [ np.nan, ]*9 )

        npt.assert_allclose( expected, actual[0] )

    ########################################################################

    @patch( 'galaxy_diver.analyze_data.ahf_updater.HaloUpdater.get_analytic_concentration' )
    def test_save_ahf_halos_add_including_mass_radii( self, mock_get_analytic_concentration ):

        mock_get_analytic_concentration.side_effect = [ np.arange( 9 ), ]

        # Save halos_add
        sim_data_dir = os.path.join( data_sdir, 'output' )

        mass_radii_kwargs = {
            'mass_fractions' : [ 0.5, 0.99 ],
            'galaxy_cut' : 0.15,
            'length_scale' : 'Rvir',
        }
        self.ahf_updater.save_ahf_halos_add(
            600,
            include_enclosed_mass = False,
            include_v_circ = False,
            metafile_dir = data_sdir,
            simulation_data_dir = sim_data_dir,
            mass_radii_kwargs = mass_radii_kwargs,
        )

        # Load halos_add
        self.ahf_updater.get_halos( 600, True )
        self.ahf_updater.get_halos_add( 600, True )

        # Make sure the columns exist
        assert 'Rstar0.5' in self.ahf_updater.ahf_halos.columns
        assert 'Rstar0.99' in self.ahf_updater.ahf_halos.columns

########################################################################
########################################################################


class TestEnclosedMass( unittest.TestCase ):

    def setUp( self ):

        self.ahf_updater = ahf_updater.HaloUpdater( sdir )

        snum = 600
        self.ahf_updater.get_halos( snum )

    ########################################################################

    @patch( 'galaxy_diver.galaxy_finder.finder.GalaxyFinder.__init__', )
    @patch( 'galaxy_diver.analyze_data.particle_data.ParticleData.redshift', )
    @patch( 'galaxy_diver.analyze_data.particle_data.ParticleData.__init__', )
    def test_get_enclosed_mass( self, mock_init, mock_redshift, mock_init_finder, ):
        '''Given a galaxy with four dark matter particles, and 3 halos, can we get the mass inside each of the halos?
        Only two of the halos contain galaxies.
        '''

        mock_init.side_effect = [ None, ]
        mock_init_finder.side_effect = [ None, ]


        # Mock what redshift it is
        mock_redshift.side_effect = [ 0., ]

        # I mock data here. Note that I had to mock __init__ too, to prevent overwriting.
        with patch.object( particle_data.ParticleData, 'data', new_callable=PropertyMock, create=True ) as mock_data, \
            patch.object( particle_data.ParticleData, 'data_attrs', new_callable=PropertyMock, \
                                        create=True ) as mock_d_attrs, \
            patch.object( gal_finder.GalaxyFinder, 'mass_inside_all_halos', new_callable=PropertyMock, \
                                        create=True ) as mock_mass_inside_all_halos:


            # Mock simulation data.
            mock_data.return_value = {
                'M' : np.array( [ 1., 2., 3., 4., ] ),
                'P' : np.random.rand( 3, 4 ),
            }
            mock_d_attrs.return_value = {
                'hubble' : .702,
            }
        # Mock what particles are in the halos.
            mock_mass_inside_all_halos.return_value = np.array( [ 6., 3., np.nan, ] )

            expected = np.array([ 6., 3., np.nan, ])*.702
            data_dir = os.path.join( data_sdir, 'output' )
            actual = self.ahf_updater.get_enclosed_mass( data_dir, 'star', 1., 'Rstar0.5', )

            npt.assert_allclose( expected, actual )

    ########################################################################

    @patch( 'galaxy_diver.galaxy_finder.finder.GalaxyFinder.find_containing_halos', )
    @patch( 'galaxy_diver.galaxy_finder.finder.GalaxyFinder.__init__', )
    @patch( 'galaxy_diver.analyze_data.particle_data.ParticleData.__init__', )
    def test_get_enclosed_mass_no_particles_ptype( self, mock_init, mock_init_finder, mock_find_containing_halos ):
        '''Given a galaxy with four dark matter particles, and 3 halos, can we get the mass inside each of the halos?
        '''

        mock_init.side_effect = [ None, ]
        mock_init_finder.side_effect = [ None, ]

        # Mock what particles are in the halos.
        mock_find_containing_halos.side_effect = [
            np.array([
                [ 1, 1, 0, ],
                [ 1, 1, 0, ],
                [ 1, 0, 0, ],
                [ 0, 0, 0, ],
            ]).astype( bool ),
        ]

        # I mock data here. Note that I had to mock __init__ too, to prevent overwriting.
        with patch.object( particle_data.ParticleData, 'data', new_callable=PropertyMock, create=True ) as mock_data:

            # Mock simulation data.
            mock_data.return_value = {}

            expected = np.array( [0., ]*9 )
            data_dir = os.path.join( data_sdir, 'output' )
            actual = self.ahf_updater.get_enclosed_mass( data_dir, 'star', 1., 'Rstar0.5', )

            npt.assert_allclose( expected, actual )

    ########################################################################

    @patch( 'galaxy_diver.analyze_data.ahf_updater.HaloUpdater.get_analytic_concentration' )
    def test_save_ahf_halos_add_including_masses( self, mock_get_analytic_concentration ):
        '''Test that we can write-out files that contain additional information, with that information
        including the mass inside some radius from the center of the halo.
        '''

        mock_get_analytic_concentration.side_effect = [ np.arange( 9 ), ]

        # Save halos_add
        sim_data_dir = os.path.join( data_sdir, 'output' )

        enclosed_mass_kwargs = {
            'galaxy_cut' : 1.0,
            'length_scale' : 'Rvir',
        }
        self.ahf_updater.save_ahf_halos_add(
            600,
            include_analytic_concentration = True,
            include_enclosed_mass = True,
            include_v_circ = False,
            metafile_dir = data_sdir,
            simulation_data_dir = sim_data_dir,
            enclosed_mass_ptypes = [ 'star', 'DM', ],
            enclosed_mass_kwargs = enclosed_mass_kwargs,
        )

        # Load halos_add
        self.ahf_updater.get_halos( 600, True )
        self.ahf_updater.get_halos_add( 600, True )

        # Make sure the columns exist
        assert 'Mstar(Rvir)' in self.ahf_updater.ahf_halos.columns

########################################################################
########################################################################


class TestAverageInsideGalaxy( unittest.TestCase ):

    def setUp( self ):

        self.ahf_updater = ahf_updater.HaloUpdater( sdir )

        snum = 600
        self.ahf_updater.get_halos( snum )

    ########################################################################

    @patch( 'galaxy_diver.galaxy_finder.finder.GalaxyFinder.weighted_summed_quantity_inside_galaxy', )
    @patch( 'galaxy_diver.galaxy_finder.finder.GalaxyFinder.__init__', )
    def test_get_average_x_velocity( self, mock_init_finder, mock_weighted_summed ):
        '''Given a galaxy with four dark matter particles, and 3 halos, can we get the mass inside each of the halos?
        Only two of the halos contain galaxies.
        '''

        mock_init_finder.side_effect = [ None, ]

        # Mock what particles are in the halos.
        mock_weighted_summed.side_effect = [ np.array( [ 16./9., 3., np.nan, ] ), ]
        data_dir = os.path.join( data_sdir, 'output' )

        actual = self.ahf_updater.get_average_quantity_inside_galaxy( 'Vx', data_dir, 'star', 1., 'Rstar0.5', )
        expected = np.array([ 16./9., 3., np.nan, ])

        v_x = sane_eq_array([
            -143.67724609, -347.99859619,   21.3268013 ,  -55.31821823,
            -143.2099762 , -115.62627411,  -63.72027588,  -53.12395096,
            -108.77227783,   62.66027451,  -34.19268036, -150.98040771,
            -217.43572998,  -65.45542145,   99.38283539, -126.13137817,
            -30.76893997,   93.85800171,  177.28361511,  109.82576752,
            128.3290863 ,   94.03945923,  116.39562225,  264.83105469,
            -99.63202667, -201.7532196 ,   89.95071411, -144.08346558,
            225.93339539, -113.14592743,  125.1570816 ,   41.58389664,
            -154.07600403,  -56.06444931,   15.92539692, -251.2993927 ,
            220.0350647 , -140.63465881,  -11.98395061, -173.34442139
        ])
        m = sane_eq_array([
            5.21625374e-07,   1.00258722e-06,   4.62027236e-07,
            5.93719874e-07,   6.26516786e-07,   5.39901424e-07,
            8.95511302e-07,   5.75699458e-07,   6.39436234e-07,
            5.04677290e-07,   1.26963946e-06,   5.28015400e-07,
            5.60212541e-07,   5.11218372e-07,   5.61493907e-07,
            7.03654884e-07,   5.62677780e-07,   6.92173080e-07,
            6.03309812e-07,   5.24588317e-07,   5.99036394e-07,
            6.69138338e-07,   5.52032433e-07,   6.70654891e-07,
            5.54359693e-07,   4.65115283e-07,   5.84278441e-07,
            7.92890469e-07,   5.68287222e-07,   5.06611262e-07,
            5.90570652e-07,   5.35295811e-07,   7.25091169e-07,
            1.21602794e-06,   4.94437377e-07,   6.32833491e-07,
            5.09029334e-07,   4.88467037e-07,   5.06468749e-07,
            5.65739066e-07,
        ])
        mock_weighted_summed.assert_called_once_with( v_x, m, np.nan, )

        npt.assert_allclose( expected, actual )

    ########################################################################

    @patch( 'galaxy_diver.galaxy_finder.finder.GalaxyFinder.find_containing_halos', )
    @patch( 'galaxy_diver.galaxy_finder.finder.GalaxyFinder.__init__', )
    @patch( 'galaxy_diver.analyze_data.particle_data.ParticleData.__init__', )
    def test_get_average_x_velocity_no_particles_ptype( self, mock_init, mock_init_finder, mock_find_containing_halos ):
        '''Given a galaxy with four dark matter particles, and 3 halos, can we get the mass inside each of the halos?
        '''

        mock_init.side_effect = [ None, ]
        mock_init_finder.side_effect = [ None, ]

        # Mock what particles are in the halos.
        mock_find_containing_halos.side_effect = [
            np.array([
                [ 1, 1, 0, ],
                [ 1, 1, 0, ],
                [ 1, 0, 0, ],
                [ 0, 0, 0, ],
            ]).astype( bool ),
        ]

        # I mock data here. Note that I had to mock __init__ too, to prevent overwriting.
        with patch.object( particle_data.ParticleData, 'data', new_callable=PropertyMock, create=True ) as mock_data:

            # Mock simulation data.
            mock_data.return_value = {}

            expected = np.array( [np.nan, ]*9 )
            data_dir = os.path.join( data_sdir, 'output' )
            actual = self.ahf_updater.get_average_quantity_inside_galaxy( 'Vx', data_dir, 'star', 1., 'Rstar0.5', )

            npt.assert_allclose( expected, actual )

    ########################################################################

    @patch( 'galaxy_diver.analyze_data.ahf_updater.HaloUpdater.get_analytic_concentration' )
    def test_save_ahf_halos_add_including_masses( self, mock_get_analytic_concentration ):
        '''Test that we can write-out files that contain additional information, with that information
        including the mass inside some radius from the center of the halo.
        '''

        mock_get_analytic_concentration.side_effect = [ np.arange( 9 ), ]

        # Save halos_add
        sim_data_dir = os.path.join( data_sdir, 'output' )

        average_quantity_inside_galaxy_kwargs = {
            'ptype' : 'star',
            'galaxy_cut' : 1.0,
            'length_scale' : 'Rvir',
        }
        self.ahf_updater.save_ahf_halos_add(
            600,
            include_analytic_concentration = True,
            include_enclosed_mass = False,
            include_average_quantity_inside_galaxy = True,
            include_v_circ = False,
            metafile_dir = data_sdir,
            simulation_data_dir = sim_data_dir,
            average_quantity_inside_galaxy_kwargs = average_quantity_inside_galaxy_kwargs,
        )

        # Load halos_add
        self.ahf_updater.get_halos( 600, True )
        self.ahf_updater.get_halos_add( 600, True )

        # Make sure the columns exist
        assert 'Vxstar(Rvir)' in self.ahf_updater.ahf_halos.columns

########################################################################
########################################################################


class TestCircularVelocity( unittest.TestCase):

    def setUp( self ):

        self.ahf_updater = ahf_updater.HaloUpdater( sdir )

        snum = 500
        self.ahf_updater.get_halos( snum, force_reload=True )

    ########################################################################

    def test_get_circular_velocity( self ):

        with patch.object( read_ahf.AHFReader, 'ahf_halos', new_callable=PropertyMock, create=True ) as mock_ahf_halos:

            mock_ahf_halos.return_value = {
                'Rvir' : np.array( [ 1., np.nan, 2., ] ),
                'Mstar(1.9Rvir)' : np.array([ 1., np.nan, 4., ]),
                'Mgas(1.9Rvir)' : np.array([ 2., np.nan, np.nan, ]),
                'MDM(1.9Rvir)' : np.array([ 2., np.nan, np.nan, ]),
                'MlowresDM(1.9Rvir)' : np.array([ 0., np.nan, np.nan, ]),
            }

            actual = self.ahf_updater.get_circular_velocity( 1.9, 'Rvir', data_sdir )
            expected = np.array([ 0.00363807, np.nan, np.nan ])
            npt.assert_allclose( expected, actual, rtol=1e-5 )

    ########################################################################

    @patch( 'galaxy_diver.analyze_data.ahf_updater.HaloUpdater.get_analytic_concentration' )
    def test_save_ahf_halos_add_including_v_circ( self, mock_get_analytic_concentration ):
        '''Test that we can write-out files that contain additional information, with that information
        including the circular velocity
        '''

        mock_get_analytic_concentration.side_effect = [ np.arange( 9 ), ]

        # Save halos_add
        sim_data_dir = os.path.join( data_sdir, 'output' )

        self.ahf_updater.save_ahf_halos_add(
            600,
            metafile_dir = data_sdir,
            simulation_data_dir = sim_data_dir,
        )

        # Load halos_add
        self.ahf_updater.get_halos( 600, True )
        self.ahf_updater.get_halos_add( 600, True )

        # Make sure the columns exist
        assert 'Vc(5.0Rstar0.5)' in self.ahf_updater.ahf_halos.columns

