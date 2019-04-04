#!/usr/bin/env python
'''Tools for modifying halo data output files.

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import copy
import glob
import numpy as np
import os
import pandas as pd

import galaxy_dive.galaxy_linker.linker as galaxy_linker
import galaxy_dive.analyze_data.halo_data as halo_data
import galaxy_dive.read_data.metafile as read_metafile
import galaxy_dive.utils.astro as astro_utils
import galaxy_dive.utils.constants as constants
import galaxy_dive.utils.data_constants as data_constants
import galaxy_dive.utils.data_operations as data_operations
import galaxy_dive.utils.utilities as utilities

import galaxy_dive.analyze_data.ahf as ahf
import galaxy_dive.analyze_data.particle_data as particle_data

########################################################################
########################################################################

class HaloUpdater( halo_data.HaloData ):
    '''Class for updating Halo data (smoothing, adding in additional columns, etc)'''

    def __init__( self, *args, **kwargs ):

        self.key_parser = ahf.HaloKeyParser()

        super( HaloUpdater, self ).__init__( *args, **kwargs )

    ########################################################################
    # Get Data Values
    ########################################################################

    def get_accurate_redshift( self, metafile_dir ):
        '''Get a better values of the redshift than what's stored in the Halo filename, by loading them from an external file.

        Args:
            metafile_dir (str): The directory the snapshot_times are stored in.

        Modifies:
            self.mtree_halos (dict of pd.DataFrames): Updates the redshift column
        '''

        # Get the redshift data out
        metafile_reader = read_metafile.MetafileReader( metafile_dir )
        metafile_reader.get_snapshot_times()

        # Replace the old data
        for halo_id in self.mtree_halos.keys():

            mtree_halo = self.mtree_halos[ halo_id ]

            # Read and replace the new redshift
            new_redshift = metafile_reader.snapshot_times['redshift'][ mtree_halo.index ]
            mtree_halo['redshift'] = new_redshift

    ########################################################################

    def get_analytic_concentration( self, metafile_dir, type_of_halo_id='merger_tree' ):
        '''Get analytic values for the halo concentration, using colossus, Benedikt Diemer's cosmology code
        ( https://bitbucket.org/bdiemer/colossus ; http://www.benediktdiemer.com/code/colossus/ ).

        Assumptions:
            - We're using the default formula of Diemer&Kravtstov15
            - We're using the Bryan&Norman1998 version of the virial radius.

        Args:
            metafile_dir (str): The directory the snapshot_times are stored in.
            type_of_halo_id (str): 'merger_tree' if the halo id is a merger tree halo id.
                                                          'halos' if the halo id is a *.AHF_halos halo id.

        Returns
            c_vir (np.array of floats): The concentration, defined as R_vir/r_scale.
        '''

        # Include imports here, because this function may not in general work if colossus is not available,
        # and the rest of the module should still be made useable
        # There may be some warnings here about the version of scipy colossus uses, as opposed to the version galaxy_dive uses
        import colossus.cosmology.cosmology as co_cosmology
        import colossus.halo.concentration as co_concentration

        # Get simulation parameters, for use in creating a cosmology
        metafile_reader = read_metafile.MetafileReader( metafile_dir )
        metafile_reader.get_used_parameters()

        # Setup the cosmology used by the simulations
        sim_cosmo = {
            'flat': True,
            'H0' : float( metafile_reader.used_parameters['HubbleParam'] )*100.,
            'Om0' : float( metafile_reader.used_parameters['Omega0'] ),
            'Ob0' : float( metafile_reader.used_parameters['OmegaBaryon'] ),
            'sigma8' : co_cosmology.cosmologies['WMAP9']['sigma8'], # Use WMAP9 for values we don't store in our simulations explicitly.
            'ns' : co_cosmology.cosmologies['WMAP9']['ns'], # Use WMAP9 for values we don't store in our simulations explicitly.
        }
        cosmo = co_cosmology.setCosmology( 'sim_cosmo', sim_cosmo )

        if type_of_halo_id == 'merger_tree':

            # Loop over all mt halos
            for halo_id in self.mtree_halos.keys():

                # Load the data
                mtree_halo = self.mtree_halos[ halo_id ]

                # Get the concentration out
                c_vir = []
                for m_vir, z in zip( mtree_halo['Mvir'], mtree_halo['redshift'] ):
                    c = co_concentration.concentration( m_vir, 'vir', z, model='diemer15', statistic='median')
                    c_vir.append( c )

                # Turn the concentration into an array
                c_vir = np.array( c_vir )

                # Save the concentration
                mtree_halo['cAnalytic'] = c_vir

        elif type_of_halo_id == 'halos':

            # Get the redshift for the halo file.
            metafile_reader.get_snapshot_times()
            redshift = metafile_reader.snapshot_times['redshift'][self.halos_snum]

            # Get the concentration
            c = co_concentration.concentration( self.halos['Mvir'], 'vir', redshift, model='diemer15', statistic='median')

            return c

    ########################################################################

    def get_mass_radii(
                self,
                mass_fractions,
                simulation_data_dir,
                galaxy_cut,
                length_scale,
        ):
        '''Get radii that enclose a fraction (mass_fractions[i]) of a halo's stellar mass.

        Args:
            mass_fractions (list of floats) :
                Relevant mass fractions.

            simulation_data_dir (str) :
                Directory containing the raw particle data.

            galaxy_cut (float) :
                galaxy_cut*length_scale defines the radius around the center of the halo to look for stars.

            length_scale (str) :
                galaxy_cut*length_scale defines the radius around the center of the halo to look for stars.

        Returns:
            mass_radii (list of np.ndarrays) :
                If M_sum_j = all mass inside galaxy_cut*length_scale for halo j, then mass_radii[i][j] is the radius that
                contains a fraction mass_fractions[i] of M_sum_j.
        '''
                
        # Load the simulation data
        s_data = particle_data.ParticleData(
            simulation_data_dir,
            self.halos_snum,
            ptype = data_constants.PTYPES['star'],
        )

        try:
            particle_positions = s_data.data['P'].transpose()
        # Case where there are no star particles at this redshift.
        except KeyError:
            return [ np.array( [ np.nan, ]*self.halos.index.size ), ]*len( mass_fractions )

        # Find the mass radii
        galaxy_linker_kwargs = {
            'particle_positions' : particle_positions,
            'particle_masses' : s_data.data['M'],
            'snum' : self.halos_snum,
            'redshift' : s_data.redshift,
            'hubble' : s_data.data_attrs['hubble'],
            'galaxy_cut' : galaxy_cut,
            'length_scale' : length_scale,
            'halo_data' : self,
        }
        gal_linker = galaxy_linker.GalaxyLinker( **galaxy_linker_kwargs )
        
        mass_radii = [ gal_linker.get_mass_radius( mass_fraction ) for mass_fraction in mass_fractions ]
            
        return mass_radii

    ########################################################################

    def get_enclosed_mass( self,
        simulation_data_dir,
        ptype,
        galaxy_cut,
        length_scale,
        ):
        '''Get the mass inside galaxy_cut*length_scale for each Halo halo.

        Args:
            simulation_data_dir (str) :
                Directory containing the raw particle data.

            ptype (str) :
                What particle type to get the mass for.

            galaxy_cut (float) :
                galaxy_cut*length_scale defines the radius around the center of the halo within which to get the mass.

            length_scale (str) :
                galaxy_cut*length_scale defines the radius around the center of the halo within which to get the mass.

        Returns:
            mass_inside_all_halos (np.ndarray) :
                mass_inside_all_halos[i] is the mass of particle type ptype inside galaxy_cut*length scale around a galaxy.
        '''

        # Load the simulation data
        s_data = particle_data.ParticleData(
            simulation_data_dir,
            self.halos_snum,
            data_constants.PTYPES[ptype],
        )

        try:
            particle_positions = s_data.data['P'].transpose()
        # Case where there are no star particles at this redshift.
        except KeyError:
            return np.array( [ 0., ]*self.halos.index.size )

        # Find the mass radii
        galaxy_linker_kwargs = {
            'particle_positions' : particle_positions,
            'particle_masses' : s_data.data['M']*constants.UNITMASS_IN_MSUN,
            'snum' : self.halos_snum,
            'redshift' : s_data.redshift,
            'hubble' : s_data.data_attrs['hubble'],
            'galaxy_cut' : galaxy_cut,
            'length_scale' : length_scale,
            'halo_data' : self,
        }
        gal_linker = galaxy_linker.GalaxyLinker( **galaxy_linker_kwargs )

        mass_inside_all_halos = gal_linker.mass_inside_all_halos
        
        # Make sure to put hubble constant back in so we have consistent units.
        mass_inside_all_halos *= s_data.data_attrs['hubble']

        return mass_inside_all_halos

    ########################################################################

    def get_average_quantity_inside_galaxy( self,
        data_key,
        simulation_data_dir,
        ptype,
        galaxy_cut,
        length_scale,
        weight_data_key = 'M',
        fill_value = np.nan,
        ):
        '''Get the mass inside galaxy_cut*length_scale for each Halo halo.

        Args:
            data_key (str) :
                Data key for the quantity to get the average of.

            simulation_data_dir (str) :
                Directory containing the raw particle data.

            ptype (str) :
                What particle type to get the mass for.

            galaxy_cut (float) :
                galaxy_cut*length_scale defines the radius around the center of the halo within which to get the mass.

            length_scale (str) :
                galaxy_cut*length_scale defines the radius around the center of the halo within which to get the mass.

            weight_data_key (str) :
                Data key for the weight to use when averaging.

            fill_value (float) :
                What value to use when the average quantity inside the galaxy is not resolved.

        Returns:
            average_quantity_inside_galaxy (np.ndarray) :
                average_quantity_inside_galaxy[i] is the average value of the requested quantity for particle type ptype
                inside galaxy_cut*length scale around a galaxy.
        '''

        # Load the simulation data
        s_data = particle_data.ParticleData(
            simulation_data_dir,
            self.halos_snum,
            data_constants.PTYPES[ptype],

            # The following values need to be set, because they come into play when a galaxy is centered on halo finder
            # data. That's obviously not the case here...
            centered = True,
            vel_centered = True,
            hubble_corrected = True,
        )

        try:
            particle_positions = s_data.data['P'].transpose()
        # Case where there are no particles of the given ptype at this redshift.
        except KeyError:
            return np.array( [ fill_value, ]*self.halos.index.size )

        # Find the mass radii
        galaxy_linker_kwargs = {
            'particle_positions' : particle_positions,
            'snum' : self.halos_snum,
            'redshift' : s_data.redshift,
            'hubble' : s_data.data_attrs['hubble'],
            'galaxy_cut' : galaxy_cut,
            'length_scale' : length_scale,
            'halo_data' : self,
        }
        gal_linker = galaxy_linker.GalaxyLinker( low_memory_mode=False, **galaxy_linker_kwargs )

        average_quantity_inside_galaxy = gal_linker.weighted_summed_quantity_inside_galaxy(
            s_data.get_data( data_key ),
            s_data.get_data( weight_data_key ),
            fill_value,
        )

        return average_quantity_inside_galaxy

    ########################################################################
    
    def get_circular_velocity( self,
        galaxy_cut,
        length_scale,
        metafile_dir,
        ptypes = data_constants.STANDARD_PTYPES,
        ):
        '''Get the circular velocity at galaxy_cut*length_scale.

        Args:
            galaxy_cut (float) :
                galaxy_cut*length_scale defines the radius around the center of the halo within which to get the mass.

            length_scale (str) :
                galaxy_cut*length_scale defines the radius around the center of the halo within which to get the mass.

            metafile_dir (str) :
                Directory containing metafile data, for getting out the redshift given a snapshot.

            ptypes (list of strs) :
                Particle types to count the mass inside the halo of.

        Returns:
            v_circ (np.ndarray) 
                Circular velocity at galaxy_cut*length_scale using mass from the given ptypes.
        '''

        # Get the redshift, for converting the radius to pkpc/h.
        metafile_reader = read_metafile.MetafileReader( metafile_dir )
        metafile_reader.get_snapshot_times()
        redshift = metafile_reader.snapshot_times['redshift'][self.halos_snum]

        # Get the radius in pkpc/h
        try:
            radius = galaxy_cut*self.halos[length_scale]
        except KeyError:
            radius = galaxy_cut*self.halos_add[length_scale]
        radius /= ( 1. + redshift )

        # Get the mass in Msun/h
        masses = []
        for ptype in ptypes:
            mass_key = self.key_parser.get_enclosed_mass_key( ptype, galaxy_cut, length_scale )
            try:
                ptype_mass = self.halos_add[mass_key]
            except:
                ptype_mass = self.halos[mass_key]
            masses.append( ptype_mass )
        mass = np.array( masses ).sum( axis=0 )

        # Now get the circular velocity out
        # (note that we don't need to bother with converting out the 1/h's, because in this particular case they'll cancel)
        v_circ = astro_utils.circular_velocity( radius, mass )

        return v_circ

    ########################################################################
    # Alter Data
    ########################################################################

    def smooth_mtree_halos(
            self,
            metafile_dir,
            keys_to_smooth = [],
            smooth_kwargs = { 'window_len' : 20, 'window' : 'flat' },
        ):
        '''Make Rvir and Mvir monotonically increasing, to help mitigate artifacts in the Halo-calculated merger tree.
        NOTE: This smooths in *physical* coordinates, so it may not be exactly smooth in comoving coordinates.

        Args:
            metafile_dir (str) :
                The directory the snapshot_times are stored in.

                keys_to_smooth (list of strs) :
                        If given, also smooth the data given by these keys.
                        This smoothing isn't done to assume a monotonic increase, but is
                        a convolve with a moving filter through data_operations.smooth()

                smooth_kwargs (dict) :
                        Specific arguments that determine exactly how the smoothing is
                        done, when also smoothing for specific keys.

        Modifies:
            self.mtree_halos (dict of pd.DataFrames) :
                Changes self.mtree_halos[halo_id]['Rvir'] and self.mtree_halos[halo_id]['Mvir'] to be monotonically increasing.
        '''

        # We need to get an accurate redshift in order to smooth properly
        self.get_accurate_redshift( metafile_dir )

        for halo_id in self.mtree_halos.keys():

            # Load the data
            mtree_halo = self.mtree_halos[ halo_id ]

            # Convert into physical coords for smoothing (we'll still leave the 1/h in place)
            r_vir_phys = mtree_halo['Rvir']/( 1. + mtree_halo['redshift'] )

            # Smooth r_vir
            r_vir_phys_smooth = np.maximum.accumulate( r_vir_phys[::-1] )[::-1]

            # Convert back into comoving and save
            mtree_halo['Rvir'] = r_vir_phys_smooth*( 1. + mtree_halo['redshift'] )

            # Smooth Mvir
            mtree_halo['Mvir'] = np.maximum.accumulate( mtree_halo['Mvir'][::-1] )[::-1]

            for smooth_key in keys_to_smooth:

                original_data = copy.copy( mtree_halo[smooth_key].values )

                smoothed_data = data_operations.smooth(
                        original_data,
                        **smooth_kwargs
                )

                # Replace NaN values with original values, where possible
                smoothed_nan = np.isnan( smoothed_data )
                smoothed_data[smoothed_nan] = original_data[smoothed_nan]

                smooth_save_key = 's' + smooth_key

                mtree_halo[smooth_save_key] = smoothed_data

    ########################################################################

    def include_halos_to_mtree_halos( self ):
        '''While most of the halofile data are contained in *.AHF_halos files, some quantities are stored in
        *.AHF_halos files. These are usually computed manually, external to what's inherent in AHF. This routine adds
        on the the information from these files to the loaded merger tree data (which don't usually include them, because
        they're not inherent to AHF.

        Modifies:
            self.mtree_halos (dict of pd.DataFrames) :
                Adds additional columns contained in *.AHF_halos_add files.
        '''

        for mtree_id, mtree_halo in self.mtree_halos.items():

            print( "Looking at merger tree ID {}".format( mtree_id ) )

            halo_ids = mtree_halo['ID'].values
            snums = mtree_halo.index

            ahf_frames = []
            for snum, halo_id in zip( snums, halo_ids, ):

                print( "Getting data for snapshot {}".format( snum ) )

                self.get_halos( snum )

                # Get the columns we want to add on.
                halofile_columns = set( self.halos.columns )
                mtree_columns = set( mtree_halo.columns )
                columns_to_add = list( halofile_columns - mtree_columns )
                columns_to_add.sort()

                # Now get the values to add
                if self.halos.index.size != 0:
                    full_ahf_row = self.halos.loc[halo_id:halo_id] 
                    ahf_row = full_ahf_row[columns_to_add]

                # Check for edge case, where there isn't an AHF row with specified halo number or there are no more halos
                if ( self.halos.index.size == 0 ) or ( ahf_row.size == 0 ):
                    
                    # Borrow a previous row for formatting
                    ahf_row = copy.copy( ahf_frames[-1] )

                    # Drop the previous row
                    ahf_row = ahf_row.drop( ahf_row.index[0] )

                    # Turn all the values to np.nan, so we know that they're invalid, but the format still works.
                    [ ahf_row.set_value( halo_id, column_name, np.nan ) for column_name in columns_to_add ]

                ahf_frames.append( ahf_row )

            custom_mtree_halo = pd.concat( ahf_frames )

            # Add in the snapshots, and use them as the index
            custom_mtree_halo['snum'] = snums
            custom_mtree_halo = custom_mtree_halo.set_index( 'snum', )

            # Now merge onto the mtree_halo DataFram
            self.mtree_halos[mtree_id] = pd.concat( [ mtree_halo, custom_mtree_halo, ], axis=1 )

    ########################################################################
    # Save Data
    ########################################################################

    def save_mtree_halos( self, tag ):
        '''Save loaded mergertree halo files in a csv file.

        Args:
            tag (str) : If the previous file was for example '/path/to/file/halo_00000.dat',
                                    the new file will be '/path/to/file/halo_00000_{}.dat'.format( tag )
        '''

        for halo_id in self.mtree_halos.keys():

            # Load the data
            mtree_halo = self.mtree_halos[ halo_id ]
            halo_filepath = self.mtree_halo_filepaths[ halo_id ]

            # Create the new filename
            filepath_base, file_ext = os.path.splitext( halo_filepath )
            save_filepath = '{}_{}{}'.format( filepath_base, tag, file_ext )

            mtree_halo.to_csv( save_filepath, sep='\t' )

    ########################################################################

    def save_smooth_mtree_halos(
            self,
            metafile_dir,
            index = None,
            include_halos_add = True,
            include_concentration = False,
            smooth_keys = [ 'Rstar0.5', ],
            **get_mtree_halo_kwargs
        ):
        '''Load halo files, smooth them, and save as a new file e.g., halo_00000_smooth.dat

        Args:
            metafile_dir (str) :
                The directory the metafiles (snapshot_times and used_parameters) are stored in.

            index (str or int) :
                What type of index to use. Defaults to None, which raises an exception. You *must* choose an
                index, to avoid easy mistakes. See get_mtree_halos() for a full description.

            include_concentration (bool):
                Whether or not to add an additional column that gives an analytic value for the
                halo concentration.
        '''
        
        # Load the data
        self.get_mtree_halos( index=index, **get_mtree_halo_kwargs )

        # Include data stored in *AHF_halos_add files.
        if include_halos_add:
            self.include_halos_to_mtree_halos()

        # Smooth the halos
        self.smooth_mtree_halos( metafile_dir, smooth_keys, )

        # Include the concentration, if chosen.
        if include_concentration:
            self.get_analytic_concentration( metafile_dir )

        # Save the halos
        self.save_mtree_halos( 'smooth' )

    ########################################################################

    def save_custom_mtree_halos( self, snums, halo_ids, metafile_dir, ):
        '''Save a custom merger tree.

        Args:
            snums (array-like or int) :
                What snapshots to generate the custom merger tree for.
                If a single integer, then snums will start at that integer and count backwards by single snapshots for the
                length of halo_ids

            halo_ids (array-like) :
                halo_ids[i] is the AHF_halos halo ID for the merger tree halo at snums[i].

            metafile_dir (str) :
                Directory for the metafile (used to get simulation redshift).

        Modifies:
            self.data_dir/halo_00000_custom.dat (text file) : Saves the custom merger tree at this location.
        '''

        if isinstance( snums, int ):
            snums = np.arange( snums, snums - len( halo_ids ), -1 )

        # Concatenate the data
        ahf_frames = []
        for snum, halo_id in zip( snums, halo_ids, ):

            print( "Getting data for snapshot {}".format( snum ) )

            self.get_halos( snum )

            ahf_frames.append( self.halos.loc[halo_id:halo_id] )

        custom_mtree_halo = pd.concat( ahf_frames )

        # Make sure to store the IDs too
        custom_mtree_halo['ID'] = halo_ids

        # Add in the snapshots, and use them as the index
        custom_mtree_halo['snum'] = snums
        custom_mtree_halo = custom_mtree_halo.set_index( 'snum', )

        # Get and save the redshift
        metafile_reader = read_metafile.MetafileReader( metafile_dir )
        metafile_reader.get_snapshot_times()
        custom_mtree_halo['redshift'] = metafile_reader.snapshot_times['redshift'][snums]

        # Save the data
        save_filepath = os.path.join( self.data_dir, 'halo_00000_custom.dat' )
        custom_mtree_halo.to_csv( save_filepath, sep='\t' )

    ########################################################################

    def save_halos_add( self,
        snum,
        include_analytic_concentration = True,
        include_mass_radii = True,
        include_enclosed_mass = True,
        include_average_quantity_inside_galaxy = False,
        include_v_circ = True,
        metafile_dir = None,
        simulation_data_dir = None,
        mass_radii_kwargs = {
            'mass_fractions' : [ 0.5, 0.75, 0.9, ],
            'galaxy_cut' : 0.15,
            'length_scale' : 'Rvir',
        },
        enclosed_mass_ptypes = data_constants.STANDARD_PTYPES,
        enclosed_mass_kwargs = {
            'galaxy_cut' : 5.0,
            'length_scale' : 'Rstar0.5',
        },
        average_quantity_data_keys = [ 'Vx', 'Vy', 'Vz', ],
        average_quantity_inside_galaxy_kwargs = {
            'ptype' : 'star',
            'galaxy_cut' : 5.0,
            'length_scale' : 'Rstar0.5',
        },
        v_circ_kwargs = {
            'galaxy_cut' : 5.0,
            'length_scale' : 'Rstar0.5',
        },
        verbose = False,
        ):
        '''Save additional columns that would be part of *.AHF_halos files, if that didn't break AHF.

        Args:
            snum (int) :
                Snapshot number to load.

            include_analytic_concentration (bool) :
                Include analytic concentration as one of the columns?

            include_mass_radii (bool) :
                Include radius that include some fraction of a particle's mass as one of the columns?

            include_enclosed_mass (bool) :
                Include the mass enclosed in some specified radii as one of the columns?

            include_average_quantity_inside_galaxy (bool) :
                Include the average value inside each galaxy for the quantities listed in average_quantity_data_keys?

            include_v_circ (bool) :
                Include the circular mass at some specified radii as one of the columns?

            metafile_dir (str) :
                The directory the metafiles (snapshot_times and used_parameters) are stored in.

            simulation_data_dir (str) :
                Directory containing the simulation data (used for getting the position and masses of the star particles).

            mass_radii_kwargs (dict) :
                Keyword args for self.get_mass_radii()

            enclosed_mass_ptypes (list of strs) :
                Particle types to get the mass inside a radii of.

            enclosed_mass_kwargs (dict) :
                Keyword args for self.get_enclosed_mass()

            average_quantity_data_keys (list of strs) :
                What data keys (to be passed to a standard ParticleData.get_data() function) to get the average quantity for?

            average_quantity_kwargs (dict) :
                Keyword args for self.get_average_quantity_inside_galaxy()

            v_circ_kwargs (dict) :
                Keyword args for self.get_circular_velocity()

            verbose (bool) :
                If True, print out additional information about how the steps are progressing.
        '''

        print('Saving *.AHF_halos_add for snum {}'.format( snum ))

        # Load the AHF_halos data
        self.get_halos( snum )

        # Figure out if there are any valid halos at this redshift if not, then a *lot* can be skipped.
        # TODO: Don't hard-code this in....
        valid_halos = self.halos['n_star'] >= 10
        no_valid_halos = valid_halos.sum() == 0
        blank_array = np.array( [ np.nan, ]*self.halos.index.size )

        # Create AHF_halos add
        self.halos_add = pd.DataFrame( {}, index=self.halos.index )
        self.halos_add.index.names = ['ID']

        # Get the analytic concentration
        if include_analytic_concentration:
            if verbose:
                print( "Including Analytic Concentration..." )
            self.halos_add['cAnalytic'] = self.get_analytic_concentration( metafile_dir, type_of_halo_id='halos' )

        # Get characteristic radii
        if include_mass_radii:
            if verbose:
                print( "Including Mass Radii..." )

            if no_valid_halos:
                mass_radii = [ blank_array, ]*len( mass_radii_kwargs['mass_fractions'] )
            else:
                mass_radii = self.get_mass_radii( simulation_data_dir = simulation_data_dir, **mass_radii_kwargs )

            for i, mass_fraction in enumerate( mass_radii_kwargs['mass_fractions'] ):
                label = 'Rstar{}'.format( mass_fraction )
                self.halos_add[label] = mass_radii[i]

        # Get mass enclosed in a particular radius
        if include_enclosed_mass:
            if verbose:
                print( "Including Enclosed Mass..." )
            for i, ptype in enumerate( enclosed_mass_ptypes ):

                if no_valid_halos:
                    halo_masses = blank_array
                else:
                    halo_masses = self.get_enclosed_mass( simulation_data_dir, ptype, **enclosed_mass_kwargs )

                label = self.key_parser.get_enclosed_mass_key( ptype, enclosed_mass_kwargs['galaxy_cut'], \
                                                                                                              enclosed_mass_kwargs['length_scale'], )
                self.halos_add[label] = halo_masses

        # Get average quantity inside each galaxy (for halos that have galaxies)
        if include_average_quantity_inside_galaxy:
            if verbose:
                print( "Including Average Quantities..." )

            for i, data_key in enumerate( average_quantity_data_keys ):

                if verbose:
                    print( "  Finding average {}...".format( data_key ) )

                if no_valid_halos:
                    average_quantity = blank_array
                else:
                    average_quantity = self.get_average_quantity_inside_galaxy(
                        data_key,
                        simulation_data_dir,
                        **average_quantity_inside_galaxy_kwargs
                    )

                label = self.key_parser.get_average_quantity_key(
                    data_key,
                    average_quantity_inside_galaxy_kwargs['ptype'],
                    average_quantity_inside_galaxy_kwargs['galaxy_cut'],
                    average_quantity_inside_galaxy_kwargs['length_scale'],
                )

                self.halos_add[label] = average_quantity

        # Get circular velocity at a particular radius
        if include_v_circ:
            if verbose:
                print( "Including Circular Velocity..." )
            v_circ = self.get_circular_velocity( metafile_dir=metafile_dir, **v_circ_kwargs )

            label = self.key_parser.get_velocity_at_radius_key(
                'Vc',
                v_circ_kwargs['galaxy_cut'],
                v_circ_kwargs['length_scale']
            ) 

            self.halos_add[label] = v_circ

        # Save AHF_halos add
        save_filepath = '{}_add'.format( self.halos_path )
        self.halos_add.to_csv( save_filepath, sep='\t' )

    ########################################################################

    def save_multiple_halos_adds( self, metafile_dir, snum_start, snum_end, snum_step ):
        '''Save additional columns that would be part of *.AHF_halos files, if that didn't break AHF.
        Do this for every *.AHF_halos file in self.data_dir.

        Args:
            metafile_dir (str): The directory the metafiles (snapshot_times and used_parameters) are stored in.
            snum_start (int): Starting snapshot.
            snum_end (int): Ending snapshot.
            snum_step (int): Step between snapshots.
        '''

        # Save the halos
        for snum in range( snum_start, snum_end+snum_step, snum_step):

            # Save the data
            self.save_halos_add( snum, metafile_dir )

