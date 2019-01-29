#!/usr/bin/env python
'''Routines for reading in GADGET/GIZMO snapshots.

@author: Zach Hafen, Phil Hopkins, Claude-Andre Faucher-Giguere,
  and maybe others (this file's been passed around a lot).
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import glob
import h5py as h5py
import os

import numpy as np
import os.path
import scipy.interpolate as interpolate
import scipy.optimize as optimize
import math

########################################################################
########################################################################

def readsnap( sdir, snum, ptype, load_additional_ids=0, snapshot_name='snapshot', extension='.hdf5', h0=0,cosmological=0, skip_bh=0, four_char=0, header_only=0, loud=0):
    '''Reads in a particle snapshot.

    Args:
      sdir (str): Simulation directory to load from.

      snum (int): Snapshot to load.

      ptype (int): Particle type to load. Typically, 0=gas, 1=hi-res dark matter, 2=low-res dark matter, 4=stars.

      load_additional_ids (bool): Whether or not to also load the fields 'ParticleChildIDsNumber' and 'ParticleIDGenerationNumber'.

      snapshot_name (str): The base name for the snapshot.

      extension (str): The extension for the snapshot.

      h0 (bool): If True, use present day values.

      cosmological (bool): If True, convert from comoving to physical units, and return in physical units, with factors of h accounted for.

      skip_bh (bool): If True, don't read BH-specific fields for particles of type 5.

      four_char (bool): If True, then snapshot files have 4 digits, instead of three (e.g. 0020 instead of 020)

      header_only (bool): If True, return only the header, in stead of the full data.

      loud (bool): If True, print additional information.

    Returns:
      P (dict): A dictionary of numpy arrays containing the particle data.
    '''
    
    if (ptype<0): return {'k':-1};
    if (ptype>5): return {'k':-1};

    fname,fname_base,fname_ext = check_if_filename_exists(sdir,snum,\
        snapshot_name=snapshot_name,extension=extension,four_char=four_char)
    if(fname=='NULL'): 
      raise Exception( "Couldn't read snapshot {} in {}.\nCheck if this is the right location.".format( snum, sdir ) )
    if(loud==1): print('loading file : '+fname)

    ## open file and parse its header information
    nL = 0 # initial particle point to start at 
    if(fname_ext=='.hdf5'):
        file = h5py.File(fname,'r') # Open hdf5 snapshot file
        header_master = file["Header"] # Load header dictionary (to parse below)
        header_toparse = header_master.attrs
    else:
        file = open(fname) # Open binary snapshot file
        header_toparse = load_gadget_binary_header(file)

    npart = header_toparse["NumPart_ThisFile"]
    massarr = header_toparse["MassTable"]
    time = header_toparse["Time"]
    redshift = header_toparse["Redshift"]
    flag_sfr = header_toparse["Flag_Sfr"]
    flag_feedbacktp = header_toparse["Flag_Feedback"]
    npartTotal = header_toparse["NumPart_Total"]
    flag_cooling = header_toparse["Flag_Cooling"]
    numfiles = header_toparse["NumFilesPerSnapshot"]
    boxsize = header_toparse["BoxSize"]
    omega_matter = header_toparse["Omega0"]
    omega_lambda = header_toparse["OmegaLambda"]
    hubble = header_toparse["HubbleParam"]
    flag_stellarage = header_toparse["Flag_StellarAge"]
    flag_metals = header_toparse["Flag_Metals"]

    hinv=1.
    if (h0==1):
        hinv=1./hubble
    ascale=1.
    if (cosmological==1):
        ascale=time
        hinv=1./hubble
    if (cosmological==0): 
        time*=hinv

    # CAFG: changed order of the following two lines, so that header
    # info is returned even if no particles.
    if (header_only==1):
      file.close()
      return {'k':0,'time':time,'hubble':hubble,'redshift':redshift, 'npart':npartTotal[ptype]}
    if (npartTotal[ptype]<=0): file.close(); return {'k':-1};


    # initialize variables to be read
    pos=np.zeros([npartTotal[ptype],3],dtype=float)
    vel=np.copy(pos)
    ids=np.zeros([npartTotal[ptype]],dtype=np.long)
    if load_additional_ids:
      child_ids=np.zeros([npartTotal[ptype]],dtype=np.long)
      id_gens=np.zeros([npartTotal[ptype]],dtype=np.long)
    mass=np.zeros([npartTotal[ptype]],dtype=float)
    potential = np.copy( mass )
    if (ptype==0):
        ugas=np.copy(mass)
        rho=np.copy(mass)
        hsml=np.copy(mass) 
        if (flag_cooling>0): 
            nume=np.copy(mass)
            numh=np.copy(mass)
        if (flag_sfr>0): 
            sfr=np.copy(mass)
    if (ptype==0 or ptype==4) and (flag_metals > 0):
        metal=np.zeros([npartTotal[ptype],flag_metals],dtype=float)
    if (ptype==4) and (flag_sfr>0) and (flag_stellarage>0):
        stellage=np.copy(mass)
    if (ptype==5) and (skip_bh==0):
        bhmass=np.copy(mass)
        bhmdot=np.copy(mass)

    # loop over the snapshot parts to get the different data pieces
    for i_file in range(numfiles):
        if (numfiles>1):
            file.close()
            fname = fname_base+'.'+str(i_file)+fname_ext
            if(fname_ext=='.hdf5'):
                file = h5py.File(fname,'r') # Open hdf5 snapshot file
            else:
                file = open(fname) # Open binary snapshot file
                header_toparse = load_gadget_binary_header(file)
                
        if (fname_ext=='.hdf5'):
            input_struct = file
            npart = file["Header"].attrs["NumPart_ThisFile"]
            ptype_full_name = "PartType{}".format( ptype )
            bname = ptype_full_name + "/"
        else:
            npart = header_toparse['NumPart_ThisFile']
            input_struct = load_gadget_binary_particledat(file, header_toparse, ptype, skip_bh=skip_bh)
            bname = ''
            
        
        # now do the actual reading
        
        # CAFG: there can be an error here when particles of a certain
        # type (e.g., stars) are only in a subset of all files that
        # constitute the snapshot. Added 'if' on npart[ptype]>0.

        if npart[ptype]>0:
            nR=nL + npart[ptype]
            pos[nL:nR,:]=input_struct[bname+"Coordinates"]
            vel[nL:nR,:]=input_struct[bname+"Velocities"]
            ids[nL:nR]=input_struct[bname+"ParticleIDs"]
            mass[nL:nR]=massarr[ptype]

            potential_stored = 'Potential' in input_struct[ptype_full_name].keys()
            if potential_stored:
                potential[nL:nR] = input_struct[bname+"Potential"]

            if (massarr[ptype] <= 0.):
                mass[nL:nR]=input_struct[bname+"Masses"]
            if (ptype==0):
                ugas[nL:nR]=input_struct[bname+"InternalEnergy"]
                rho[nL:nR]=input_struct[bname+"Density"]
                hsml[nL:nR]=input_struct[bname+"SmoothingLength"]
                if (flag_cooling > 0): 
                    nume[nL:nR]=input_struct[bname+"ElectronAbundance"]
                    numh[nL:nR]=input_struct[bname+"NeutralHydrogenAbundance"]
                if (flag_sfr > 0):
                    sfr[nL:nR]=input_struct[bname+"StarFormationRate"]
            if (ptype==0 or ptype==4) and (flag_metals > 0):
                metal_t=input_struct[bname+"Metallicity"]
                if (flag_metals > 1):
                    if (metal_t.shape[0] != npart[ptype]): 
                        metal_t=np.transpose(metal_t)
                else:
                    metal_t=np.reshape(np.array(metal_t),(np.array(metal_t).size,1))
                metal[nL:nR,:]=metal_t
            if (ptype==4) and (flag_sfr>0) and (flag_stellarage>0):
                stellage[nL:nR]=input_struct[bname+"StellarFormationTime"]
            if (ptype==5) and (skip_bh==0):
                bhmass[nL:nR]=input_struct[bname+"BH_Mass"]
                bhmdot[nL:nR]=input_struct[bname+"BH_Mdot"]

            if load_additional_ids:
              child_ids[nL:nR]=input_struct[bname+"ParticleChildIDsNumber"]
              id_gens[nL:nR]=input_struct[bname+"ParticleIDGenerationNumber"]

            nL = nR # sets it for the next iteration	

        ## correct to same ID as original gas particle for new stars, if bit-flip applied
    if ((np.min(ids)<0) | (np.max(ids)>1.e9)):
        bad = (ids < 0) | (ids > 1.e9)
        ids[bad] += (int(1) << 31)

    # do the cosmological conversions on final vectors as needed
    pos *= hinv*ascale # snapshot units are comoving
    boxsize *= hinv*ascale
    mass *= hinv
    vel *= np.sqrt(ascale) # remember gadget's weird velocity units!
    if (ptype==0):
        rho *= (hinv/((ascale*hinv)**3))
        hsml *= hinv*ascale
    if (ptype==4) and (flag_sfr>0) and (flag_stellarage>0) and (cosmological==0):
        stellage *= hinv
    if (ptype==5) and (skip_bh==0):
        bhmass *= hinv

    file.close();
    if (ptype==0):
       ret_dict = {'hubble':hubble,'boxsize':boxsize,'time':time,'redshift':redshift,'flag_sfr':flag_sfr,'flag_feedbacktp':flag_feedbacktp,'flag_cooling':flag_cooling,'omega_matter':omega_matter,'omega_lambda':omega_lambda,'flag_stellarage':flag_stellarage,'flag_metals':flag_metals,'k':1,'p':pos,'v':vel,'m':mass,'id':ids,'u':ugas,'rho':rho,'h':hsml,'ne':nume,'nh':numh,'sfr':sfr}
       if flag_metals>0:
         ret_dict['z'] = metal
    elif (ptype==4):
        ret_dict = {'hubble':hubble,'boxsize':boxsize,'time':time,'redshift':redshift,'flag_sfr':flag_sfr,'flag_feedbacktp':flag_feedbacktp,'flag_cooling':flag_cooling,'omega_matter':omega_matter,'omega_lambda':omega_lambda,'flag_stellarage':flag_stellarage,'flag_metals':flag_metals,'k':1,'p':pos,'v':vel,'m':mass,'id':ids,'age':stellage}
        if flag_metals>0:
          ret_dict['z'] = metal
    elif (ptype==5) and (skip_bh==0):
        ret_dict = {'hubble':hubble,'boxsize':boxsize,'time':time,'redshift':redshift,'flag_sfr':flag_sfr,'flag_feedbacktp':flag_feedbacktp,'flag_cooling':flag_cooling,'omega_matter':omega_matter,'omega_lambda':omega_lambda,'flag_stellarage':flag_stellarage,'flag_metals':flag_metals,'k':1,'p':pos,'v':vel,'m':mass,'id':ids,'mbh':bhmass,'mdot':bhmdot}

    else:
      ret_dict = {'hubble':hubble,'boxsize':boxsize,'time':time,'redshift':redshift,'flag_sfr':flag_sfr,'flag_feedbacktp':flag_feedbacktp,'flag_cooling':flag_cooling,'omega_matter':omega_matter,'omega_lambda':omega_lambda,'flag_stellarage':flag_stellarage,'flag_metals':flag_metals,'k':1,'p':pos,'v':vel,'m':mass,'id':ids,}

    # Include optional ids
    if load_additional_ids:
      ret_dict['child_id'] = child_ids
      ret_dict['id_gen'] = id_gens

    if potential_stored:
      ret_dict['potential'] = potential/ascale # Convert to physical km^2/s^2

    return ret_dict

########################################################################

def check_if_filename_exists(sdir,snum,snapshot_name='snapshot',extension='.hdf5',four_char=0):
    '''Check if a file name exists, and if so return information about it.

    Args:
      sdir (str): Simulation directory to load from.
      snum (int): Snapshot to load.
      snapshot_name (str): The base name for the snapshot.
      extension (str): The extension for the snapshot.
      four_char (bool): If True, then snapshot files have 4 digits, instead of three (e.g. 0020 instead of 020)

    Returns:
      fname_found (bool): If true, the filename was found.
      fname_base_found (str): The base for the found filename.
      fname_ext (str): The extension for the found filename.
    '''
    for extension_touse in [extension,'.bin','']:
        fname=sdir+'/'+snapshot_name+'_'
        ext='00'+str(snum);
        if (snum>=10): ext='0'+str(snum)
        if (snum>=100): ext=str(snum)
        if (four_char==1): ext='0'+ext
        if (snum>=1000): ext=str(snum)
        fname+=ext
        fname_base=fname

        s0=sdir.split("/"); snapdir_specific=s0[len(s0)-1];
        if(len(snapdir_specific)<=1): snapdir_specific=s0[len(s0)-2];

        ## try several common notations for the directory/filename structure
        fname=fname_base+extension_touse;
        if not os.path.exists(fname): 
            ## is it a multi-part file?
            fname=fname_base+'.0'+extension_touse;
        if not os.path.exists(fname): 
            ## is the filename 'snap' instead of 'snapshot'?
            fname_base=sdir+'/snap_'+ext; 
            fname=fname_base+extension_touse;
        if not os.path.exists(fname): 
            ## is the filename 'snap' instead of 'snapshot', AND its a multi-part file?
            fname=fname_base+'.0'+extension_touse;
        if not os.path.exists(fname): 
            ## is the filename 'snap(snapdir)' instead of 'snapshot'?
            fname_base=sdir+'/snap_'+snapdir_specific+'_'+ext; 
            fname=fname_base+extension_touse;
        if not os.path.exists(fname): 
            ## is the filename 'snap' instead of 'snapshot', AND its a multi-part file?
            fname=fname_base+'.0'+extension_touse;
        if not os.path.exists(fname): 
            ## is it in a snapshot sub-directory? (we assume this means multi-part files)
            fname_base=sdir+'/snapdir_'+ext+'/'+snapshot_name+'_'+ext; 
            fname=fname_base+'.0'+extension_touse;
        if not os.path.exists(fname): 
            ## is it in a snapshot sub-directory AND named 'snap' instead of 'snapshot'?
            fname_base=sdir+'/snapdir_'+ext+'/'+'snap_'+ext; 
            fname=fname_base+'.0'+extension_touse;
        if not os.path.exists(fname): 
            ## wow, still couldn't find it... ok, i'm going to give up!
            fname_found = 'NULL'
            fname_base_found = 'NULL'
            fname_ext = 'NULL'
            continue;
        fname_found = fname;
        fname_base_found = fname_base;
        fname_ext = extension_touse
        break; # filename does exist! 
    return fname_found, fname_base_found, fname_ext;



def load_gadget_binary_header(f):
    '''Get the header information for a binary snapshot file.
  
    Args:
      f (binary file): The file to read the information from.

    Returns:
      header_info (dict): A dictionary of info contained in the header.
    '''
    ### Read header.
    import array
    # Skip 4-byte integer at beginning of header block.
    f.read(4)
    # Number of particles of each type. 6*unsigned integer.
    Npart = array.array('I')
    Npart.fromfile(f, 6)
    # Mass of each particle type. If set to 0 for a type which is present, 
    # individual particle masses from the 'mass' block are used instead.
    # 6*double.
    Massarr = array.array('d')
    Massarr.fromfile(f, 6)
    # Expansion factor (or time, if non-cosmological sims) of output. 1*double. 
    a = array.array('d')
    a.fromfile(f, 1)
    a = a[0]
    # Redshift of output. Should satisfy z=1/a-1. 1*double.
    z = array.array('d')
    z.fromfile(f, 1)
    z = float(z[0])
    # Flag for star formation. 1*int.
    FlagSfr = array.array('i')
    FlagSfr.fromfile(f, 1)
    # Flag for feedback. 1*int.
    FlagFeedback = array.array('i')
    FlagFeedback.fromfile(f, 1)
    # Total number of particles of each type in the simulation. 6*int.
    Nall = array.array('i')
    Nall.fromfile(f, 6)
    # Flag for cooling. 1*int.
    FlagCooling = array.array('i')
    FlagCooling.fromfile(f, 1)
    # Number of files in each snapshot. 1*int.
    NumFiles = array.array('i')
    NumFiles.fromfile(f, 1)
    # Box size (comoving kpc/h). 1*double.
    BoxSize = array.array('d')
    BoxSize.fromfile(f, 1)
    # Matter density at z=0 in units of the critical density. 1*double.
    Omega0 = array.array('d')
    Omega0.fromfile(f, 1)
    # Vacuum energy density at z=0 in units of the critical density. 1*double.
    OmegaLambda = array.array('d')
    OmegaLambda.fromfile(f, 1)
    # Hubble parameter h in units of 100 km s^-1 Mpc^-1. 1*double.
    h = array.array('d')
    h.fromfile(f, 1)
    h = float(h[0])
    # Creation times of stars. 1*int.
    FlagAge = array.array('i')
    FlagAge.fromfile(f, 1)
    # Flag for metallicity values. 1*int.
    FlagMetals = array.array('i')
    FlagMetals.fromfile(f, 1)

    # For simulations that use more than 2^32 particles, most significant word 
    # of 64-bit total particle numbers. Otherwise 0. 6*int.
    NallHW = array.array('i')
    NallHW.fromfile(f, 6)

    # Flag that initial conditions contain entropy instead of thermal energy
    # in the u block. 1*int.
    flag_entr_ics = array.array('i')
    flag_entr_ics.fromfile(f, 1)

    # Unused header space. Skip to particle positions.
    f.seek(4+256+4+4)

    return {'NumPart_ThisFile':Npart, 'MassTable':Massarr, 'Time':a, 'Redshift':z, \
    'Flag_Sfr':FlagSfr[0], 'Flag_Feedback':FlagFeedback[0], 'NumPart_Total':Nall, \
    'Flag_Cooling':FlagCooling[0], 'NumFilesPerSnapshot':NumFiles[0], 'BoxSize':BoxSize[0], \
    'Omega0':Omega0[0], 'OmegaLambda':OmegaLambda[0], 'HubbleParam':h, \
    'Flag_StellarAge':FlagAge[0], 'Flag_Metals':FlagMetals[0], 'Nall_HW':NallHW, \
    'Flag_EntrICs':flag_entr_ics[0]}

########################################################################
########################################################################

def gas_mu(num_e):
    '''Calculates the mean weight from gadget.'''

    XH=0.76 # we track this with metal species now, could do better...
    yhelium=(1.-XH)/(4.*XH)
    return (1.+4.*yhelium)/(1.+yhelium+num_e)

########################################################################

def gas_temperature(u, num_e, keV=0):
    '''Returns gas particles temperature in Kelvin. Originally from PFH's gadget lib (maybe somewhere else before?).'''

    g_gamma= 5.0/3.0
    g_minus_1= g_gamma-1.0
    PROTONMASS = 1.6726e-24
    BoltzMann_ergs= 1.3806e-16
    UnitMass_in_g= 1.989e43 # 1.0e10 solar masses
    UnitEnergy_in_cgs= 1.989e53
    # note gadget units of energy/mass = 1e10 ergs/g,
    # this comes into the formula below

    mu = gas_mu(num_e);
    MeanWeight= mu*PROTONMASS
    Temp= MeanWeight/BoltzMann_ergs * g_minus_1 * u * 1.e10

    # do we want units of keV?  (0.001 factor converts from eV to keV)
    if (keV==1):
        BoltzMann_keV = 8.617e-8;
        Temp *= BoltzMann_keV;

    return Temp

########################################################################
########################################################################
'''Unfinished below this line. TODO.'''

def read_snapshot( sdir, snum, ptype, load_additional_ids=False, cosmological=True, ):
  '''
  Note: This is created as a function, not a class, because I'm encountering memory issues that may be related to
  classes persisting in memory. -- Zach Hafen
  '''

  filepaths = get_snapshot_filepaths( sdir, snum )

  if cosmological:
    convert_data_to_physical_units()

  return data, data_attrs

########################################################################

def get_snapshot_filepaths( sdir, snum ):
  '''Get the filepath(s) to a snapshot located in a specified directory.

  Args:
    sdir (str) : Directory containing the simulation snapshot.
    snum (int) : Snapshot number.

  Returns:
    filepaths (list of strs) : Filepath(s) to the snapshot files.
  '''

  filecase_filepath = os.path.join( sdir, 'snapshot_{:03d}.hdf5'.format( snum ) )
  is_file = os.path.isfile( filecase_filepath )

  dir_filepath = os.path.join( sdir, 'snapdir_{:03d}'.format( snum ) )
  is_dir = os.path.isdir( dir_filepath )

  if not ( is_file or is_dir):
    raise NameError( "Snapshot {:03d} not found in {}".format( snum, sdir ) )
    
  if is_file:
    return [ filecase_filepath, ]
  elif is_dir:
    return glob.glob( '{}/*'.format( dir_filepath ) )

########################################################################

def read_snapshot_files( filepaths ):


  data_attrs = {}
  for i, filepath in enumerate( filepaths ):
    with h5py.File( filepath, 'r' ) as f:

      # First time data
      if i == 0:

        pass

      # Get the attrs/header information

########################################################################

def setup_data_struct( ptype, flag_metals ):
  '''
  Args:
    ptype (int) : Particle type to load.
    flag_metals (int) : How many metal species there are.
  '''

  # Setup data storage.
  data = {
    'P' : np.zeros( [ n_particles, 3], dtype=np.float64 ),
    'V' : np.zeros( [ n_particles, 3], dtype=np.float32 ),
    'ID' : np.zeros( n_particles, dtype=np.long ),
    'M' : np.zeros( n_particles, dtype=np.float32 ),
  }
  if load_additional_ids:
    data['ChildID'] = np.zeros( n_particles, dtype=np.long )
    data['IDGen'] = np.zeros( n_particles, dtype=np.long )

  # Information that varies depending on ptype
  if ptype == 0:
    data['U'] = np.zeros( n_particles, dtype=np.float32 )
    data['Den'] = np.zeros( n_particles, dtype=np.float32 )
    data['XHI'] = np.zeros( n_particles, dtype=np.float32 )
  if ( ptype == 0 ) or ( ptype == 4 ):
    data['Z'] = np.zeros( (n_particles, flag_metals), dtype=np.float32 )
  if ( ptype == 4):
    data['BirthTime'] = np.zeros( n_particles, dtype=np.float32 )

  return data
