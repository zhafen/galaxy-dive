#!/usr/bin/env python
'''Constants for general use in analyzing simulations.

@author: Zach Hafen, Daniel Angles-Alcazar
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import numpy as np

########################################################################
# Physical Constants
########################################################################

Z_MASSFRAC_SUN           = 0.02          # Traditional value
Z_MASSFRAC_SUN_ALTERNATE = 0.0143        # Asplund et al. (2009)

HUBBLE                   = 3.2407789e-18 # in h/sec
GAMMA                    = 5.0/3.0
GAMMA_MINUS1             = GAMMA - 1.

########################################################################
# Physical constants in CGS

MSUN                     = 1.989e33       # in g
SPEED_OF_LIGHT           = 2.99792458e10  # speed of light (cm s^-1)
G_UNIV                   = 6.672e-8       # [ cm^3 g^-1 s^-2 ]
K_B                      = 1.3806488e-16  # Boltzmann constant in erg K^-1
PROTON_MASS              = 1.6726e-24     # in g
ELECTRON_MASS            = 9.10938356e-28 # Electron mass in g.
ELECTRON_CHARGE          = 4.8e-10        # Charge of electron in StatC
N_A                      = 6.0221413e23   # Avogadro's number.
SIGMA_T                  = 6.6524e-25     # cm^(-2) Thomson cross-section

########################################################################
# Physical constants in SI

MSUN_SI                  = 1.989e30       # Mass of the sun in kg
SPEED_OF_LIGHT_SI        = 2.99792458e8   # Speed of light in m/s
G_SI                     = 6.67384e-11    # Gravitational constant in m^3/(kg s^2)
K_B_SI                   = 1.3806488e-23  # Boltzmann constant in m^2 kg s^-2 K^-1
PROTON_MASS_SI           = 1.67262178e-27 # Proton mass in kg.
ELECTRON_MASS_SI         = 9.10938356e-31 # Electron mass in kg.
ELECTRON_CHARGE_SI       = 1.6e-19        # Charge of electron in Coulombs

########################################################################
# Absorption constants

SIGMA_HI                 = 6.30e-18         # HI absorption cross section.
LAMBDA_LY_ALPHA          = 1215.67          # Lyman-alpha wavelength (A).
F_LY_ALPHA               = 0.416            # oscillator strength for Lyman alpha
NU_LY_ALPHA              = 2.46606774865e15 # frequency of Lyman alpha (s^-1)

########################################################################
# Conversion Constants
########################################################################

CM_PER_KM                = 1e5
CM_PER_MPC               = 3.085678e24
CM_PER_KPC               = 3.085678e21
M_PER_KPC                = 3.08567758e19
KM_PER_KPC               = 3.08567758e16

SEC_PER_YEAR             = 3.15569e7

########################################################################
# Code Unit Conversions

UNITMASS_IN_G            = 1.989e43    # 1.e10/h solar masses
UNITMASS_IN_MSUN         = 1e10
UNITVELOCITY_IN_CM_PER_S = 1e5         # 1 km/s
UNITLENGTH_IN_CM         = 3.085678e21 # 1 kpc/h
UNITLENGTH_IN_MPC        = 0.001           
UNITTIME_IN_S            = 3.08568e+16 # seconds / h
UNITTIME_IN_GYR          = 0.977814
UNITTIME_IN_MYR          = 977.814
UNITDENSITY_IN_CGS       = 6.76991e-22
UNITDENSITY_IN_NUMDEN    = 404.768     # Conversion from 10^10 Msun/(proper kpc)^3 to number of m_p per (proper cm)^3
UNITENERGY_IN_CGS        = 1.989e53
UNITMDOT_IN_MSUN_PER_YR  = 10.2269
UNITG_UNIV               = 43007.1

########################################################################
# Default Cosmology
########################################################################

HUBBLE_PARAM             = 0.702
RHO_CRIT                 = 8.62213e-30 # in cgs
OMEGA_0                  = 0.272
OMEGA_LAMBDA             = 0.728
OMEGA_BARYON             = 0.0455
