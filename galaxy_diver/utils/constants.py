#!/usr/bin/env python
'''Constants for general use in analyzing simulations.

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

# Unit conversion constants

kpc_to_cm = 3.08567758e21
kpc_to_m = 3.08567758e19
kpc_to_km = 3.08567758e16

Z_massfrac_sun = 0.0143 # Asplund et al. (2009)

gas_den_to_nb = 404.768  # Conversion from 10^10 Msun/(proper kpc)^3 to number of m_p per (proper cm)^3

Msun_to_kg = 1.989e30 # Mass of the sun in kg

########################################################################
# Physical constants
########################################################################

# Primary physics constants in cgs

c = 2.99792458e10 # Speed of light in cm/s
g = 6.67259e-8 # Gravitational constant in cm^3/(g s^2)
k_b = 1.3806488e-16 # Boltzmann constant in erg K^-1
m_p = 1.67262178e-24 # Proton mass in g.
m_e = 9.10938356e-28 # Electron mass in g.
q_e = (1.6e-19)*(3.0e9) # Charge of electron in StatC
n_a = 6.0221413e23 # Avogadro's number.

########################################################################

# Primary physics constants in SI.

C = 2.99792458e8 # Speed of light in m/s
G = 6.67384e-11 # Gravitational constant in m^3/(kg s^2)
K_B = 1.3806488e-23 # Boltzmann constant in m^2 kg s^-2 K^-1
M_P = 1.67262178e-27 # Proton mass in kg.
M_E = 9.10938356e-31 # Electron mass in kg.
Q_E = 1.6e-19 # Charge of electron in Coulombs

########################################################################

# Absorption constants.

sigma_HI = 6.30e-18 # HI absorption cross section.
lambda_Ly_alpha = 1215.67 # Lyman-alpha wavelength (A).
f_Ly_alpha = 0.416 # oscillator strength for Lyman alpha
nu_Ly_alpha = c/(lambda_Ly_alpha*(1.e-8)) # frequency of Lyman alpha (s^-1)
