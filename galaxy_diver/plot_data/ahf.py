#!/usr/bin/env python
'''Tools for plotting AHF data.

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import numpy as np

import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patheffects as path_effects

import plotting as gen_plot
import pu_colormaps as pu_cm

import galaxy_diver.read_data.ahf as read_ahf

########################################################################

# For catching default values
default = object()

########################################################################
########################################################################

class AHFPlotter( object ):
  
  def __init__( self, ahf_reader ):
    '''
    Args:
      ahf_reader (read_ahf.AHFReader) : Data reader.
    '''

    self.ahf_reader = ahf_reader

  ########################################################################

  def plot_halos_snapshot( self, snum ):

    fig = plt.figure( figsize=(7,6), facecolor='white' )
    ax = plt.gca()


















