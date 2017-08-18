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
import matplotlib.patches as mpatches
import matplotlib.transforms as transforms

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

  def plot_halos_snapshot( self,
                           snum,
                           ax = default,
                           type_of_halo_id = 'ahf_halos',
                           color = 'w',
                           linestyle = 'solid',
                           outline = True,
                           center = True,
                           hubble_param = default,
                           radius_fraction = 1.,
                           ):
    '''Plot the halos as circles at their respective locations.
    
    Args:
      snum (int) : Snapshot to plot at.
      type_of_halo_id (str): Should the merger tree halos be plotted, or the first 50 ahf halos at that redshift?
      ax (axis object) : Axis to use. If default, one is created.
      color (str) : What color should the circle be?
      linestyle (str) : What linestyle to use.
      outline (bool) : Should the circles be outlined for increased visibility?
      center (bool) : Should the plot be centered at the most massive halo at z=0?
      hubble_param (float) : If given, the positions will be converted to physical kpc.
      radius_fraction (float) : The circles will be radius_fraction*r_scale
    '''

    if ax is default:
      fig = plt.figure( figsize=(7,6), facecolor='white' )
      ax = plt.gca()
      
    if type_of_halo_id == 'ahf_halos':
      self.ahf_reader.get_halos( snum )
      x_pos = self.ahf_reader.ahf_halos['Xc'][:50]
      y_pos = self.ahf_reader.ahf_halos['Yc'][:50]

      r_vir = self.ahf_reader.ahf_halos['Rvir'][:50]

      self.ahf_reader.get_halos_add( snum )
      c_analytic = self.ahf_reader.ahf_halos_add['cAnalytic'][:50]

    elif type_of_halo_id == 'merger_tree':
      x_pos = self.ahf_reader.get_mtree_halo_quantity( 'Xc', snum, self.ahf_reader.index, self.ahf_reader.tag )
      y_pos = self.ahf_reader.get_mtree_halo_quantity( 'Yc', snum, self.ahf_reader.index, self.ahf_reader.tag )

      r_vir = self.ahf_reader.get_mtree_halo_quantity( 'Rvir', snum, self.ahf_reader.index, self.ahf_reader.tag )
      c_analytic = self.ahf_reader.get_mtree_halo_quantity( 'cAnalytic', snum, self.ahf_reader.index,
                                                            self.ahf_reader.tag )
    r_scale = r_vir/c_analytic
    radii = radius_fraction*r_scale

    if hubble_param is not default:
      redshift = self.ahf_reader.mtree_halos[0]['redshift'][snum]
      x_pos /= ( 1. + redshift )*hubble_param
      y_pos /= ( 1. + redshift )*hubble_param
      radii /= ( 1. + redshift )*hubble_param

    if center:
      x_center = self.ahf_reader.mtree_halos[0]['Xc'][snum]
      y_center = self.ahf_reader.mtree_halos[0]['Yc'][snum]
      
      if hubble_param is not default:
        x_center /= ( 1. + redshift )*hubble_param
        y_center /= ( 1. + redshift )*hubble_param

      x_pos -= x_center
      y_pos -= y_center

    # To set the window size and such automatically, this is easiest
    ax.scatter( x_pos, y_pos, color=color, s=0 )

    for i, radius in enumerate( radii ):

      cir = mpatches.Circle( (x_pos[i], y_pos[i]), radius=radius, linewidth=2, \
                            color=color, linestyle=linestyle, fill=False, facecolor='w' )
      ax.add_patch( cir )

      if outline:
        cir.set_path_effects([ path_effects.Stroke(linewidth=5, foreground='black'),
                               path_effects.Normal() ])

  ########################################################################

  def plot_halo_time( self,
    data_key,
    halo_id = 0,
    plot_change_in_halo_id = False,
    ):

    fig = plt.figure( figsize=(10, 6), facecolor='white' )
    ax = plt.gca()

    plotted_mtree_halo = self.ahf_reader.mtree_halos[halo_id]

    # Make a blended transformation
    trans = transforms.blended_transform_factory( ax.transData, ax.transAxes )

    # Plot vertical lines when there's a change
    if plot_change_in_halo_id:
      for i, change_in_halo_id in enumerate( change_in_halo_ids ):
        if change_in_halo_id != 0:
            ax.plot( [plotted_mtree_halo.index[i], plotted_mtree_halo.index[i] ], [0., 1.],
                        transform=trans, color='k', linewidth=1, linestyle='--')

    # Plot position data
    ax.plot( plotted_mtree_halo.index, plotted_mtree_halo[data_key], color='#e41a1c', linewidth=3 )

    ax.set_xlabel( 'Snapshot Number', fontsize=22, )
    ax.set_ylabel( r'$X_{\rm{COM}}$', fontsize=22, )

















