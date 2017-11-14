#!/usr/bin/env python
'''Tools for plotting AHF data.

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import copy
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
import galaxy_diver.plot_data.generic_plotter as generic_plotter

########################################################################

# For catching default values
default = object()

########################################################################
########################################################################

class AHFPlotter( generic_plotter.GenericPlotter ):

  ########################################################################

  def plot_halos_snapshot( self,
    snum,
    ax = default,
    type_of_halo_id = 'ahf_halos',
    color = 'w',
    linestyle = 'solid',
    linewidth = 2,
    outline = True,
    center = True,
    hubble_param = default,
    radius_fraction = 1.,
    length_scale = 'Rstar0.5',
    n_halos_plotted = 100,
    show_valid_halos = True,
    minimum_criteria = 'n_star',
    minimum_value = 10,
    ):
    '''Plot the halos as circles at their respective locations.
    
    Args:
      snum (int) :
        Snapshot to plot at.
        
      type_of_halo_id (str):
        Should the merger tree halos be plotted, or the first 50 ahf halos at that redshift?
        
      ax (axis object) :
        Axis to use. If default, one is created.
        
      color (str) :
        What color should the circle be?
        
      linestyle (str) :
        What linestyle to use.
        
      linewidth (int) :
        What linewidth to use.
        
      outline (bool) :
        Should the circles be outlined for increased visibility?
        
      center (bool) :
        Should the plot be centered at the most massive halo at z=0?
        
      hubble_param (float) :
        If given, the positions will be converted to physical kpc.
        
      radius_fraction (float) :
        The circles will be radius_fraction*r_scale
        
      length_scale (str) :
        What length scale to use?
        
      n_halos_plotted (int) :
        Number of AHF halos to show.
        
      show_valid_halos, minimum_criteria, minimum_value (bool,str,int) : 
        If valid_halos is True, only show halos with minimum criteria > minimum_value
    '''

    if ax is default:
      fig = plt.figure( figsize=(7,6), facecolor='white' )
      ax = plt.gca()

    if type_of_halo_id == 'ahf_halos':

      self.data_object.get_halos( snum )
      self.data_object.get_halos_add( snum )

      if show_valid_halos:
        min_criteria = self.data_object.ahf_halos[ minimum_criteria ]
        has_minimum_value = min_criteria >= minimum_value
        sl = has_minimum_value
      else:
        sl = slice( 0, n_halos_plotted )

      x_pos = self.data_object.ahf_halos['Xc'][sl]
      y_pos = self.data_object.ahf_halos['Yc'][sl]

      if length_scale == 'r_scale':
        r_vir = self.data_object.ahf_halos['Rvir'][sl]
        c_analytic = self.data_object.ahf_halos['cAnalytic'][sl]
        used_length_scale = r_vir/c_analytic
      else:
        used_length_scale = self.data_object.ahf_halos[length_scale][sl]

    elif type_of_halo_id == 'merger_tree':
      x_pos = self.data_object.get_mtree_halo_quantity( 'Xc', snum, self.data_object.index, self.data_object.tag )
      y_pos = self.data_object.get_mtree_halo_quantity( 'Yc', snum, self.data_object.index, self.data_object.tag )

      if length_scale == 'r_scale':
        r_vir = self.data_object.get_mtree_halo_quantity( 'Rvir', snum, self.data_object.index, self.data_object.tag )
        c_analytic = self.data_object.get_mtree_halo_quantity( 'cAnalytic', snum, self.data_object.index,
                                                            self.data_object.tag )
        used_length_scale = r_vir/c_analytic
      else:
        used_length_scale = self.data_object.get_mtree_halo_quantity( length_scale, snum, self.data_object.index,
                                                                      self.data_object.tag )

    radii = radius_fraction*used_length_scale

    if hubble_param is not default:
      redshift = self.data_object.mtree_halos[0]['redshift'][snum]
      x_pos /= ( 1. + redshift )*hubble_param
      y_pos /= ( 1. + redshift )*hubble_param
      radii /= ( 1. + redshift )*hubble_param

    if center:
      x_center = self.data_object.mtree_halos[0]['Xc'][snum]
      y_center = self.data_object.mtree_halos[0]['Yc'][snum]
      
      if hubble_param is not default:
        x_center /= ( 1. + redshift )*hubble_param
        y_center /= ( 1. + redshift )*hubble_param

      x_pos -= x_center
      y_pos -= y_center

    # To set the window size and such automatically, this is easiest
    ax.scatter( x_pos, y_pos, color=color, s=0 )

    for i, radius in enumerate( radii ):

      cir = mpatches.Circle( (x_pos.values[i], y_pos.values[i]), radius=radius, linewidth=linewidth, \
                            color=color, linestyle=linestyle, fill=False, facecolor='w' )
      ax.add_patch( cir )

      if outline:
        cir.set_path_effects([ path_effects.Stroke(linewidth=5, foreground='black'),
                               path_effects.Normal() ])

  ########################################################################

  def plot_halo_time( self,
    y_key,
    y_data = default,
    y_scale = default,
    hubble_param = default,
    conversion_factor = default,
    convert_to_physical = False,
    convert_to_comoving = False,
    y_data_div_key = default,
    ax = default,
    halo_id = 0,
    color = 'k',
    linestyle = '-',
    y_label = default,
    label = default,
    plot_change_in_halo_id = False,
    ):

    if ax is default:
      fig = plt.figure( figsize=(10, 6), facecolor='white' )
      ax = plt.gca()

    plotted_mtree_halo = self.data_object.data_object.mtree_halos[halo_id]

    x_data = np.log10( 1. + plotted_mtree_halo['redshift'] )

    if y_data is default:
      if y_key == 'r_scale':
        y_data = plotted_mtree_halo['Rvir']/plotted_mtree_halo['cAnalytic']
      else:
        y_data = plotted_mtree_halo[y_key]

    # Make a copy of the y-data so we don't alter it.
    y_data = copy.copy( y_data )

    if conversion_factor is not default:
      y_data *= conversion_factor

    if hubble_param is not default:
      y_data /= hubble_param

    if convert_to_physical:
      y_data /= ( 1. + plotted_mtree_halo['redshift'] )

    if convert_to_comoving:
      y_data *= ( 1. + plotted_mtree_halo['redshift'] )

    if y_data_div_key is not default:
      y_data /= plotted_mtree_halo[y_data_div_key]

    # Plot vertical lines when there's a change
    if plot_change_in_halo_id:
      # Make a blended transformation
      trans = transforms.blended_transform_factory( ax.transData, ax.transAxes )

      for i, change_in_halo_id in enumerate( change_in_halo_ids ):
        if change_in_halo_id != 0:
            ax.plot( [plotted_mtree_halo.index[i], plotted_mtree_halo.index[i] ], [0., 1.],
                        transform=trans, color='k', linewidth=1, linestyle='--')

    if label is default:
      label = self.label

    ax.plot( x_data, y_data, color=color, linewidth=3, label=label, linestyle=linestyle )

    tick_redshifts = np.array( [ 0.25, 0.5, 1, 2, 3, 4, 5, 6, 7, ] )
    x_tick_values = np.log10( 1. + tick_redshifts )
    plt.xticks( x_tick_values, tick_redshifts )

    if y_label is default:
      y_label = y_key

    if y_scale is not default:
      ax.set_yscale( y_scale )

    ax.set_xlabel( r'z', fontsize=22, )
    ax.set_ylabel( y_label, fontsize=22, )

















