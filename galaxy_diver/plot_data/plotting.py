#!/usr/bin/env python
'''General purpose plotting tools.

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import copy
import errno
import numpy as np
import os

import matplotlib
import matplotlib.pyplot as plt

import galaxy_diver.utils.dataio as dataio

########################################################################

def add_colorbar( fig_or_ax, color_object, method='fig', ax_location=[0.9, 0.1, 0.03, 0.8], **kw_args ):
  '''Add a colorbar to a figure(fig) according to an object on the figure(color_object). Added to the right side of the figure.
  Or add a colorbar to the right of an axis.

  Args:
    fig_or_ax (figure or axis object): The object to add the colorbar to.
    color_object (object): The object providing the colorscheme, i.e. what you want the colorbar to provide a map for.
    method (str): If 'fig', add to the right of the figure. If 'ax', add to the right of the axis.
    ax_location (list): If adding to a figure, this is where and how large the axis will be.
    **kw_args will be passed to colorbar()
  '''

  if method == 'fig':

    cax = fig_or_ax.add_axes(ax_location)
    cbar = fig_or_ax.colorbar(color_object, cax=cax, **kw_args)

  if method == 'ax':

    # Create divider for existing axes instance
    divider = make_axes_locatable(fig_or_ax)

    # Append axes to the right of ax, with 5% width of ax
    cax = divider.append_axes("right", pad=0.05, size='5%')

    # Create colorbar in the appended axes
    cbar = plt.colorbar(color_object, cax=cax, **kw_args)

  return cbar

########################################################################

# Functions for switching between a single index and a pair of indexes in the same way that add_subplot maps single subplots.

def reading_indexes( index, num_cols ):
  '''Switch to multiple indexes for a plot.

  Args:
    index (int): An index that can be used to access a subplot in the same way add_subplot() accesses subplots.
    num_cols (int): Number of columns in the figure.

  Returns:
    horizontal_index (int): Horizontal index of the plot.
    vertical_index (int): Vertical index of the plot.
  '''

  horizontal_index = index % num_cols
  vertical_index = index / num_cols

  return (horizontal_index, vertical_index)

def single_index( horizontal_index, vertical_index, num_cols ):
  '''Switch to a single index for a plot.

  Args:
    horizontal_index (int): Horizontal index of the plot.
    vertical_index (int): Vertical index of the plot.
    num_cols (int): Number of columns in the figure.

  Returns:
    index (int): An index that can be used to access a subplot in the same way add_subplot() accesses subplots.
  '''

  return vertical_index*num_cols + horizontal_index

########################################################################


def calc_num_rows( num_plots, num_cols ):
  '''Find the number of plots given the number of plots and the number of columns.

  Args:
    num_plots (int): Number of subplots.
    num_cols (int): Number of columns in the figure.

  Returns:
    num_rows (int): Number of rows in the figure.
  '''

  h_index_end, v_index_end = readingIndexes(num_plots - 1, num_cols)
  num_rows = v_index_end + 1

  return num_rows
  
########################################################################

def remove_all_space( figure, num_rows, num_cols, num_plots ):
  '''Removes any space between subplots for the given figure instance.
  
  Args:
    figure (figure object): The figure to alter.
    num_rows (int): Number of rows in the figure.
    num_cols (int): Number of columns in the figure.
    num_plots (int): Number of subplots.
  
  Modifies:
    figure (figure object): Alters the spacing.
  '''

  # Remove any space between the plots.
  figure.subplots_adjust(hspace=0.0001, wspace=0.0001)  
  
  for k in range(0, num_plots):

    # Get the subplot to plot it on.
    (i, j) = readingIndexes(k, num_cols)
    ax = figure.add_subplot(num_rows, num_cols, k+1) 
        
    # Get rid of x tick marks for middle graphs
    if singleIndex(i, j + 1, num_cols) < num_plots:

      plt.setp(ax.get_xticklabels(), visible=False)

    # Get rid of y tick marks for middle graphs      
    if i > 0:

      plt.setp(ax.get_yticklabels(), visible=False)
      
    # Get rid of tick labels that overlap with each other
    if i != num_cols - 1 and j == num_rows - 1:

      plt.setp(ax.get_xticklabels()[-1], visible=False)      
    if j != 0:

      plt.setp(ax.get_yticklabels()[-1], visible=False)
      
    # Get rid of x-tick labels that overlap with the other plots.
    if j + 1 == num_rows:
      
      plt.setp(ax.get_xticklabels()[0], visible=False)

    # Avoid overlapping x- and y-tick marks of the same subplot.

########################################################################


def save_fig( out_dir, save_file, fig=None, **save_args ):
  '''Save a figure using a pretty frequent combination of details.

  Args:
    out_dir (str): Output directory.
    save_file (str): Save file name.

  Keyword Args:
    fig : Figure or axis.
  '''

  # Make sure the output directory exists
  try:
    os.makedirs(out_dir)
  except OSError as exc: # Python >2.5
    if exc.errno == errno.EEXIST and os.path.isdir(out_dir):
      pass
    else: raise

  default_save_args = {'dpi':300, 'bbox_inches':'tight'}
  used_save_args = dataio.mergeDict(save_args, default_save_args)

  # Save the figure.
  if fig == None:
    plt.savefig('%s/%s'%(out_dir, save_file), **used_save_args)
  else:
    fig.savefig('%s/%s'%(out_dir, save_file), **used_save_args)

  print 'File saved at %s/%s'%(out_dir, save_file)

########################################################################


def fill_between_steps( ax, x, y1, y2=0, step_where='pre', **kwargs ):
    '''Fill between for a step plot.

    Args:
      ax (Axes) The axes to draw to
      x (array-like): Array/vector of index values.
      y1 (array-like or float): Array/vector of values to be filled under.
      y2 (array-Like or float, optional): Array/vector or bottom values for filled area. Default is 0.
      step_where ({'pre', 'post', 'mid'}): where the step happens, same meanings as for `step`
      **kwargs will be passed to the matplotlib fill_between() function.

    Returns
      ret (PolyCollection): The added artist
    '''
    if step_where not in {'pre', 'post', 'mid'}:
        raise ValueError("where must be one of {{'pre', 'post', 'mid'}} "
                         "You passed in {wh}".format(wh=step_where))

    # make sure y values are up-converted to arrays 
    if np.isscalar(y1):
        y1 = np.ones_like(x) * y1

    if np.isscalar(y2):
        y2 = np.ones_like(x) * y2

    # temporary array for up-converting the values to step corners
    # 3 x 2N - 1 array 

    vertices = np.vstack((x, y1, y2))

    # this logic is lifted from lines.py
    # this should probably be centralized someplace
    if step_where == 'pre':
        steps = np.zeros((3, 2 * len(x) - 1), np.float)
        steps[0, 0::2], steps[0, 1::2] = vertices[0, :], vertices[0, :-1]
        steps[1:, 0::2], steps[1:, 1:-1:2] = vertices[1:, :], vertices[1:, 1:]

    elif step_where == 'post':
        steps = np.zeros((3, 2 * len(x) - 1), np.float)
        steps[0, ::2], steps[0, 1:-1:2] = vertices[0, :], vertices[0, 1:]
        steps[1:, 0::2], steps[1:, 1::2] = vertices[1:, :], vertices[1:, :-1]

    elif step_where == 'mid':
        steps = np.zeros((3, 2 * len(x)), np.float)
        steps[0, 1:-1:2] = 0.5 * (vertices[0, :-1] + vertices[0, 1:])
        steps[0, 2::2] = 0.5 * (vertices[0, :-1] + vertices[0, 1:])
        steps[0, 0] = vertices[0, 0]
        steps[0, -1] = vertices[0, -1]
        steps[1:, 0::2], steps[1:, 1::2] = vertices[1:, :], vertices[1:, :]
    else:
        raise RuntimeError("should never hit end of if-elif block for validated input")

    # un-pack
    xx, yy1, yy2 = steps

    # now to the plotting part:
    return ax.fill_between(xx, yy1, y2=yy2, **kwargs)
  
