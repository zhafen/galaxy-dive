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
import subprocess

import matplotlib.pyplot as plt
import matplotlib.colors as plt_colors
import matplotlib.patches
import matplotlib.ticker 
from mpl_toolkits.axes_grid1 import make_axes_locatable

import galaxy_dive.utils.data_operations as data_ops
import galaxy_dive.utils.astro as astro_utils

########################################################################
########################################################################

def box_plot(
    x_datas,
    y_datas,
    ax = None,
    color = 'k',
    y_mean_statistic = np.mean,
    x_scale = 'log',
    y_scale = 'log',
    y_floor = None,
    box_upper_limit = None,
    box_lower_limit = None,
    box_zorder = 50,
    blank_zorder = 30,
    line_zorder = 10,
    zorder_offset = 0,
    plot_boxes = True,
    skip_single_points = True,
    linewidth = 4,
    linestyle = '-',
    line_x_min = None,
    line_x_max = None,
):

    if ax is None:
        fig = plt.figure( figsize=(11,6) )
        ax = plt.gca()

    means = {
        'x' : [],
        'y' : [],
    }
    errs = {
        'x' : [],
        'y' : [],
    }
    for x_data, y_data in zip( x_datas, y_datas ):

        if ( len( x_data ) < 2 ) and skip_single_points:
            continue

        # Make copies so we don't modify the data
        x_data = copy.copy( x_data )
        y_data = copy.copy( y_data )

        # Apply a floor
        y_data[y_data<y_floor] = y_floor

        # Put in logspace
        if x_scale == 'log':
            x_data = np.log10( x_data )
        if y_scale == 'log':
            y_data = np.log10( y_data )

        x_mean = x_data.mean()
        y_mean = y_mean_statistic( y_data )

        x_err = x_data.std()
        y_err = y_data.std()

        # Get the negative and positive errs
        if x_scale == 'log':
            x_err_neg, x_err_pos = errs_from_log_errs(
                x_mean,
                x_err,
            )
        elif x_scale == 'linear':
            x_err_neg = x_err
            x_err_pos = x_err
        if y_scale == 'log':
            y_err_neg, y_err_pos = errs_from_log_errs(
                y_mean,
                y_err,
                upper_limit = box_upper_limit,
                lower_limit = box_lower_limit,
            )
        elif y_scale == 'linear':
            y_err_neg = y_err
            y_err_pos = y_err

        if x_scale == 'log':
            x_mean = 10.**x_mean
        if y_scale == 'log':
            y_mean = 10.**y_mean

        # Store for plotting
        means['x'].append( x_mean )
        means['y'].append( y_mean )
        errs['x'].append( x_mean )
        errs['y'].append( x_mean )

        left = x_mean - x_err_neg
        bottom = y_mean - y_err_neg

        width = x_err_neg + x_err_pos
        height = y_err_neg + y_err_pos

        if plot_boxes:
            # Rectangle itself
            rect = matplotlib.patches.Rectangle(
                (left,bottom),
                width,
                height,
                fill = False,
                edgecolor = color,
                linewidth = linewidth,
                linestyle = linestyle,
                zorder = box_zorder - zorder_offset,
            )                
            ax.add_patch( rect )
            
            # Blank rectangles to clean things up
            rect = matplotlib.patches.Rectangle(
                (left,bottom),
                width,
                height,
                facecolor = 'w',
                edgecolor = 'w',
                linewidth = linewidth,
                zorder = blank_zorder - zorder_offset,
            )                
            ax.add_patch( rect )

    sorted_inds = np.argsort( means['x'] )
    sorted_xs = np.array( means['x'] )[sorted_inds]
    sorted_ys = np.array( means['y'] )[sorted_inds]

    # Extend the lines if requested
    if ( line_x_min is not None ) or ( line_x_max is not None ):
        if x_scale == 'log':
            line_xs = list( np.log10( sorted_xs ) )
            if line_x_min is not None:
                line_x_min = np.log10( line_x_min )
            if line_x_max is not None:
                line_x_max = np.log10( line_x_max )
        else:
            line_xs = list( sorted_xs )
        if y_scale == 'log':
            line_ys = list( np.log10( sorted_ys ) )
        else:
            line_ys = list( sorted_ys )
    if line_x_min is not None:
        slope_start = (
            ( line_ys[1] - line_ys[0] ) /
            ( line_xs[1] - line_xs[0] )
        )
        line_y_min = line_ys[0] - slope_start * ( line_xs[0] - line_x_min )
            
        line_xs.insert( 0, line_x_min )
        line_ys.insert( 0, line_y_min )
    if line_x_max is not None:
        slope_end = (
            ( line_ys[-1] - line_ys[-2] ) /
            ( line_xs[-1] - line_xs[-2] )
        )
        line_y_max = line_ys[-1] + slope_end * ( line_x_max - line_xs[-1] )
            
        line_xs.append( line_x_max )
        line_ys.append( line_y_max )
    if ( line_x_min is not None ) or ( line_x_max is not None ):
        if x_scale == 'log':
            line_xs = 10.**( np.array( line_xs ) )
        if y_scale == 'log':
            line_ys = 10.**( np.array( line_ys ) )
        sorted_xs = line_xs
        sorted_ys = line_ys

    # Plot everything else
    ax.plot(
        sorted_xs,
        sorted_ys,
        linewidth = linewidth,
        linestyle = linestyle,
        color = color,
        zorder = line_zorder - zorder_offset,
    )

    # Set scale
    ax.set_xscale( x_scale )
    ax.set_yscale( y_scale )

########################################################################

def errs_from_log_errs( log_mean, log_err, upper_limit=None, lower_limit=None ):
    
    err_positive = ( 10.**log_mean ) * \
        ( 10.**log_err - 1. )
    err_negative = -1.*( 10.**log_mean ) * \
        ( 10.**-log_err - 1. )
        
    if upper_limit is not None:
        exceeds_limit = 10.**log_mean + err_positive > upper_limit
        try:
            err_positive[exceeds_limit] = ( upper_limit - 10.**log_mean )[exceeds_limit]
        except TypeError:
            err_positive = ( upper_limit - 10.**log_mean )

    if lower_limit is not None:
        exceeds_limit = 10.**log_mean - err_negative < lower_limit
        try:
            err_negative[exceeds_limit] = ( 10.**log_mean - lower_limit )[exceeds_limit]
        except TypeError:
            err_negative = ( 10.**log_mean - lower_limit )
        
    return err_negative, err_positive

########################################################################

def add_redshift_to_axis(
        ax,
        hubble,
        omega_matter,
        tick_redshifts = np.array(
            [ 0.1, 0.25, 0.4, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5, ]
        )
    ):
    '''Add redshift labels to the top axis of plot that has age of the universe
    on the bottom axis.

    Args:
        ax (axis object): Original axis

        hubble (float): Hubble parameter, used for calculating the age.

        omega_matter (float): Cosmological param used for calcing the age.

        tick_redshifts (array-like): Redshifts to plot ticks at.
    
    '''

    tick_times = astro_utils.age_of_universe(
        tick_redshifts,
        h = hubble,
        omega_matter = omega_matter,
    )

    # Make sure we aren't trying to plot ticks that would go out of bounds,
    # because that breaks things
    ax2_ticks = []
    ax2_tick_labels = []
    x_range = ax.get_xlim()
    for ax2_tick, ax2_tick_label in zip( tick_times, tick_redshifts ):
        if ( ax2_tick > x_range[0] ) and ( ax2_tick < x_range[1] ):
            ax2_ticks.append( ax2_tick )
            ax2_tick_labels.append( ax2_tick_label )

    # Add a second axis for plotting
    ax2 = ax.twiny()
    ax2.set_xlim( x_range )
    ax2.set_xticks( ax2_ticks )
    ax2.set_xticklabels( ax2_tick_labels )

    return ax2
        
########################################################################

def make_movie( img_dir, file_pattern, movie_save_file, quality=1 ):
    '''Make a movie using ffmpeg.

    Args:
        img_dir (str) : Directory containing images to stitch together.
        file_pattern (str) : Pattern for the images.
        movie_save_file (str) : What to save the movie as. Typically .mp4 or .avi
        quality (int) : What quality to use for making the movie (1-31). Default is 1, which is highest quality.
    '''

    prev_dir = os.getcwd()
    os.chdir( img_dir )

    subprocess.call( [ "ffmpeg", "-pattern_type", "glob", "-i", file_pattern, "-q:v", str( quality ), movie_save_file, ])

    os.chdir( prev_dir )

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


def custom_sequential_colormap(
    color1,
    color2 = (1, 1, 1),
    cmap_name = 'custom',
    n_bins = 256,
):
    '''Create a custom sequential colormap using one to two colors

    Args:
        color1 (color identifier, e.g. hex or RGB tuple) :
            One of the two colors used. The colormap transtions from color1
            to color2 on a gradient.

        color2 (color identifier, e.g. hex or RGB tuple) :
            One of the two colors used. The colormap transtions from color1
            to color2 on a gradient.

        cmap_name (str) :
            What to call the colormap.

        n_bins (int) :
            Number of bins to divide the colormap into.

    Returns:
        cmap (colormap instance) :
            Resulting colormap.
    '''

    colors = [ color1, color2 ]

    cmap = plt_colors.LinearSegmentedColormap.from_list(
        cmap_name,
        colors,
        N = n_bins
    )

    return cmap


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

    return vertical_index * num_cols + horizontal_index

########################################################################


def calc_num_rows( num_plots, num_cols ):
    '''Find the number of plots given the number of plots and the number of columns.

    Args:
        num_plots (int): Number of subplots.
        num_cols (int): Number of columns in the figure.

    Returns:
        num_rows (int): Number of rows in the figure.
    '''

    h_index_end, v_index_end = reading_indexes(num_plots - 1, num_cols)
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
        (i, j) = reading_indexes(k, num_cols)
        ax = figure.add_subplot(num_rows, num_cols, k + 1)

        # Get rid of x tick marks for middle graphs
        if single_index(i, j + 1, num_cols) < num_plots:

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


def save_fig( out_dir, save_file, fig=None, resolution='auto', **save_args ):
    '''Save a figure using a pretty frequent combination of details.

    Args:
        out_dir (str): Output directory.
        save_file (str): Save file name.

    Keyword Args:
        fig : Figure or axis.
        resolution : What resolution to save the figure as. Defaults to 3600 pixels on the shortest side.
    '''

    # Make sure the output directory exists
    try:
        os.makedirs(out_dir)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(out_dir):
            pass
        else:
            raise
    
    if resolution == 'auto':
        figsize = np.min( fig.get_size_inches() )
        dpi = int( 3600./figsize )
    else:
        dpi = resolution

    default_save_args = {'dpi': dpi, 'bbox_inches': 'tight'}
    used_save_args = data_ops.merge_dict( save_args, default_save_args)

    # Save the figure.
    if fig is None:
        plt.savefig('%s/%s' % (out_dir, save_file), **used_save_args)
    else:
        fig.savefig('%s/%s' % (out_dir, save_file), **used_save_args)

    print('File saved at %s/%s' % (out_dir, save_file))

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
                raise ValueError(
                    "where must be one of {{'pre', 'post', 'mid'}} "
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

########################################################################

def custom_log_formatter( x, y ): 
    '''Better tick labels for log axes, as coded up by Alex Gurvich, who was
    deeply inspired by Jonathan Stern.

    To use, do, for example,
    >>> my_log_ticker = matplotlib.ticker.FuncFormatter(custom_log_formatter)
    >>> ax.yaxis.set_major_formatter(my_log_ticker)

    Args:
        x (list-like): List of tick values.

        y (mystery):
            Without including a second argument this formatter breaks. I
            haven't looked into why.

    Returns:
        Tick Formatter
    '''

    if x in [1e-2,1e-1,1,10,100]: 
        return r"$%g$"%x 
    elif ( x <= 1. ) & ( x>= 0.1 ):
        return r"$%g$"%x 
    else: 
        return matplotlib.ticker.LogFormatterMathtext()(x) 
