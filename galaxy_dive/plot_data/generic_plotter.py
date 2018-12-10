#!/usr/bin/env python
'''Class for plotting simulation data.

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

# Base python imports
import numpy as np
import os
import scipy.stats
import scipy.signal as signal
import warnings
import verdict

import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as plt_colors
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as path_effects
import matplotlib.transforms as transforms

import galaxy_dive.utils.mp_utils as mp_utils
import galaxy_dive.utils.utilities as utilities

import galaxy_dive.plot_data.plotting as gen_plot
import galaxy_dive.plot_data.pu_colormaps as pu_cm

########################################################################
########################################################################

class GenericPlotter( object ):

    @utilities.store_parameters
    def __init__( self, data_object, label=None, color='black', ):
        '''
        Args:
            data_object ( generic_data.GenericData object or subclass of such ) : The data container to use.
        '''

        pass

    ########################################################################
    # Alternate inherent methods
    ########################################################################

    def __getattr__( self, attr):
        '''By replacing getattr with the following code, we allow automatically searching the data_object
        for the appropriate attribute as well, while losing none of the original functionality.
        '''

        print( "Attribute {} not found in plotting object. Checking data object.".format( attr ) )

        return getattr( self.data_object, attr )

    ########################################################################
    # Specific Generic Plots
    ########################################################################

    def histogram( self,
        data_key,
        provided_data = None,
        provided_hist = None,
        weight_key = None,
        slices = None,
        ax = None,
        fix_invalid = False,
        mask_zeros = False,
        invalid_fix_method = None,
        bins = 32,
        normed = True,
        norm_type = 'probability',
        scaling = None,
        smooth = False,
        smoothing_window_length = 9,
        smoothing_polyorder = 3,
        histogram_style = 'step',
        color = 'black',
        linestyle = '-',
        linewidth = 3.5,
        alpha = 1.,
        x_range = None, y_range = None,
        x_label = None, y_label = None,
        add_x_label = True, add_y_label = True,
        add_plot_label = True,
        plot_label = None,
        line_label = None,
        label_fontsize = 24,
        x_scale = 'linear', y_scale = 'linear',
        cdf = False,
        vertical_line = None,
        vertical_line_kwargs = { 'linestyle': '--', 'linewidth': 3, 'color': 'k', },
        return_dist = False,
        assert_contains_all_data = True,
        data_kwargs = {},
        *args, **kwargs ):
        '''Make a histogram of the data. Extra arguments are passed to
        self.data_object.get_selected_data.

        Args:
            data_key (str) :
                Data key to plot.

            weight_key (str) :
                Data key for data to use as a weight. By None, no weight.

            slices (int or tuple of slices) :
                How to slices the data.

            ax (axis) :
                What axis to use. By default creates a figure and places the axis on it.

            fix_invalid (bool) :
                Throw away invalid values?

            invalid_fix_method (float or int) :
                How to handle invalid values. By None throw them away. Providing a value to this argument instead replaces
                them with that value.

            bins (int or array-like) :
                bins argument to be passed to np.histogram

            normed (bool) :
                Normalize the histogram?

            color (str) :
                Color of histogram.

            linestyle (str) :
                Linestyle of histogram.

            linewidth (float) :
                Linewidth for histogram.

            alpha (float) :
                Alpha value for histogram

            x_range, y_range ( [min,max] ) :
                What are the minimum and maximum x- and y- values to include?
                Defaults to matplotlib's automatic choices

            x_label, ylabel (str) :
                Axes labels. Defaults to the data_key for the x-axis and "Normalized Histogram" for the y-axis.

            add_x_label, add_y_label (bool) :
                Include axes labels?

            plot_label (str or dict) :
                What to label the plot with. By None, uses self.label.

            line_label (str) :
                What label to give the line.

            label_fontsize (int) :
                Fontsize for the labels.

            x_scale, y_scale (str) :
                What scales to use for the x and y axes.

            cdf (bool) :
                Plot a CDF instead.

            vertical_line (float) :
                Plot a vertical line at this value on the x-axis, if true.

            vertical_line_kwargs (dict) :
                Arguments to pass to ax.plot( for the vertical line.

            return_dist (bool) :
                If True, return the data values and the edges for the histogram.

            assert_contains_all_data (bool) :
                If True, make sure that the histogram plots all selected data.

            *args, **kwargs :
                Extra arguments to pass to self.data_object.get_selected_data()
        '''

        print( "Plotting histogram for {}".format( data_key ) )

        if provided_hist is None:

            if isinstance( slices, int ):
                sl = ( slice(None), slices )
            else:
                sl = slices

            data_kwargs = utilities.merge_two_dicts( data_kwargs, kwargs )

            if provided_data is None:
                data = self.data_object.get_selected_data(
                    data_key,
                    sl=sl,
                    *args, **data_kwargs
                ).copy()
            else:
                data = provided_data.copy()

            if weight_key is None:
                weights = None
            else:

                if 'scale_key' in kwargs:
                    warnings.warn(
                        "Scaling weights by {}. Is this correct?".format(
                            kwargs['scale_key']
                        )
                    )
                weights = self.data_object.get_selected_data( weight_key, sl=sl, *args, **kwargs )

            if fix_invalid:
                if invalid_fix_method is None:
                    data = np.ma.fix_invalid( data ).compressed()
                else:
                    data = np.ma.fix_invalid( data )
                    data.fill_value = invalid_fix_method
                    data = data.filled()

            # Make the histogram itself
            hist, edges = np.histogram( data, bins=bins, weights=weights )

            # Make sure we have all the data in the histogram
            if assert_contains_all_data:
                assert data.size == hist.sum()

            if normed:

                if norm_type == 'probability':
                    hist = hist.astype( float ) / ( hist.sum()*(edges[1] - edges[0]) )
                elif norm_type == 'bin_width':
                    hist = hist.astype( float ) / (edges[1] - edges[0])
                elif norm_type == 'outer_edge':
                    hist = hist.astype( float ) / hist[-1]
                elif norm_type == 'max_value':
                    hist = hist.astype( float ) / hist.max()
                else:
                    raise Exception(
                        "Unrecognized norm_type, {}".format( norm_type )
                    )

            if scaling is not None:
                hist *= scaling

            if cdf:
                hist = np.cumsum( hist )*(edges[1] - edges[0])

        else:
            hist = provided_hist
            edges = bins

        if mask_zeros:
            hist = np.ma.masked_where(
                hist < 1e-14,
                hist,
            )

        if smooth:
            hist = signal.savgol_filter(
                hist,
                window_length = smoothing_window_length,
                polyorder = smoothing_polyorder,
            )


        if ax is None:
            fig = plt.figure( figsize=(11,5), facecolor='white', )
            ax = plt.gca()
        if line_label is None:
            line_label = self.label

        if color is None:
            color = self.color

        # Inserting a 0 at the beginning allows plotting a numpy histogram with a step plot
        if histogram_style == 'step':
            ax.step(
                edges,
                np.insert(hist, 0, 0.),
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                label=line_label,
                alpha=alpha,
            )
        elif histogram_style == 'line':
            x_values = 0.5 * ( edges[:-1] + edges[1:] )
            ax.plot(
                x_values,
                hist,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                label=line_label,
                alpha=alpha,
            )
        else:
            raise KeyError(
                "Unrecognized histogram_style, {}".format( histogram_style )
            )

        # Plot a vertical line?
        if vertical_line is not None:
            trans = transforms.blended_transform_factory( ax.transData, ax.transAxes )
            ax.plot(
                [ vertical_line, ]*2,
                [ 0., 1., ],
                transform = trans,
                **vertical_line_kwargs
)

        # Plot label
        if add_plot_label:
            if plot_label is None:
                plt_label = ax.annotate(
                    s = self.label,
                    xy = (0.,1.0),
                    va = 'bottom',
                    xycoords = 'axes fraction',
                    fontsize = label_fontsize,
                 )
            elif isinstance( plot_label, str ):
                plt_label = ax.annotate(
                    s = plot_label,
                    xy = (0.,1.0),
                    va = 'bottom',
                    xycoords = 'axes fraction',
                    fontsize = label_fontsize,
                 )
            elif isinstance( plot_label, dict ):
                plt_label = ax.annotate( **plot_label )
            elif plot_label is None:
                pass
            else:
                raise Exception( 'Unrecognized plot_label arguments, {}'.format( plot_label ) )

        # Add axis labels
        if add_x_label:
            if x_label is None:
                x_label = data_key
            ax.set_xlabel( x_label, fontsize=label_fontsize )
        if add_y_label:
            if y_label is None:
                if not cdf:
                    y_label = r'Normalized Histogram'
                else:
                    y_label = r'CDF'
            ax.set_ylabel( y_label, fontsize=label_fontsize )

        if x_range is not None:
            ax.set_xlim( x_range )
        if y_range is not None:
            ax.set_ylim( y_range )

        ax.set_xscale( x_scale )
        ax.set_yscale( y_scale )

        if return_dist:
            return hist, edges

    ########################################################################

    def histogram2d( self,
        x_key, y_key,
        x_data = None, y_data = None,
        weight_key = None,
        x_data_args = {}, y_data_args = {},
        weight_data_args = {},
        slices = None,
        ax = None,
        x_range = None, y_range = None,
        x_scale = 'linear', y_scale = 'linear', z_scale = 'log',
        n_bins = 128,
        average = False,
        normed = False,
        hist_div_arr = None,
        conditional_y = False,
        y_div_function = None,
        vmin = None, vmax = None,
        min_bin_value_displayed = None,
        zorder = 0,
        add_colorbar = True,
        cmap = pu_cm.magma,
        colorbar_args = None,
        x_label = None, y_label = None,
        add_x_label = True, add_y_label = True,
        plot_label = None,
        outline_plot_label = False,
        label_galaxy_cut = False,
        label_redshift = False,
        label_fontsize = 24,
        tick_param_args = None,
        out_dir = None,
        save_file = None,
        close_plot_after_saving = True,
        fix_invalid = True,
        line_slope = None,
        cdf = False,
        horizontal_line = None, vertical_line = None,
        horizontal_line_kwargs = { 'linestyle': '--', 'linewidth': 5, 'color': '#337DB8', },
        vertical_line_kwargs = { 'linestyle': '--', 'linewidth': 5, 'color': '#337DB8', },
        return_dist = False,
        *args, **kwargs ):
        '''Make a 2D histogram of the data. Extra arguments are passed to get_selected_data.

        Args:
            x_key, y_key (str) : Data keys to plot.
            weight_key (str) : Data key for data to use as a weight. By None, no weight.
            x_data_args, y_data_args (dicts) : Keyword arguments to be passed only to x or y.
            slices (int or tuple of slices) : How to slices the data.
            ax (axis) : What axis to use. By None creates a figure and places the axis on it.
            x_range, y_range ( (float, float) ) : Histogram edges. If None, all data is enclosed. If list, set manually.
                If float, is +- x_range*length scale at that snapshot.
            n_bins (int) : Number of bins in the histogram.
            vmin, vmax (float) : Limits for the colorbar.
            aspect (str) : What should the aspect ratio of the plot be?
            plot_halos (bool) : Whether or not to plot merger tree halos on top of the histogram.
                Only makes sense for when dealing with positions.
            add_colorbar (bool) : If True, add a colorbar to colorbar_args
            colorbar_args (axis) : What axis to add the colorbar to. By None, is ax.
            x_label, ylabel (str) : Axes labels.
            add_x_label, add_y_label (bool) : Include axes labels?
            plot_label (str or dict) : What to label the plot with. By None, uses self.label.
                Can also pass a dict of full args.
            outline_plot_label (bool) : If True, add an outline around the plot label.
            label_galaxy_cut (bool) : If true, add a label that indicates how the galaxy was defined.
            label_redshift (bool) : If True, add a label indicating the redshift.
            label_fontsize (int) : Fontsize for the labels.
            tick_param_args (args) : Arguments to pass to ax.tick_params. By None, don't change inherent defaults.
            out_dir (str) : If given, where to save the file.
            fix_invalid (bool) : Fix invalid values.
            line_slope (float) : If given, draw a line with the given slope.
        '''

        if isinstance( slices, int ):
            sl = ( slice(None), slices )
        else:
            sl = slices

        varying_kwargs = {
            'x': x_data_args,
            'y': y_data_args,
            'weight': weight_data_args
        }
        data_kwargs = utilities.dict_from_defaults_and_variations( kwargs, varying_kwargs )

        # Get data
        if x_data is None:
            x_data = self.data_object.get_selected_data( x_key, sl=sl, *args, **data_kwargs['x'] ).copy()
        if y_data is None:
            y_data = self.data_object.get_selected_data( y_key, sl=sl, *args, **data_kwargs['y'] ).copy()

        if y_div_function is not None:
            y_div_values = y_div_function( x_data )
            y_data /= y_div_values

        # Fix NaNs
        if fix_invalid:
            x_mask = np.ma.fix_invalid( x_data ).mask
            y_mask = np.ma.fix_invalid( y_data ).mask
            mask = np.ma.mask_or( x_mask, y_mask )

            x_data = np.ma.masked_array( x_data, mask=mask ).compressed()
            y_data = np.ma.masked_array( y_data, mask=mask ).compressed()

        if weight_key is None:
            weights = None
        else:
            weights = self.data_object.get_selected_data(
                weight_key,
                sl=sl,
                *args,
                **data_kwargs['weight']
            ).flatten()

            if fix_invalid:
                weights = np.ma.masked_array( weights, mask=mask ).compressed()

        if x_range is None:
            x_range = [ x_data.min(), x_data.max() ]
        elif isinstance( x_range, float ):
            x_range = np.array( [ -x_range, x_range ])*self.data_object.length_scale[slices]
        if y_range is None:
            y_range = [ y_data.min(), y_data.max() ]
        elif isinstance( y_range, float ):
            y_range = np.array( [ -y_range, y_range ])*self.data_object.length_scale[slices]

        if x_scale == 'log':
            x_edges = np.logspace( np.log10( x_range[0] ), np.log10( x_range[1] ), n_bins )
        else:
            x_edges = np.linspace( x_range[0], x_range[1], n_bins )
        if y_scale == 'log':
            y_edges = np.logspace( np.log10( y_range[0] ), np.log10( y_range[1] ), n_bins )
        else:
            y_edges = np.linspace( y_range[0], y_range[1], n_bins )

        # Make the histogram
        hist2d, x_edges, y_edges = np.histogram2d( x_data, y_data, [x_edges, y_edges], weights=weights, normed=normed )

        # If doing an average, divide by the number in each bin
        if average:
            average_hist2d, x_edges, y_edges = np.histogram2d( x_data, y_data, [x_edges, y_edges], normed=normed )
            hist2d /= average_hist2d

        # If making the y-axis conditional, divide by the distribution of data for the x-axis.
        if conditional_y:
            hist_x, x_edges = np.histogram( x_data, x_edges, normed=normed )
            hist2d /= hist_x[:,np.newaxis]

        # Divide the histogram bins by this array
        if hist_div_arr is not None:
            hist2d /= hist_div_arr

        # Mask bins below a specified value
        if min_bin_value_displayed is not None:
            hist2d = np.ma.masked_where(
                hist2d < min_bin_value_displayed,
                hist2d,
            )

        # Plot
        if ax is None:
            fig = plt.figure( figsize=(10,9), facecolor='white' )
            ax = plt.gca()

        if z_scale == 'linear':
            norm = plt_colors.Normalize()
        elif z_scale == 'log':
            norm = plt_colors.LogNorm()

        if cdf:
            raise Exception(
                "Not implemented yet. When implementing, use utilities.cumsum2d"
            )

        im = ax.pcolormesh(
            x_edges,
            y_edges,
            hist2d.transpose(),
            cmap = cmap,
            norm = norm,
            vmin = vmin,
            vmax = vmax,
            zorder = zorder,
        )

        # Add a colorbar
        if add_colorbar:
            if colorbar_args is None:
                colorbar_args = ax
                cbar = gen_plot.add_colorbar( colorbar_args, im, method='ax' )
            else:
                colorbar_args['color_object'] = im
                cbar = gen_plot.add_colorbar( **colorbar_args )
            cbar.ax.tick_params( labelsize=20 )

        # Plot Line for easier visual interpretation
        if line_slope is not None:
            line_x = np.array( [ x_data.min(), x_data.max() ] )
            line_y = line_slope*line_x
            ax.plot( line_x, line_y, linewidth=3, linestyle='dashed', )

        if horizontal_line is not None:
            trans = transforms.blended_transform_factory( ax.transAxes, ax.transData )
            ax.plot( [ 0., 1. ], [ horizontal_line, ]*2, transform=trans, **horizontal_line_kwargs )
        if vertical_line is not None:
            trans = transforms.blended_transform_factory( ax.transData, ax.transAxes )
            ax.plot( [ vertical_line, ]*2, [ 0., 1. ], transform=trans, **vertical_line_kwargs )

        # Plot label
        if plot_label is not None:
            if plot_label is None:
                plt_label = ax.annotate(
                    s = self.label,
                    xy = (0.,1.0),
                    va = 'bottom',
                    xycoords = 'axes fraction',
                    fontsize = label_fontsize,
                 )
            elif isinstance( plot_label, str ):
                plt_label = ax.annotate(
                    s = plot_label,
                    xy = (0.,1.0),
                    va = 'bottom',
                    xycoords = 'axes fraction',
                    fontsize = label_fontsize,
                 )
            elif isinstance( plot_label, dict ):
                plt_label = ax.annotate( **plot_label )
            else:
                raise Exception( 'Unrecognized plot_label arguments, {}'.format( plot_label ) )
            if outline_plot_label:
                plt_label.set_path_effects([ path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal() ])

        # Upper right label (info label)
        info_label = ''
        if label_galaxy_cut:
            info_label = r'$r_{ \rm cut } = ' + '{:.3g}'.format( self.data_object.galids.parameters['galaxy_cut'] ) + 'r_{ s}$'
        if label_redshift:
            try:
                info_label = r'$z=' + '{:.3f}'.format( self.data_object.redshift ) + '$'+ info_label
            except ValueError:
                info_label = r'$z=' + '{:.3f}'.format( self.data_object.redshift.values[sl[1]] ) + '$'+ info_label
        if label_galaxy_cut or label_redshift:
            ax.annotate( s=info_label, xy=(1.,1.0225), xycoords='axes fraction', fontsize=label_fontsize,
                ha='right' )

        # Add axis labels
        if add_x_label:
            if x_label is None:
                x_label = x_key
            ax.set_xlabel( x_label, fontsize=label_fontsize )
        if add_y_label:
            if y_label is None:
                y_label = y_key
            ax.set_ylabel( y_label, fontsize=label_fontsize )

        # Limits
        ax.set_xlim( x_range )
        ax.set_ylim( y_range )

        # Scale
        ax.set_xscale( x_scale )
        ax.set_yscale( y_scale )

        # Set tick parameters
        if tick_param_args is not None:
            ax.tick_params( **tick_param_args )

        # Save the file
        if out_dir is not None:
            if save_file is None:
                save_file = '{}_{:03d}.png'.format( self.label, self.data_object.ptracks.snum[slices] )

            gen_plot.save_fig( out_dir, save_file, fig=fig, dpi=75 )

            if close_plot_after_saving:
                plt.close()

        # Return?
        if return_dist:
            return hist2d, x_edges, y_edges

    ########################################################################

    def statistic_and_interval(
        self,
        x_key, y_key,
        x_data = None, y_data = None,
        weights = None,
        statistic = 'median',
        lower_percentile = 16,
        upper_percentile = 84,
        plot_interval = True,
        x_data_args = {}, y_data_args = {},
        ax = None,
        slices = None,
        fix_invalid = False,
        bins = 64,
        linewidth = 3,
        linestyle = '-',
        color = 'k',
        label = None,
        zorder = 100,
        alpha = 0.5,
        plot_label = None,
        add_plot_label = True,
        plot_label_kwargs = {
            'xy': (0.,1.0),
            'va': 'bottom',
            'xycoords': 'axes fraction',
            'fontsize': 22,
        },
        return_values = False,
        *args, **kwargs
    ):

        if isinstance( slices, int ):
            sl = ( slice(None), slices )
        else:
            sl = slices

        varying_kwargs = {
            'x': x_data_args,
            'y': y_data_args,
        }
        data_kwargs = utilities.dict_from_defaults_and_variations( kwargs, varying_kwargs )

        # Get data
        if x_data is None:
            x_data = self.data_object.get_selected_data( x_key, sl=sl, *args, **data_kwargs['x'] ).copy()
        if y_data is None:
            y_data = self.data_object.get_selected_data( y_key, sl=sl, *args, **data_kwargs['y'] ).copy()

        # Fix NaNs
        if fix_invalid:
            x_mask = np.ma.fix_invalid( x_data ).mask
            y_mask = np.ma.fix_invalid( y_data ).mask
            mask = np.ma.mask_or( x_mask, y_mask )

            x_data = np.ma.masked_array( x_data, mask=mask ).compressed()
            y_data = np.ma.masked_array( y_data, mask=mask ).compressed()

        # Calculate the statistic
        if statistic == 'weighted_mean':

            assert weights is not None, "Need to provide weights."

            weighted_sum, bin_edges, binnumber = scipy.stats.binned_statistic(
                x = x_data,
                values = y_data * weights,
                statistic = 'sum',
                bins = bins,
            )
            weights_sum, bin_edges, binnumber = scipy.stats.binned_statistic(
                x = x_data,
                values = weights,
                statistic = 'sum',
                bins = bins,
            )

            stat = weighted_sum / weights_sum

        else:

            assert weights is None, "weights only works with weighted_mean"

            # Usual statistic
            stat, bin_edges, binnumber = scipy.stats.binned_statistic(
                x = x_data,
                values = y_data,
                statistic = statistic,
                bins = bins,
            )

        # Calculate the percentiles
        def get_lower_percentile( data ):
            return np.percentile( data, lower_percentile )
        def get_upper_percentile( data ):
            return np.percentile( data, upper_percentile )
        low_p, bin_edges, binnumber = scipy.stats.binned_statistic(
            x = x_data,
            values = y_data,
            statistic = get_lower_percentile,
            bins = bins,
        )
        high_p, bin_edges, binnumber = scipy.stats.binned_statistic(
            x = x_data,
            values = y_data,
            statistic = get_upper_percentile,
            bins = bins,
        )

        # Get plotting axis
        if ax is None:
            fig = plt.figure( figsize=(10,9), facecolor='white' )
            ax = plt.gca()

        # X Values fo rplot
        x_values = bin_edges[:-1] + 0.5 * ( bin_edges[1] - bin_edges[0] )

        # Plot statistic
        ax.plot(
            x_values,
            stat,
            linewidth = linewidth,
            linestyle = linestyle,
            color = color,
            zorder = zorder,
            label = label,
        )

        # Plot interval
        if plot_interval:
            ax.fill_between(
                x_values,
                low_p,
                high_p,
                color = color,
                alpha = alpha,
            )

        # Add plot label
        if add_plot_label:
            if plot_label is None:
                plot_label = self.label
            if plot_label is not None:
                plt_label = ax.annotate(
                    s = plot_label,
                    **plot_label_kwargs
                 )

        if return_values:
            return stat, low_p, high_p, bin_edges

    ########################################################################

    def scatter(
        self,
        x_key, y_key,
        slices = None,
        n_subsample = None,
        ax = None,
        marker_size = 100,
        color = 'k',
        marker = '.',
        zorder = -100,
        x_range = None, y_range = None,
        x_label = None, y_label = None,
        add_x_label = True, add_y_label = True,
        plot_label = None,
        outline_plot_label = False,
        label_galaxy_cut = False,
        label_redshift = False,
        label_fontsize = 24,
        tick_param_args = None,
        out_dir = None,
        fix_invalid = True,
        line_slope = None,
        *args, **kwargs ):
        '''Make a 2D scatter plot of the data. Extra arguments are passed to get_selected_data.
        Args:
            x_key, y_key (str) : Data keys to plot.
            weight_key (str) : Data key for data to use as a weight. By None, no weight.
            slices (int or tuple of slices) : How to slices the data.
            ax (axis) : What axis to use. By None creates a figure and places the axis on it.
            x_range, y_range ( (float, float) ) : Histogram edges. If None, all data is enclosed. If list, set manually.
                If float, is +- x_range*length scale at that snapshot.
            n_bins (int) : Number of bins in the histogram.
            vmin, vmax (float) : Limits for the colorbar.
            plot_halos (bool) : Whether or not to plot merger tree halos on top of the histogram.
                Only makes sense for when dealing with positions.
            add_colorbar (bool) : If True, add a colorbar to colorbar_args
            colorbar_args (axis) : What axis to add the colorbar to. By None, is ax.
            x_label, ylabel (str) : Axes labels.
            add_x_label, add_y_label (bool) : Include axes labels?
            plot_label (str or dict) : What to label the plot with. By None, uses self.label.
                Can also pass a dict of full args.
            outline_plot_label (bool) : If True, add an outline around the plot label.
            label_galaxy_cut (bool) : If true, add a label that indicates how the galaxy was defined.
            label_redshift (bool) : If True, add a label indicating the redshift.
            label_fontsize (int) : Fontsize for the labels.
            tick_param_args (args) : Arguments to pass to ax.tick_params. By None, don't change inherent defaults.
            out_dir (str) : If given, where to save the file.
            fix_invalid (bool) : Fix invalid values.
            line_slope (float) : If given, draw a line with the given slope.
        '''

        if isinstance( slices, int ):
            sl = ( slice(None), slices )
        else:
            sl = slices

        # Get data
        x_data = self.data_object.get_selected_data( x_key, sl=sl, *args, **kwargs )
        y_data = self.data_object.get_selected_data( y_key, sl=sl, *args, **kwargs )

        # Fix NaNs
        if fix_invalid:
            x_mask = np.ma.fix_invalid( x_data ).mask
            y_mask = np.ma.fix_invalid( y_data ).mask
            mask = np.ma.mask_or( x_mask, y_mask )

            x_data = np.ma.masked_array( x_data, mask=mask ).compressed()
            y_data = np.ma.masked_array( y_data, mask=mask ).compressed()

        # Subsample
        if n_subsample is not None:
            sampled_inds = np.random.randint( 0, x_data.size, n_subsample )

            x_data = x_data[sampled_inds]
            y_data = y_data[sampled_inds]

        if x_range is None:
            x_range = [ x_data.min(), x_data.max() ]
        elif isinstance( x_range, float ):
            x_range = np.array( [ -x_range, x_range ])*self.data_object.ptracks.length_scale.iloc[slices]
        if y_range is None:
            y_range = [ y_data.min(), y_data.max() ]
        elif isinstance( y_range, float ):
            y_range = np.array( [ -y_range, y_range ])*self.data_object.ptracks.length_scale.iloc[slices]

        # Plot
        if ax is None:
            fig = plt.figure( figsize=(10,9), facecolor='white' )
            ax = plt.gca()

        s = ax.scatter( x_data, y_data, s=marker_size, color=color, marker=marker )

        # Change the z order
        s.set_zorder( zorder )

        # Halo Plot
        if line_slope is not None:
            line_x = np.array( [ x_data.min(), x_data.max() ] )
            line_y = line_slope*line_x
            ax.plot( line_x, line_y, linewidth=3, linestyle='dashed', )

        # Plot label
        if plot_label is None:
            plt_label = ax.annotate( s=self.label, xy=(0.,1.0225), xycoords='axes fraction', fontsize=label_fontsize,  )
        elif isinstance( plot_label, str ):
            plt_label = ax.annotate( s=plot_label, xy=(0.,1.0225), xycoords='axes fraction', fontsize=label_fontsize,  )
        elif isinstance( plot_label, dict ):
            plt_label = ax.annotate( **plot_label )
        elif plot_label is None:
            pass
        else:
            raise Exception( 'Unrecognized plot_label arguments, {}'.format( plot_label ) )
        if outline_plot_label:
            plt_label.set_path_effects([ path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal() ])

        # Upper right label (info label)
        info_label = ''
        if label_galaxy_cut:
            info_label = r'$r_{ \rm cut } = ' + '{:.3g}'.format( self.data_object.galids.parameters['galaxy_cut'] ) + 'r_{ s}$'
        if label_redshift:
            info_label = r'$z=' + '{:.3f}'.format( self.data_object.ptracks.redshift.iloc[slices] ) + '$, '+ info_label
        if label_galaxy_cut or label_redshift:
            ax.annotate( s=info_label, xy=(1.,1.0225), xycoords='axes fraction', fontsize=label_fontsize,
                ha='right' )

        # Add axis labels
        if add_x_label:
            if x_label is None:
                x_label = x_key
            ax.set_xlabel( x_label, fontsize=label_fontsize )
        if add_y_label:
            if y_label is None:
                y_label = y_key
            ax.set_ylabel( y_label, fontsize=label_fontsize )

        # Limits
        ax.set_xlim( x_range )
        ax.set_ylim( y_range )

        # Set tick parameters
        if tick_param_args is not None:
            ax.tick_params( **tick_param_args )

        # Save the file
        if out_dir is not None:
            save_file = '{}_{:03d}.png'.format( self.label, self.data_object.ptracks.snum[slices] )
            gen_plot.save_fig( out_dir, save_file, fig=fig, dpi=75 )

            plt.close()

    ########################################################################

    def plot_stacked_data(
        self,
        x_key,
        y_keys,
        colors,
        ax = None,
        *args, **kwargs
    ):
        
        if ax is None:
            plt.figure( figsize=(11, 5), facecolor='white' )
            ax = plt.gca()

        y_prev = np.zeros( shape=y_datas.values()[0].shape )

        y_datas = []
        for y_key in y_keys:
            y_data = self.data_object.get_selected_data(
                y_key,
                *args, **kwargs
            ).copy()

            y_datas.append( y_data )

        for i, y_key in y_keys:

            y_next = y_prev + y_datas[i]

            ax.fill_between(
                x_data,
                y_prev,
                y_next,
                color = classification_colors[key],
                alpha = p_constants.CLASSIFICATION_ALPHA,
            )

            # Make virtual artists to allow a legend to appear
            color_object = matplotlib.patches.Rectangle(
                (0, 0),
                1,
                1,
                fc = classification_colors[key],
                ec = classification_colors[key],
                alpha = p_constants.CLASSIFICATION_ALPHA,
            )
            color_objects.append( color_object )
            labels.append( p_constants.CLASSIFICATION_LABELS[key] )

        ax.annotate(
            s=self.label,
            xy=(0., 1.0225),
            xycoords='axes fraction',
            fontsize=22,
        )

        ax.legend(
            color_objects,
            labels,
            prop={'size': 14.5},
            ncol=5,
            loc=(0., -0.28),
            fontsize=20
        )

    ########################################################################

    def plot_time_dependent_data( self,
        ax = None,
        x_range = [ 0., np.log10(8.) ], y_range = None,
        y_scale = 'log',
        x_label = None, y_label = None,
        ):
        '''Make a plot like the top panel of Fig. 3 in Angles-Alcazar+17

        Args:
            ax (axis object) :
                What axis to put the plot on. By None, create a new one on a separate figure.

            x_range, y_range (list-like) :
                [ x_min, x_max ] or [ y_min, y_max ] for the displayed range.

            x_label, y_label (str) :
                Labels for axis. By None, redshift and f(M_star), respectively.

            plot_dividing_line (bool) :
                Whether or not to plot a line at the edge between stacked regions.
        '''

        if ax is None:
            fig = plt.figure( figsize=(11,5), facecolor='white' )
            ax = plt.gca()

        x_data = np.log10( 1. + self.data_object.get_data( 'redshift' ) )

        y_data = self.data_object.get_categories_galaxy_mass()

        for key in p_constants.CLASSIFICATION_LIST_A[::-1]:

            y_data = y_datas[key]

            ax.plot(
                x_data,
                y_data,
                linewidth = 3,
                color = p_constants.CLASSIFICATION_COLORS[key],
                label = p_constants.CLASSIFICATION_LABELS[key],
            )

        if x_range is not None:
            ax.set_xlim( x_range )

        if y_range is not None:
            ax.set_ylim( y_range )

        ax.set_yscale( y_scale )

        tick_redshifts = np.array( [ 0.25, 0.5, 1, 2, 3, 4, 5, 6, 7, ] )
        x_tick_values = np.log10( 1. + tick_redshifts )
        plt.xticks( x_tick_values, tick_redshifts )

        ax.set_xlabel( r'z', fontsize=22, )
        ax.set_ylabel( r'$M_{\star} (M_{\odot})$', fontsize=22, )

        ax.annotate( s=self.label, xy=(0.,1.0225), xycoords='axes fraction', fontsize=22,  )

        ax.legend( prop={'size':14.5}, ncol=5, loc=(0.,-0.28), fontsize=20 )

    ########################################################################
    # Generic Plotting Methods
    ########################################################################

    def same_axis_plot(
        self,
        axis_plotting_method_str,
        variations,
        ax = None,
        figsize = (11, 5),
        out_dir = None,
        add_line_label = False,
        legend_args = { 'prop': {'size': 16.5}, 'loc': 'upper right', 'fontsize': 20 },
        *args, **kwargs
    ):

        if ax is None:
            fig = plt.figure( figsize=figsize, facecolor='white', )
            ax = plt.gca()

        all_plotting_kwargs = utilities.dict_from_defaults_and_variations( kwargs, variations )

        axis_plotting_method = getattr( self, axis_plotting_method_str )
        for key, plotting_kwargs in all_plotting_kwargs.items():

            plotting_kwargs['ax'] = ax

            if add_line_label:
                plotting_kwargs['line_label'] = key

            axis_plotting_method( *args, **plotting_kwargs )

        ax.legend( **legend_args )

        # Save the file
        if out_dir is not None:
            save_file = '{}_{:03d}.png'.format( self.label, self.data_object.ptracks.snum[kwargs['slices']] )
            gen_plot.save_fig( out_dir, save_file, fig=fig, dpi=75 )

            plt.close()

    ########################################################################

    def panel_plot( self,
        panel_plotting_method_str,
        defaults,
        variations,
        slices = None,
        n_rows = 2,
        n_columns = 2,
        plot_locations = [ (0,0), (0,1), (1,0), (1,1) ],
        figsize = (10,9),
        plot_label = None,
        outline_plot_label = False,
        label_galaxy_cut = False,
        label_redshift = True,
        label_fontsize = 24,
        subplot_label_args = { 'xy': (0.075, 0.88), 'xycoords': 'axes fraction', 'fontsize': 20, 'color': 'k',  },
        subplot_spacing_args = { 'hspace': 0.0001, 'wspace': 0.0001, },
        out_dir = None,
        ):
        '''
        Make a multi panel plot of the type of your choosing.

        Args:
            panel_plotting_method_str (str) : What type of plot to make.
            defaults (dict) : Default arguments to pass to panel_plotting_method.
            variations (dict of dicts) : Differences in plotting arguments per subplot.
            slices (slice) : What slices to select. By None, this doesn't pass any slices argument to panel_plotting_method
            plot_label (str or dict) : What to label the plot with. By None, uses self.label.
                Can also pass a dict of full args.
            outline_plot_label (bool) : If True, add an outline around the plot label.
            label_galaxy_cut (bool) : If true, add a label that indicates how the galaxy was defined.
            label_redshift (bool) : If True, add a label indicating the redshift.
            label_fontsize (int) : Fontsize for the labels.
            subplot_label_args (dict) : Label arguments to pass to each subplot for the label for the subplot.
                The actual label string itself corresponds to the keys in variations.
            subplot_spacing_args (dict) : How to space the subplots.
            out_dir (str) : If given, where to save the file.
        '''

        fig = plt.figure( figsize=figsize, facecolor='white', )
        ax = plt.gca()

        fig.subplots_adjust( **subplot_spacing_args )

        if slices is not None:
            defaults['slices'] = slices

        plotting_kwargs = utilities.dict_from_defaults_and_variations( defaults, variations )

        # Setup axes
        gs = gridspec.GridSpec(n_rows, n_columns)
        axs = []
        for plot_location in plot_locations:
            axs.append( plt.subplot( gs[plot_location] ) )

        # Setup arguments further
        for i, key in enumerate( plotting_kwargs.keys() ):
            ax_kwargs = plotting_kwargs[key]

            ax_kwargs['ax'] = axs[i]

            # Subplot label args
            this_subplot_label_args = subplot_label_args.copy()
            this_subplot_label_args['s'] = key
            ax_kwargs['plot_label'] = this_subplot_label_args

            if ax_kwargs['add_colorbar']:
                ax_kwargs['colorbar_args'] = { 'fig_or_ax': fig, 'ax_location': [0.9, 0.125, 0.03, 0.775 ],  }

            # Clean up interior axes
            ax_tick_parm_args = ax_kwargs['tick_param_args'].copy()
            plot_location = plot_locations[i]
            # Hide repetitive x labels
            if plot_location[0] != n_rows -1 :
                ax_kwargs['add_x_label'] = False
                ax_tick_parm_args['labelbottom'] = False
            # Hide repetitive y labels
            if plot_location[1] != 0:
                ax_kwargs['add_y_label'] = False
                ax_tick_parm_args['labelleft'] = False
            ax_kwargs['tick_param_args'] = ax_tick_parm_args

        # Actual panel plots
        panel_plotting_method = getattr( self, panel_plotting_method_str )
        for key in plotting_kwargs.keys():
            panel_plotting_method( **plotting_kwargs[key] )

        # Main axes labels
        # Plot label
        if plot_label is None:
            plt_label = axs[0].annotate( s=self.label, xy=(0.,1.0225), xycoords='axes fraction', fontsize=label_fontsize,  )
        elif isinstance( plot_label, str ):
            plt_label = axs[0].annotate( s=plot_label, xy=(0.,1.0225), xycoords='axes fraction', fontsize=label_fontsize,  )
        elif isinstance( plot_label, dict ):
            plt_label = axs[0].annotate( **plot_label )
        else:
            raise Exception( 'Unrecognized plot_label arguments, {}'.format( plot_label ) )
        if outline_plot_label:
            plt_label.set_path_effects([
                path_effects.Stroke(linewidth=3, foreground='white', background='white'),
                path_effects.Normal()
            ])

        # Upper right label (info label)
        info_label = ''
        if label_galaxy_cut:
            info_label = r'$r_{ \rm cut } = ' + '{:.3g}'.format( self.data_object.galids.parameters['galaxy_cut'] ) + 'r_{ s}$'
        if label_redshift:
            ind = defaults['slices']
            info_label = r'$z=' + '{:.3f}'.format( self.data_object.ptracks.redshift.iloc[ind] ) + '$'+ info_label
        if label_galaxy_cut or label_redshift:
            label_ax = plt.subplot( gs[0,n_columns-1,] )
            label_ax.annotate(
                s=info_label,
                xy=(1.,1.0225),
                xycoords='axes fraction',
                fontsize=label_fontsize,
                ha='right'
            )

        # Save the file
        if out_dir is not None:
            save_file = '{}_{:03d}.png'.format( self.label, self.data_object.ptracks.snum[slices] )
            gen_plot.save_fig( out_dir, save_file, fig=fig )

            plt.close()

    ########################################################################

    def make_multiple_plots( self,
        plotting_method_str,
        iter_args_key,
        iter_args,
        n_processors = 1,
        out_dir = None,
        make_out_dir_subdir = True,
        make_movie = False,
        clear_data = False,
        *args, **kwargs ):
        '''Make multiple plots of a selected type. *args and **kwargs are passed to plotting_method_str.

        Args:
            plotting_method_str (str) : What plotting method to use.
            iter_args_key (str) : The name of the argument to iterate over.
            iter_args (list) : List of argument values to change.
            n_processors (int) : Number of processors to use. Should only be used when saving the data.
            out_dir (str) : Where to save the data.
            make_movie (bool) : Make a movie out of the plots, if True.
            clear_data (bool) : If True, clear memory of the data after making the plots.
        '''

        plotting_method = getattr( self, plotting_method_str )

        if ( out_dir is not None ) and make_out_dir_subdir:
            out_dir = os.path.join( out_dir, self.label )

        def plotting_method_wrapper( process_args ):

            used_out_dir, used_args, used_kwargs = process_args

            plotting_method( out_dir=used_out_dir, *used_args, **used_kwargs )

            del used_out_dir, used_args, used_kwargs

            return

        all_process_args = []
        for iter_arg in iter_args:
            process_kwargs = dict( kwargs )
            process_kwargs[iter_args_key] = iter_arg
            all_process_args.append( ( out_dir, args, process_kwargs ) )

        if n_processors > 1:
            # For safety, make sure we've loaded the data already
            self.data_object.ptracks, self.data_object.galids, self.data_object.classifications

            mp_utils.parmap( plotting_method_wrapper, all_process_args, n_processors=n_processors, return_values=False )
        else:
            for i, iter_arg in enumerate( iter_args ):
                plotting_method_wrapper( all_process_args[i] )

        if make_movie:
            gen_plot.make_movie( out_dir, '{}_*.png'.format( self.label ), '{}.mp4'.format( self.label ), )

        if clear_data:
            del self.data_object.ptracks
            del self.data_object.galids
            del self.data_object.classifications

########################################################################
########################################################################

class PlotterSet( verdict.Dict ):
    '''Container for multiple plotters that is an enhanced dictionary.
    '''

    def __init__( self, data_object_cls, plotter_object_cls, defaults, variations ):
        '''
        Args:
            data_object_cls (object) : Class for the data object.
            plotter_object_cls (object) : Class for the plotter object.
            defaults (dict) : Set of None arguments for loading worldline data.
            variations (dict of dicts) : Labels and differences in arguments to be passed to Worldlines
        '''

        # Load the worldline sets
        storage = {}
        for key in variations.keys():

            kwargs = dict( defaults )
            for var_key in variations[key].keys():
                kwargs[var_key] = variations[key][var_key]

            storage[key] = { 'data_object': data_object_cls( **kwargs ), 'label': key }

        plotters_storage = utilities.SmartDict.from_class_and_args( plotter_object_cls, storage )

        super( PlotterSet, self ).__init__( plotters_storage )
