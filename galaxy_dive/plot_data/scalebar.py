# -*- coding: utf-8 -*-
# -*- mode: python -*-
# Adapted from mpl_toolkits.axes_grid1
# LICENSE: Python Software Foundation (http://docs.python.org/license.html)

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.offsetbox import AnchoredOffsetbox
from matplotlib.ticker import MaxNLocator

########################################################################

class AnchoredScaleBar( AnchoredOffsetbox ):

    def __init__( self, transform, sizex=0, sizey=0, labelx=None, labely=None, loc=4,
                 pad=0.1, borderpad=0.1, sep=2, prop=None, xorder='text above', 
                 textprops=None, linewidth=3, bbox=[1, 1.115, 0, 0], bbox_transform=None, **kwargs ):
        """Draw a horizontal and/or vertical  bar with the size in data coordinate
        of the give axes. A label will be drawn underneath (center-aligned).
        
        This is from GitHubGist user dmeliza, with the code found at https://gist.github.com/dmeliza/3251476

        Args:
          transform : the coordinate frame (typically axes.transData)
          sizex,sizey : width of x,y bar, in data units. 0 to omit
          labelx,labely : labels for x,y bars; None to omit
          loc : position in containing axes
          pad, borderpad : padding, in fraction of the legend font size (or prop)
          sep : separation between labels and bars in points.
          **kwargs : additional arguments passed to base class constructor
          linewidth : Thickness of the scale bar
          bbox : Where the scale bar should be relative to the axis
        """
        from matplotlib.patches import Rectangle
        from matplotlib.offsetbox import AuxTransformBox, VPacker, HPacker, TextArea, DrawingArea
        bars = AuxTransformBox(transform)
        if sizex:
            bars.add_artist(Rectangle((0,0), sizex, 0, fc="none", linewidth=linewidth))

        if sizey:
            bars.add_artist(Rectangle((0,0), 0, sizey, fc="none"), linewidth=linewidth)

        if sizex and labelx:
            text_area = TextArea(labelx, textprops=textprops, minimumdescent=False)
            if xorder == 'text above':
              bar_child = [text_area, bars]
            elif xorder == 'text below':
              bar_child = [bars, text_area]
            bars = VPacker(children=bar_child, align="right", pad=0, sep=sep)
        if sizey and labely:
            text_area = TextArea(labely, textprops=textprops)
            bars = HPacker(children=[text_area, bars],
                            align="center", pad=0, sep=sep)

        AnchoredOffsetbox.__init__(self, loc, pad=pad, borderpad=borderpad,
                                   child=bars, prop=prop, frameon=False, 
                                   bbox_transform=bbox_transform, **kwargs)

        AnchoredOffsetbox.set_bbox_to_anchor(self, bbox, transform=bbox_transform)

def add_scalebar( ax, matchx=True, matchy=True, hidex=True, hidey=True, **kwargs ):
    """ Add scalebars to axes
    Adds a set of scale bars to *ax*, matching the size to the ticks of the plot
    and optionally hiding the x and y axes

    Args:
      ax : the axis to attach ticks to
      matchx,matchy : if True, set size of scale bars to spacing between ticks
                    if False, size should be set using sizex and sizey params
      hidex,hidey : if True, hide x-axis and y-axis of parent
      **kwargs : additional arguments passed to AnchoredScaleBars

    Returns:
      created scalebar object
    """
    def f(axis):
        l = axis.get_majorticklocs()
        return len(l)>1 and (l[1] - l[0])
    
    if matchx:
        kwargs['sizex'] = f(ax.xaxis)
        kwargs['labelx'] = str(kwargs['sizex'])
    if matchy:
        kwargs['sizey'] = f(ax.yaxis)
        kwargs['labely'] = str(kwargs['sizey'])
        
    sb = AnchoredScaleBar(ax.transData, **kwargs)
    ax.add_artist(sb)

    if hidex : ax.xaxis.set_visible(False)
    if hidey : ax.yaxis.set_visible(False)

    return sb

