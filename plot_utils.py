import pandas
import numpy
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from faceted_jointplots import SeabornFig2Grid 
plt.rcParams.update({'figure.max_open_warning': 0})


def myjoint(group, x, y, facet_wrap, **kwargs):
    """Normal hexbin jointplot """
    kind = kwargs.get('kind', "hex")
    gridsize = kwargs.get('gridsize', 20)
    x_label = kwargs.get('x_label', True)
    y_label = kwargs.get('y_label', True)
    cmap = kwargs.get('cmap', "Blues")
    x_range = kwargs.get('x_range', None)
    y_range = kwargs.get('y_range', None)
    main_x_label = kwargs.get('main_x_label', "")
    main_y_label = kwargs.get('main_y_label', "")
    save_each_fig = kwargs.get('save_each_fig', False)
    line = kwargs.get('line', False)
    
    N = len(group.index)

    if kind == "scatter":
        g = sns.jointplot(data=group, x=x, y=y, kind=kind, joint_kws={'cmap':cmap, 'alpha':0.1, 's':2}, xlim=x_range, ylim=y_range)
    else:
        g = sns.jointplot(data=group, x=x, y=y, kind=kind, joint_kws={'gridsize':gridsize, 'mincnt':0, 'cmap':cmap, 'bins':N}, xlim=x_range, ylim=y_range)

    if not x_label:
        g.ax_joint.set_xlabel("")

    if not y_label:
        g.ax_joint.set_ylabel("")

    # custom  plot vline for each facet
    if line and len(line) == 2: 
        g.ax_joint.plot(line[0], line[1], color="black", linestyle='--')

    # Labelling x, y axis titles, plot titles:
    if facet_wrap is None:
        # Name facet Columns on topmost facets
        if group.name[0] == kwargs['colLabel']:
            g.ax_marg_x.set_title(group.name[1])

        # Label Row and y axis on leftmost facets
        if group.name[1] == kwargs['ylabcol']:
            g.ax_joint.set_ylabel(f"{group.name[0]}\n\n{main_y_label}")

        # Label x axis on lowermost facets
        if group.name[0] == kwargs['xlabcol']:
            g.ax_joint.set_xlabel(main_x_label)
    else:
        # Set titles for all facets
        g.ax_marg_x.set_title(group.name[1])

        # Label Row and y axis on leftmost facets
        if group.name[1] in kwargs['ylabcol']:
            g.ax_joint.set_ylabel(f"{main_y_label}")

        # Label x axis on lowermost facets
        if group.name[0] in kwargs['xlabcol']:
            g.ax_joint.set_xlabel(main_x_label)
        else:
            g.ax_joint.set_xlabel("")

    if save_each_fig:
        figname = f"{save_each_fig}.{group.name[0]}.{group.name[1]}.pdf"
        plt.tight_layout()
        plt.savefig(figname)
        
    return g


def makeGrid(d, row, col, x, y, plotfunc=myjoint, figsize=(6,6), figname=None, facet_wrap=None, **kwargs):
    """Take a pandas dataframe and make a grid of jointplots for variables x and y, faceted over given 'row' and 'column' columns of the dataframe.
    Usage : makeGrid(d, 'rowCol', 'colCol', 'x', 'y', {'kind':'kde'})
    row : dataframe column name of categorical variables which would make rows
    col : dataframe column name of categorical variables which would make columns
    x : dataframe column name for x axis
    y : dataframe column name for y axis
    plotfunc : function to plot. Default = myjoint which makes a jointplot optionally taking in kwargs
    figsize : (width, height) in inches
    facet_wrap: Wrap columns after these many plots in the top row. Labels column names for all plot titles. Row names for left most rows. X axis titles for bottom most row. Still needs col and row parameters though.
    kwargs: 
    kind : Kind of seaborn jointplot
    gridsize : Jointplot hexbin gridsize
    x_label : Default True. Will label x axis for each facet. 
    y_label : Default True. Will label y axis for each facet. 
    main_x_label : x label on the lowermost facets
    main_y_label : y label on the leftmost facets
    x_range : range of x axis (tuple)
    y_range : range of y axis (tuple)
    save_each_fig : string - save each individual facet at string.groupname.pdf 
    line : Default False. Provide a list of lists of x and y coordinates to plot a line on each ax_joint. Eg. supply [[0,0],[-2,2]] to plot a vertical line x=[0,0] and y=[-2,2]
    """
    
    nrow = len(d[row].drop_duplicates())
    ncol = len(d[col].drop_duplicates())
    if facet_wrap is not None:
        ncol = facet_wrap
        get_ylabcol = d[col].drop_duplicates().sort_values()
        kwargs['ylabcol'] = [get_ylabcol.iloc[0], get_ylabcol.iloc[facet_wrap]]
        kwargs['xlabcol'] = d[row].drop_duplicates().sort_values(ascending=False).iloc[0]
        get_rowLabel = d[col].drop_duplicates().sort_values(ascending=False)
        kwargs['rowLabel'] = [get_rowLabel.iloc[0], get_rowLabel.iloc[facet_wrap]]

    else:
        kwargs['colLabel'] = d[row].drop_duplicates().sort_values().iloc[0]
        kwargs['ylabcol'] = d[col].drop_duplicates().sort_values().iloc[0]
        kwargs['xlabcol'] = d[row].drop_duplicates().sort_values(ascending=False).iloc[0]
        kwargs['rowLabel'] = d[col].drop_duplicates().sort_values(ascending=False).iloc[0]

    list_of_plots = d.groupby([row, col]).apply(lambda group: plotfunc(group, x, y, facet_wrap,**kwargs))
    plotdf = list_of_plots.reset_index().reset_index()

    sfg = SeabornFig2Grid
    
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(nrow, ncol)
    gs.update(wspace=0.00, hspace=0.00)
    
    plotdf.apply(lambda x: sfg(x[0], fig, gs[x['index']]), axis=1)

    gs.tight_layout(fig)

    if ".png" in figname:
        plt.savefig(figname)

    else:
        plt.savefig(figname)

