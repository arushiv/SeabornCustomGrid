import pandas
import numpy
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from faceted_jointplots import SeabornFig2Grid 



def myjoint(group, x, y, **kwargs):
    """Normal hexbin jointplot """
    kind = kwargs.get('kind', "hex")
    gridsize = kwargs.get('gridsize', 25)
    x_label = kwargs.get('x_label', True)
    y_label = kwargs.get('y_label', True)
    cmap = kwargs.get('cmap', "Blues")

    N = len(group.index)
    g = sns.jointplot(data=group, x=x, y=y, kind=kind, joint_kws={'gridsize':gridsize, 'cmap':cmap, 'bins':N})

    if not x_label:
        g.ax_joint.set_xlabel("")

    if not y_label:
        g.ax_joint.set_ylabel("")

    # Name facet Columns on topmost facets
    if group.name[0] == kwargs['colLabel']:
        g.ax_marg_x.set_title(group.name[1])

    # Label Row and y axis on leftmost facets
    if group.name[1] == kwargs['ylabcol']:
        g.ax_joint.set_ylabel(f"{group.name[0]}\n\n{kwargs['main_y_label']}")

    # Label x axis on lowermost facets
    if group.name[0] == kwargs['xlabcol']:
        g.ax_joint.set_xlabel(kwargs['main_x_label'])

    # # Label rows on the rightmost facets
    # if group.name[1] == kwargs['rowLabel']:
    #     g.ax_marg_y.set_title
                                                 
    return g


def makeGrid(d, row, col, x, y, plotfunc=myjoint, figsize=(6,6), figname=None, **kwargs):
    """Take pandas dataframe and make a grid of jointplots for variables x and y, faceted over given 'row' and 'column' columns of the dataframe.
    Usage: makeGrid(d, 'rowCol', 'colCol', 'x', 'y', {'kind':'kde'})
    kwargs: 
    gridsize = Jointplot hexbin gridsize
    main_x_label = x label on the lowermost facets
    main_y_label = y label on the leftmost facets
    """
    
    nrow = len(d[row].drop_duplicates())
    ncol = len(d[col].drop_duplicates())
    
    kwargs['colLabel'] = d[row].drop_duplicates().sort_values().iloc[0]
    kwargs['ylabcol'] = d[col].drop_duplicates().sort_values().iloc[0]
    kwargs['xlabcol'] = d[row].drop_duplicates().sort_values(ascending=False).iloc[0]
    kwargs['rowLabel'] = d[col].drop_duplicates().sort_values(ascending=False).iloc[0]
    
    list_of_plots = d.groupby([row, col]).apply(lambda group: plotfunc(group, x, y, **kwargs))
    plotdf = list_of_plots.reset_index().reset_index()

    sfg = SeabornFig2Grid
    
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(nrow, ncol)
    gs.update(wspace=0.00, hspace=0.00)
    
    plotdf.apply(lambda x: sfg(x[0], fig, gs[x['index']]), axis=1)

    gs.tight_layout(fig)
    
    plt.savefig(figname)
    
    #plt.show()
