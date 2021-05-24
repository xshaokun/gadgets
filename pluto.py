#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pluto.py - tools for analyzing outputs from PLUTO
Last Modified: 2021.05.21

Copyright(C) 2021 Shaokun Xie <https://xshaokun.com>
Licensed under the MIT License, see LICENSE file for details
"""

import os
import fire
import numpy as np
import pandas as pd
import pyPLUTO as pypl
import pyPLUTO.pload as pp
import matplotlib as mpl
import matplotlib.pyplot as plt
from utilities.tools import *
from mpl_toolkits.axes_grid1 import make_axes_locatable



# matplotlib setup
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['font.family'] = 'serif'

varlist = {
    'rho': 'Density',
    'prs': 'Pressure',
}


class Image(object):
    """Class for quickly demonstrating data results.

    Parameters:
    -----------
    figsize : tuple. Size of the figure presented. Default: (10,8)
    loc : str. Path of the data located. Default: './'
    dest : str. Destination for saving the figure. None means not saving but prom

    """
    def __init__(self, **kwargs):
        self.wdir = kwargs.get('w_dir', './')
        self.datatype = kwargs.get('datatype', 'dbl')

        hdr = ['time','dt','Nstep']
        self.log = pd.read_table(self.wdir+self.datatype+'.out',sep=' ', usecols=[1,2,3], names=hdr)  # read *.out log file

        self.fig = plt.figure(figsize=kwargs.get('figsize',(5,4)), dpi=300, tight_layout=True)

    def show(self):
        """ show figure in prompt window
        """

        plt.legend(frameon=False)
        return plt.show()

    def display(self, ns, var, **kwargs):
        """ Display a 2D data using the matplotlib's pcolormesh

        Parameters:
        -----------
        ns: number step of data filens
        var: variable that needs to be displayed

        **kwargs:
        ---------
        w_dir:     path to the directory which has the data files
        datatype:  Datatype (default is set to read .dbl data files)
        x1range:   List with min and max value of x1 coordinates for zooming 
        x2range:   List with min and max value of x2 coordinates for zooming 
        x3range:   List with min and max value of x3 coordinates for zooming 

        vmin:   The minimum value of the 2D array (Default : min(var))
        vmax:   The maximum value of the 2D array (Default : max(var))
        title:  Sets the title of the image.
        label1: Sets the X Label (Default: 'XLabel')
        label2: Sets the Y Label (Default: 'YLabel')
        cmap:  color scheme of the colorbar (Default : jet)
        size:  fontsize

        """

        ns = self.NumberStep(ns)
        D = pp.pload(ns, w_dir=kwargs.get('w_dir'), datatype=kwargs.get('datatype'), \
            x1range=kwargs.get('x1range'), x2range=kwargs.get('x2range'), x3range=kwargs.get('x3range'))

        x1 = D.x1
        x2 = D.x2
        ds = getattr(D, var).T if kwargs.get('nolog') else np.log(getattr(D, var).T)

        ax1 = self.fig.add_subplot(111)
        ax1.set_aspect('equal')
    
        ax1.axis([np.min(x1),np.max(x1),np.min(x2),np.max(x2)])
        pcm = ax1.pcolormesh(x1,x2,ds,vmin=kwargs.get('vmin',np.min(ds)),vmax=kwargs.get('vmax',np.max(ds)), cmap=kwargs.get('cmap','jet'))
        
        plt.title(kwargs.get('title',f"t = {D.SimTime:.3e}"),size=kwargs.get('size'))
        plt.xlabel(kwargs.get('label1',"Xlabel"),size=kwargs.get('size'))
        plt.ylabel(kwargs.get('label2',"Ylabel"),size=kwargs.get('size'))
        
        # Add a new axes beside the plot to present colorbar
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="2%", pad=0.0)
        cb = plt.colorbar(pcm, cax=cax,orientation='vertical')
        if kwargs.get('nolog'):
            cb.ax.set_ylabel(varlist.get(var))
        else:
            cb.ax.set_ylabel(r'$\log\;$'+varlist.get(var))

        plt.show()


    def line(self, ns, var, dir, offset, **kwargs):
        """ Show 1D profile

        Parameters:
        -----------
        ns:   Step Number of the data file
        var:  variable name that needs to be displayed
        dir:  direction of the data (only along axis)
        offset:  distance from center

        **kwargs:
        ---------
        w_dir:     path to the directory which has the data files
        datatype:  Datatype (default is set to read .dbl data files)
        x1range:   List with min and max value of x1 coordinates for zooming 
        x2range:   List with min and max value of x2 coordinates for zooming 
        x3range:   List with min and max value of x3 coordinates for zooming 
        """

        ns = self.NumberStep(ns)
        D = pp.pload(ns, w_dir=kwargs.get('w_dir'), datatype=kwargs.get('datatype'), \
            x1range=kwargs.get('x1range'), x2range=kwargs.get('x2range'), x3range=kwargs.get('x3range'))

        # slice the data
        x = D.x2 if dir=='x1' else D.x1
        idx = nearst(getattr(D,dir), offset)
        ds = getattr(D,var)[idx,:] if dir=='x1' else getattr(D,var)[:,idx]

        plt.plot(x, ds, label=f"{dir}={offset}, t={D.SimTime:.3e}")

        plt.title(kwargs.get('title',"Title"),size=kwargs.get('size'))
        plt.xlabel(kwargs.get('label1',"Xlabel"),size=kwargs.get('size'))
        plt.ylabel(kwargs.get('label2',"Ylabel"),size=kwargs.get('size'))
        plt.legend(frameon=False)

        if kwargs.get('xlog'): plt.xscale('log')
        if kwargs.get('ylog'): plt.yscale('log')

        return self


    def NumberStep(self, ns):
        """ find number step of data file

        ns -- should be a integer in default, 
                but if it is a negative integer, return the last number step
                or if it is a float, it is assumed to be time, and return the nearst number step
        """

        if type(ns) is int:
            if ns < 0:
                return self.log.index[-1]
            else:
                return ns
        elif type(ns) is float:         #  given a specific [time], find [ns] corresponding nearst existed data [time].
            return nearst(self.log['time'],ns)
        else:
            raise(TypeError(f"ns({ns}) should be int or float, now it is {type(ns)}."))


if __name__ == "__main__":
    fire.Fire(Image)