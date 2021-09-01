#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pluto.py - tools for analyzing outputs from PLUTO
Last Modified: 2021.07.06

Copyright(C) 2021 Shaokun Xie <https://xshaokun.com>
Licensed under the MIT License, see LICENSE file for details
"""

import os
import fire
import numpy as np
import pandas as pd
import pyPLUTO as pypl
import pyPLUTO.pload as pp
from astropy.visualization import quantity_support
quantity_support()
import matplotlib as mpl
import matplotlib.pyplot as plt
from skpy.utilities.tools import nearest
from astropy import units as u
from mpl_toolkits.axes_grid1 import make_axes_locatable



class PlutoDefConstants(object):
# Physical constants in c.g.s units defined in $PLUTO_DIR/Src/pluto.h
    __pluto_def_constants = {
        'CONST_AH'    : 1.008             ,
        'CONST_AHe'   : 4.004             ,
        'CONST_AZ'    : 30.0              ,
        'CONST_amu'   : 1.66053886e-24    ,
        'CONST_au'    : 1.49597892e13     ,
        'CONST_c'     : 2.99792458e10     ,
        'CONST_e'     : 4.80320425e-10    ,
        'CONST_eV'    : 1.602176463158e-12,
        'CONST_G'     : 6.6726e-8         ,
        'CONST_h'     : 6.62606876e-27    ,
        'CONST_kB'    : 1.3806505e-16     ,
        'CONST_ly'    : 0.9461e18         ,
        'CONST_mp'    : 1.67262171e-24    ,
        'CONST_mn'    : 1.67492728e-24    ,
        'CONST_me'    : 9.1093826e-28     ,
        'CONST_mH'    : 1.6733e-24        ,
        'CONST_Msun'  : 2.e33             ,
        'CONST_Mearth': 5.9736e27         ,
        'CONST_NA'    : 6.0221367e23      ,
        'CONST_pc'    : 3.0856775807e18   ,
        'CONST_PI'    : 3.14159265358979  ,
        'CONST_Rearth': 6.378136e8        ,
        'CONST_Rgas'  : 8.3144598e7       ,
        'CONST_Rsun'  : 6.96e10           ,
        'CONST_sigma' : 5.67051e-5        ,
        'CONST_sigmaT': 6.6524e-25        
    }

    def __init__(self):
        for key, value in self.__pluto_def_constants.items():
            setattr(self, key, value)

class PlutoVarsInfo(object):
    # Physical variables
    # Value name / code_unit / dimension
    known_vars = {
        "rho"   :  ("code_density",                    "g/cm**3",         "density"),
        "vx1"   :  ("code_velocity",                   "km/s",         "speed"),
        "vx2"   :  ("code_velocity",                   "km/s",         "speed"),
        "vx3"   :  ("code_velocity",                   "km/s",         "speed"),
        "prs"   :  ("code_density*code_velocity**2",   "erg/cm**3",    "pressure"),
        "speed" :  ("code_velocity",                   "km/s",         "spped"),
        "temp"  :  ("u.K",                             "K",            "temperature"),
        "mass"  :  ("code_density*code_length**3",     "Msun"          "mass"),
        "time"  :  ("code_length/code_veloctiy",       "yr",           "time"),
        "acc"   :  ("code_veloctiy**2/code_length",    "",      "acceleration")
    }

    # def __init__(self, code_unit):
        # code_density = u.def_unit('code_density', represents=code_unit['code_density'])
        # code_length = u.def_unit('code_length', represents=code_unit['code_length'])
        # code_velocity = u.def_unit('code_velocity', represents=code_unit['code_velocity'])
        # for item in self.__known_vars:
            # uni = eval(item[1][0])
            # dim = u.get_physical_type(uni)
            # setattr(self, item[0], {'unit': uni, 'dimension': dim})
        # self.known_vars = self.__known_vars[:][0]



class Dataset(object):
    """ Pluto data work directory
    """

    __slots__=[
        "wdir",
        "init_file",
        "datatype",
        "filetype",
        "endianess",
        "geometry",
        "code_unit",
        "vars",
        "derived_vars",
        "__log_file",
        "__ds"
    ]

    def __init__(self, w_dir='./', datatype='vtk', init_file='pluto.ini'):
        self.wdir = os.path.abspath(w_dir) + '/'
        self.init_file = init_file
        self.datatype = datatype
        
        varfile = self.wdir+self.datatype+'.out'
        with open(varfile,'r') as vfp:
            varinfo = vfp.read().splitlines()
            lastline = varinfo[-1].split()
            max = int(lastline[0])
            self.filetype = lastline[4]
            self.endianess = lastline[5]
            self.vars = lastline[6:]
        self.geometry = 'CARTESIAN'
        self.derived_vars = {}

        # Three base units and default values in pluto code
        self.code_unit={
            "code_density" : 1.0 * u.g/u.cm**3,
            "code_length"  : 1.0 * u.cm,
            "code_velocity": 1.0 * u.cm/u.s
        }

        if 'definitions.h' in os.listdir(self.wdir):
            with open(self.wdir+'definitions.h','r') as df:
                for line in df.readlines():
                    if line.startswith('#define  GEOMETRY'):
                        self.geometry = line.split()[-1]
                    if line.startswith('#define  UNIT'):
                        name = line.split()[1]
                        expr = line.split()[-1]
                        var = name.split('_')[-1]
                        key = 'code_'+var.lower()
                        if 'CONST' in expr:
                            expr = expr.replace('CONST', 'PlutoDefConstants().CONST')
                        self.code_unit[key] = eval(expr) * self.code_unit[key].unit
        else:
            print('Could not open definitions.h! The values of attributes [geometry, code_units] did not update, you can specifiy them later, and assign units by in_code_unit() manually.')


    def __getitem__(self, index):
        # index: int or time
        ns = self._number_step(index)
        ds = Snapshot(ns, w_dir=self.wdir, datatype=self.datatype)
        return ds


    def info(self):
        for attr in self.__slots__:
            if hasattr(self, attr):
                print(f'{attr:15}:  {getattr(self, attr)}')


    def _number_step(self, ns):
        """ find number step of data file

        ns -- should be a integer in default,
                but if it is a negative integer, return the last number step
                or if it is a float, it is assumed to be time, and return the nearst number step
        """
        hdr = ['time','dt','Nstep']
        log_file = pd.read_table(self.wdir+self.datatype+'.out',sep=' ', usecols=[1,2,3], names=hdr)

        if type(ns) is int:
            if ns < 0:
                return log_file.index[-1]
            else:
                return ns
        elif type(ns) is float:         #  given a specific [time], find [ns] corresponding nearst existed data [time].
            return nearest(log_file['time'],ns)
        else:
            raise(TypeError(f"ns({ns}) should be int or float, now it is {type(ns)}."))


class Snapshot(Dataset):
    """ 
    """

    __slots__= [
        "nstep",
        "time",
        "dt",
        "grids",
        "coord",
    ]

    def __init__(self, ns, w_dir, datatype):
        super().__init__(w_dir, datatype)

        D = pp.pload(ns, w_dir=self.wdir, datatype=self.datatype)
        self.nstep = D.NStep
        self.time = D.SimTime
        self.dt = D.Dt

        grids_info = [
            "n1","n2","n3",                 # number of computational cells
            "n1_tot","n2_tot","n3_tot",     # total cells including ghost cells
        ]
        self.grids = {}
        for key in grids_info:
            self.grids[key] = getattr(D, key)

        coord_info = [
            'x1','x2','x3',     # cell center coordinate
            'dx1','dx2','dx3',  # cell width
            'x1r','x2r','x3r'  # cell edge coordinate
        ]
        self.coord = {}
        for key in coord_info:
            self.coord[key] = getattr(D, key)

        vars_info = {}
        for var in self.vars:
            vars_info[var] = getattr(D, var).T
        self.vars = vars_info

        if 'definitions.h' in os.listdir(self.wdir):
            self.in_code_unit()

    def __getitem__(self, key):
        return self.__vars_value[key]

    def info(self):
        for attr in self.__slots__:
            value = getattr(self, attr)
            if not isinstance(value, dict):
                print(f'{attr:15}:  {value}')

    def add_vars(self, name, value, unit=None, dimensions=None): # in construction
        defined_vars = PlutoVarsInfo.known_vars
        if name in defined_vars:
            unit = eval(defined_vars[name][0])
            if dimensions:
                if dimensions not in u.get_physical_type(unit):
                    raise ValueError(f'The given dimension {dimensions} is not compatible with the unit')
            else:
                print('Warning: the option [dimensions] is empty, the unit was not checked, and it could be wrong!')
        elif unit==None and isinstance(value, u.quantity):
            raise ValueError(f'The variable {name} cannot be found in the defined list, the unit is required.')
        else:
            pass

    def in_code_unit(self):
        """ Assign code units
        """
        code_density = u.def_unit('code_density', represents=self.code_unit['code_density'])
        code_length = u.def_unit('code_length', represents=self.code_unit['code_length'])
        code_velocity = u.def_unit('code_velocity', represents=self.code_unit['code_velocity'])
        # u.add_enabled_units([code_density, code_length, code_velocity])

        if isinstance(self.coord['x1'], u.Quantity):    # in case units are already assigned
            for key in self.coord:
                self.coord[key] = self.coord[key].to(code_length)
            
            pluto_vars = PlutoVarsInfo.known_vars
            for key in self.vars:
                self.vars[key] = self.vars[key].to(eval(pluto_vars[key][0]))
            
            for key in self.derived_vars:
                self.derived_vars[key] = self.derived_vars[key].to(eval(pluto_vars[key][0]))

            self.time = self.time.to(code_length/code_velocity)
            self.dt = self.dt.to(code_length/code_velocity)
        else:
            for key in self.coord:
                self.coord[key] *= code_length
            
            pluto_vars = PlutoVarsInfo.known_vars
            for key in self.vars:
                self.vars[key] *= eval(pluto_vars[key][0])
            
            for key in self.derived_vars:
                self.derived_vars[key] *= eval(pluto_vars[key][0])

            self.time *= code_length/code_velocity
            self.dt *= code_length/code_velocity

    def in_astro_unit(self):
        """ convert the units to those commonly used in astro
        """

        if not isinstance(self.coord['x1'], u.Quantity):    # in case not quantity, assign code_unit first
            self.in_code_unit()

        for key in self.coord:
            self.coord[key] = self.coord[key].to(u.kpc)
        
        pluto_vars = PlutoVarsInfo.known_vars
        for key in self.vars:
            unit = u.Unit(pluto_vars[key][1])
            if unit == "":
                self.vars[key] = self.vars[key].cgs
            else:
                self.vars[key] = self.vars[key].to(u.Unit(pluto_vars[key][1]))
        
        for key in self.derived_vars:
            if unit == "":
                self.vars[key] = self.derived_vars[key].cgs
            else:
                self.derived_vars[key] = self.derived_vars[key].to(u.Unit(pluto_vars[key][1]))
        self.time = self.time.to(u.yr)
        self.dt = self.dt.to(u.yr)


def load(fn):  # in construction
    """ load PLUTO simulation all outputs and return time series data.
    """
    fn = os.path.expanduser(fn)
    if any(wildcard in fn for wildcard in "[]?!*"):
       return 0
    return 0






class Preview(object):
    """ Class for previewing data results.

    Parameters:
    -----------
    figsize : tuple. Size of the figure presented. Default: (10,8)
    loc : str. Path of the data located. Default: './'
    dest : str. Destination for saving the figure. None means not saving but prom

    """

    # matplotlib setup
    mpl.rcParams['mathtext.fontset'] = 'cm'
    mpl.rcParams['font.family'] = 'serif'

    def __init__(self, wdir='./', datatype='vtk'):
        self.wdir = os.path.abspath(wdir)+'/'
        self.datatype = datatype

        self.fig = plt.figure(figsize=(5,4), tight_layout=True)


    def show(self):
        """ show figure in prompt window
        """

        return plt.show()


    def save(self, name, path='./', **kwargs):
        """ save figure
        """

        model = self.wdir.split('/')[-2]
        plt.savefig(path+name+f'-{model}.jpg',bbox_inches='tight', pad_inches=0.02, dpi=kwargs.get('dpi',300))


    def display(self, ns, var, log=True, **kwargs):
        """ Display a 2D data using the matplotlib's pcolormesh

        Parameters:
        -----------
        ns: number step of data filens
        var: variable that needs to be displayed

        **kwargs:
        ---------
        wdir:     path to the directory which has the data files
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
        # quantity_support()

        ds = Dataset(w_dir=kwargs.get('wdir', self.wdir), datatype=kwargs.get('datatype', self.datatype))
        ss = ds[ns]
        if kwargs.get('in_astro_unit'):
            ss.in_astro_unit()

        x1 = ss.coord['x1']
        x2 = ss.coord['x3']
        y = ss.coord['x2']
        yaxis = nearest(y, 0.0)
        value = ss.vars[var]
        value = value[:,yaxis,:]

        x1, x2 = np.meshgrid(x1,x2)
        ax1 = self.fig.add_subplot(111)
        ax1.set_aspect('equal')

        if isinstance(x1, u.Quantity):  # pcolormesh does not support Quantity
            x1 = x1.value
            x2 = x2.value
            value = value.value
        ax1.axis([np.min(x1),np.max(x1),np.min(x2),np.max(x2)])
        pcm = ax1.pcolormesh(x1,x2,value,vmin=kwargs.get('vmin',np.min(value)),vmax=kwargs.get('vmax',np.max(value)), cmap=kwargs.get('cmap'))

        plt.title(kwargs.get('title',f"t = {ss.time:.3e}"),size=kwargs.get('size'))
        # plt.xlabel(kwargs.get('label1',"R(kpc)"),size=kwargs.get('size'))
        # plt.ylabel(kwargs.get('label2',"z(kpc)"),size=kwargs.get('size'))

        # Add a new axes beside the plot to present colorbar
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%", pad=0.0)
        cb = plt.colorbar(pcm, cax=cax,orientation='vertical')
        defined_vars = PlutoVarsInfo.known_vars
        if log:
            cb.ax.set_ylabel(r'$\log\;$'+defined_vars.get(var)[0])
        else:
            cb.ax.set_ylabel(defined_vars.get(var)[0])

        return self


    def line(self, ns, var, x1=None, x2=None, x3=None, **kwargs):
        """ Show 1D profile

        Parameters:
        -----------
        ns:   Step Number of the data file
        var:  variable name that needs to be displayed
        dir:  direction of the data (only along axis)
        offset:  distance from center

        **kwargs:
        ---------
        wdir:     path to the directory which has the data files
        datatype:  Datatype (default is set to read .dbl data files)
        x1range:   List with min and max value of x1 coordinates for zooming
        x2range:   List with min and max value of x2 coordinates for zooming
        x3range:   List with min and max value of x3 coordinates for zooming
        """

        ds = Dataset(w_dir=kwargs.get('wdir', self.wdir), datatype=kwargs.get('datatype', self.datatype))
        ss = ds[ns]
        indx = [x1,x2,x3]
        label = ['x1','x2','x3']
        for i in range(3):
            indx[i] = nearest(ss.coord[label[i]], indx[i])
            if indx[i] == None:
                x = ss.coord[label[i]]
        indx = str(indx[::-1]).replace('None',':')
        value = eval('ss.vars[var]'+indx)

        # slice the data
        # idx = nearest(ss.coord[of[dir]], offset)
        # value = ss.vars[var][:,0,:]
        # value = value[:,idx] if dir=='x1' else value[idx,:]

# {of[dir]}={offset}, 
        plt.plot(x, value, label=f"t={ss.time:.3e}")

        plt.title(kwargs.get('title',"Title"),size=kwargs.get('size'))
        # plt.xlabel(kwargs.get('label1',"Xlabel"),size=kwargs.get('size'))
        # plt.ylabel(kwargs.get('label2',"Ylabel"),size=kwargs.get('size'))

        if kwargs.get('xlog'): plt.xscale('log')
        if kwargs.get('ylog'): plt.yscale('log')

        plt.legend(frameon=False)

        return self


    def hist(self, *var, op=None, **kwargs):
        """ Preview temperal evolution stored in hist.log file

        """

        hist = pd.read_csv(self.wdir+'hist.dat', sep='\s+', index_col="t")
        hist = hist[list(var)]

        if op==None:
            ax = plt.plot(hist)
            plt.legend(ax,var)
        elif op=='diff':
            ax = plt.plot(hist-hist.iloc[[0]].values)
            plt.legend(ax,[f"{v}-{v}(0)" for v in var])
        elif op=='norm':
            ax = plt.plot(hist/hist.iloc[[0]].values)
            plt.legend(ax,[f"{v}/{v}(0)" for v in var])
        else:
            raise KeyError(f"Operation [{op}] has not defined yet.")

        plt.title(kwargs.get('title',"Title"),size=kwargs.get('size'))
        plt.xlabel(kwargs.get('label1',"Time (code unit)"),size=kwargs.get('size'))
        plt.ylabel(kwargs.get('label2',"Ylabel"),size=kwargs.get('size'))

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
            return nearest(self.log['time'],ns)
        else:
            raise(TypeError(f"ns({ns}) should be int or float, now it is {type(ns)}."))


if __name__ == "__main__":
    fire.Fire(Preview)
