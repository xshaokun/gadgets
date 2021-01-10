#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fermi.py - tools for analyzing outputs of fermi.f
Last Modified: 2021.01.09

Copyright(C) 2020 Shaokun Xie <https://xshaokun.com>
Licensed under the MIT License, see LICENSE file for details
"""


import os
import sys
import numpy as np
import pandas as pd
import skpy.astroeqs as eqs
from astropy import units as u
from astropy import constants as cons
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skpy.utilities.logger import fmLogger as mylog
import fire


# format for log
PARA = "Parameter: {0:<20}\t = {1:<10}"
PROP = "Property: {0:<20}\t = {1:<10}"
DIME = "Dimension: {0:<20}\t : {1:<10}"

# matplotlib setup
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['axes.titlepad'] = 40

# axis label for variables
variables = {
    'den': r'Density (g/cm$^3$)',
    'z': 'Metallicity',
    'e': r'$E_\mathrm{th}$ (erg/cm$3$)',
    'ecr': r'$E_\mathrm{CR}$ (erg/cm$3$)',
    'ux': r'$v_z$ (km/s)',
    'uy': r'$v_R$ (km/s)'
}

# axis label for coordinate
direction = {
    'z': 'z (kpc)',
    'R': 'R (kpc)',
    'r': 'r (kpc)',
}

# define some constants used in the code
varlist = ['den', 'z', 'e', 'pot', 'ux', 'uy']
gamma = 5./3.
qmu = 0.61
qmue = 5*qmu/(2+qmu)
mp = cons.m_p.cgs.value



class FermiData(object):
    """Tools for reading output data of fermi.f

    Used for reading output data of fermi.f, including the dimension and size of meshgrid,
    variable outputs and logging files.

    Args:
        dirpath: str, optional
            Directory path to be loaded. Default is './', which assumes the you work in
            current directory.

    Attributes:
        dir_path: str
            Path to directory form which the data is read.
        iezone: int
            the number of even spaced grids.
        ilzone: int
            the numver of logarithmic spaced grids.
        ezone: float
            the length of even spaced region.
        izone: int
            the number of total grids in one direction. (both directions are same.)
        reso: float
            the resolution for the even spaced grids.
        kprint: list
            list of time (year) for output.

    Example:
        >>> data = FermiData(dirpath='./data/fermi/')
    """

    def __init__(self, dirpath='./'):
        self.dir_path = os.path.abspath(dirpath)
        mylog.info(f'Import data from {self.dir_path}')

        with open(self.dir_path+'/fermi.inp','r') as f:
            self.input = f.readlines()
            self.bc = self.input[0].split()[:4]
            dims = self.input[2].split()
            self.iezone = int(dims[0])
            self.ilzone = int(dims[1])
            self.ezone = float(dims[2])
            self.izone = self.iezone + self.ilzone
            self.reso = self.ezone/self.iezone
            self.kprint = np.append('0',self.input[20].split()[:-1])

        mylog.info(PARA.format("bundary_condition",f"{self.bc}"))
        mylog.info(PARA.format("equally_spaced_grids",f"{self.iezone}"))
        mylog.info(PARA.format("log_spaced_grids",f"{self.ilzone}"))
        mylog.info(PARA.format("equally_spaced_region",f"{self.ezone}"))
        mylog.info(PARA.format("inner_resolution",f"{self.reso}"))
        mylog.info(PARA.format("time_series",f"{self.kprint}"))


    def read_inp(self , row, col):
        """read parameter from input file.

        Args:
            row: int
                the row number of parameter in the file. (Python style)

            col: int
                the colume number of parameter in the file. (Python style)

        Returns:
            para: str

        Example:
            >>> data = FermiData(dirpath='./data/fermi/')
            >>> data.read_inp(1,3) # Read the parameter at the 2nd row and the 4th colume.
        """
        line = self.input[row].split()
        para = line[col]

        return para


    def read_coord(self, var):
        """read coordinate file.

        Args:
            var: str
                the prefix of coordinate file.

        Returns:
            x: numpy.ndarray
                1D coordinate in the unit of kpc.

        Example:
            >>> data = FermiData(dirpath='./data/fermi/')
            >>> data.read_coord('xh')  # Read volume-centered coordinate
            >>> data.read_coord('x')  # Read volume-boundary coordinate
        """

        filename = f"{self.dir_path}/{var}ascii.out"
        mylog.info(f"====> call method {sys._getframe().f_code.co_name}(var={var})")


        x = np.fromfile(filename,sep=" ") * u.cm.to(u.kpc)
        mylog.info(DIME.format(f"{var}",f"{x.shape}"))
        return x


    def get_dvolume(self):
        """get volume of each cell

        In the cylindrical symmetric coordinate, dvol = pi * R^2 * dz

        Return:
            dvol: numpy.ndarray
                in the unit of kpc^-3
        """

        coord = self.read_coord('x') * u.kpc.to(u.cm)
        rr, zz = np.meshgrid(coord, coord)
        dz = np.diff(zz, axis=0)[:,:-1]
        dr2 = np.diff(rr*rr)[:-1]
        dvol = np.pi*dr2*dz
        return dvol


    def get_theta(self):
        """get theta coordinate of each cell

        From the cylindrical symmetric coordinate to spherical symmetric coordinate, theta = arctan( R/z )

        Return:
            theta: numpy.ndarray
                in the unit of degree
        """
        
        coord = self.read_coord('xh')
        Rh, zh = np.meshgrid(coord, coord)
        theta = np.arctan(Rh/zh)*u.rad.to(u.deg)
        return theta


    def get_radius(self, var):
        """get distance from origion

        In the cylindrical symmetric coordinate, this distance from the origin r^2 = R^2 + z^2

        Args:
            var: str
                grid boundary 'x' or grid center 'xh'.
        Return:
            rh: numpy.ndarray
                in the unit of kpc
        """

        coord = self.read_coord(var)
        Rh, zh = np.meshgrid(coord, coord)
        rh = np.sqrt(Rh*Rh+zh*zh)

        return rh

        
    def read_var(self, var, kprint):
        """read '*ascii.out*' variable outputs.

        Transfrom the variable output to an array according to the index.

        Args:
            var: str
                the variable name of output, the prefix of variable output file to read.
            kprint: int
                the kprint of output. 0 for initial value.

        Returns:
            data: numpy.ndarray
                data[0,0] corresponds to [zmax, 0], data[-1,-1] corresponds to [0, rmax]

        Example:
            >>> data = FermiData('./data/fermi/')
            >>> data.read_var('den', 1)
        """

        mylog.info(f"====> call method {sys._getframe().f_code.co_name}(var={var}, kprint={kprint})")
        if var == 'uz':
            var = 'ux'
        if var == 'ur':
            var = 'uy'

        if kprint == 0:
            filename = f"{var}atmascii.out"
        else:
            filename = f"{var}ascii.out{kprint}"

        file = f"{self.dir_path}/{filename}"

        data = np.fromfile(file,dtype=float,sep=" ")
        dmax = data.max()
        dmin = data.min()
        data = data.reshape([self.izone,self.izone])
        data = data.T  # reverse index from fortran
        mylog.info(PROP.format("(min, max)",f"({dmin}, {dmax})"))

        return data


    def read_hist(self, var, skiprows=2):
        """read '*c.out' history file.

        var
        Args:
            var : str
                the prefix of history file. For example: 'gasmass' for 'gasmassc.out'.

            skiprows : int
                number of lines to skip at the start of the file.

        Returns:
            df : pandas.DataFrame
        """

        path = f"{self.dir_path}/{var}c.out"
        df = pd.read_csv(path,skiprows=skiprows,delim_whitespace=True,index_col='tyr')
        mylog.info(f"====> call method {sys._getframe().f_code.co_name}(var={var}, skiprows={skiprows})")

        return df


def meshgrid(coord,rrange,zrange):
    """construct mesh grid based on coordinate.

    Mirror the coordinate horizontally, then construct meshgrid for matplotlib.pcolormesh

    Args:
        zrange : float
            the outer boundary of z.
        rrange : float
            the outer boundary of R.

    Returns:
        R, z: numpy.ndarray, numpy.ndarray

    Example:
        >>> data = FermiData(dirpath='./data/fermi/')
        >>> xh = data.read_coord('xh')
        >>> meshgrid(xh, 100, 100)
    """

    mylog.info(f"====> call function {sys._getframe().f_code.co_name}(data, rrange={rrange}, zrange={zrange})")
    z = coord[np.where(coord<=zrange)]
    R = coord[np.where(coord<=rrange)]
    RR = np.hstack((-R[::-1],R))
    R,z = np.meshgrid(RR,z)
    mylog.info(DIME.format('xh', f'z-{R.shape[0]} R-{R.shape[1]}'))

    return R, z


def mesh_var(data, meshgrid, flip=False):
    """construct meshgrid based on variable data.

    based on the data read by FermiData.read_var(), further constructing variable output
    to be available for matplotlib.pyplot.pcolormesh().

    Args:
        data : numpy.ndarray
            the numpy.ndarray from FermiData.read_var(var,kprint).
        meshgrid : numpy.ndarray
            the numpy.ndarray from FermiData.meshgrid(var,kprint). used to constrain the shape of var array.
        flip : bool
            left part and right part of the meshgrid are symmetric.

    Returns:
        mesh: numpy.ndarray

    Example:
        >>> data = FermiData(dirpath='./data/fermi/')
        >>> xh = data.read_coord('xh')
        >>> Rh, zh = meshgrid(xh, 100, 100)
        >>> den1 = data.read_var('den', 1)
        >>> den1 = mesh_var(den1, xh)
    """

    zrange = meshgrid.shape[0]
    rrange = int(meshgrid.shape[1]/2)

    mylog.info(f"====> calling function [mesh_var] {var}: construct the region within {meshgrid.max()} kpc.")

    meshr = data[:zrange,:rrange]

    # the horizontal mirror of r-velocity should be in opposite direction.
    if flip:
        meshl = -np.fliplr(meshr)
    else:
        meshl = np.fliplr(meshr)

    mesh = np.hstack((meshl,meshr))

    mylog.info(DIME.format(f'{var}', f'z-{mesh.shape[0]}, R-{mesh.shape[1]}'))

    return mesh


def slice_mesh(data, coord, direction='z', offset=0):
    """slice meshgrid array.

    extract data at given direction and given distance.

    Args:
        data : numpy.ndarray
            the numpy.ndarray from FermiData.read_var(var,kprint).
        coord : numpy.ndarray
            the numpy.ndarray from FermiData.read_coord(var).
        direction : str
            the direction of slice. The value can be 'z' or 'r'. Default is 'z'.
        offset : int or float
            distance to the axis in unit of 'kpc'. Default is 0.

    Returns:
        data: numpy.ndarray

    Example:
        >>> data = FermiData(dirpath='./data/fermi/')
        >>> xh = data.read_coord('xh)
        >>> den1 = data.read_var('den', 1)
        >>> den1_slice = slice_mesh(den1, xh, direction='z', offset=0)
    """

    nu = find_nearst(coord, offset)

    n_constant = 5.155e23  # num_den_electron = den * n_constant

    if direction == 'z':
        data = data[:,nu]
        return data
    elif direction == 'R':
        data = data[nu,:]
        return data
    else:
        raise ValueError("Only 'z' and 'R' are allowed.")


def cumsum(data, den, weight='mass', direction='r', interval=1):
    """ Return the cumulative sum of the elements along a given direction.

    Args:
        coord: numpy.ndarray
            the numpy.ndarray from FermiData.read_coord(var).
        den: numpy.ndarray
            density of girds. the numpy.ndarray from FermiData.read_var('den',kprint).
        weight: str
            sum up 'mass' or 'volume'. Default is 'mass'.
        direction: str
            sum up along which direction. The value can be 'z', 'R', or 'r'. Default is 'r'.
        interval: int
            interval for sampling coordinate in given direction. Default is 5.

    Returns:
        sumup: numpy.ndarray

    Example:
        >>> data = FermiData(dirpath='./data/fermi/')
        >>> den4 = data.read_var('den', 4)
        >>> summass = cumsum(data, den4)
    """

    dvol = data.get_dvolume()
    x = data.read_coord('x')
    xh = data.read_coord('xh')
    rh, zh = np.meshgrid(xh,xh)

    if weight == 'volume':
        mcell = dvol
    elif weight == 'mass':
        mcell = den*dvol

    if direction == 'z':
        dirtn = zh
    elif direction == 'R':
        dirtn = rh
    elif direction == 'r':
        rh = data.get_radius('xh')
        dirtn = rh
        
    bins = np.arange(1,x.max()+0.01,interval)
    sumup = [mcell[dirtn<=loc].sum() for loc in bins]

    return np.ndarray(sumup)

    
def average(data, var, interval=1, weights=None):
    """average the properties of the cells within the same spherical shell

    given data and array of a certain property, the whole region will be divided into spherical shells. 
    A certain property of gas within each shell will be averaged or weighted averaged, if necessary.

    Returns:
        bins: numpy.ndarray
            shell boundary coordinate, in the unit of kpc.
        avevar: the averaged or weighted averaged value of the property for each shell.

    Examples:
        >>> data = FermiData(dirpath='./data/fermi/')
        >>> den = data.read_var('den',5)
        >>> e = data.read_var('e',5)
        >>> p = 0.6667*e
        >>> temp = eqs.eos(P=p, rho=den)
        >>> z = data.read_var('z',5)
        >>> lbd = eqs.radcool(temp, z)
        >>> dvol = data.get_dvolume()
        >>> rad = (den/cons.m_p.cgs.value)**2 * dvol * lbd
        >>> bins, avetemp = average(xh, temp5, weights=rad)
    """
    
    xh = data.read_coord('xh')
    x = data.read_coord('x')
    Rh, zh = np.meshgrid(xh, xh)
    rh = np.sqrt(Rh*Rh+zh*zh)

    if weights is not None and weights.shape != var.shape:
        mylog.critical(f"The shape of weights {weights.shape} do not match the shape of variables {var.shape}.")
        raise IndexError(f"The shape of weights {weights.shape} do not match the shape of variables {var.shape}.")

    rhflt = rh.flatten()
    varflt = var.flatten()

    # sort all cells into spherical shells according to given radial interval. 
    bins = np.arange(0, x.max(), interval)
    indic = np.digitize(rhflt,bins)

    # average the cell in each cell.
    avevar = np.zeros_like(bins)
    for i in np.arange(1,bins.size+1):
        varmsk = varflt[indic==i]
        if weights is not None:
            wflt = weights.flatten()
            wts = wflt[indic==i]
            avevar[i-1] = np.average(varmsk, weights=wts)
        else:
            avevar[i-1] = np.average(varmsk)
        
    return bins, avevar


class Image(object):
    """Class for quickly demonstrating data results.

    Parameters:
    -----------
    figsize : tuple. Size of the figure presented. Default: (10,8)
    loc : str. Path of the data located. Default: './'
    dest : str. Destination for saving the figure. None means not saving but prompt a window to preview. Default: None

    """
    def __init__(self, figsize=(10,8), loc='./', dest=None):
        self.figsize = figsize
        self.loc = loc
        self.dest = dest

    def sliceplot(self, var, kprint, dir='z', offset=0, xlim=None, ylim=None, xlog=False, ylog=True):
        """ Show the 1D profile.
    
        Parameters:
        -----------
        var : str. Prefix of the variable output file.
        kprint : int. Index of the output file.
        dir : str. The direction of the profile. Default: 'z'
        offset : int or float. The distance away from the another axis. Default: 0
        xlim : tuple. Default: None.
        ylim : tuple. Default: None.
        xlog : bool. Default: False.
        ylog : bool. Default: True.
        """
    
        data = FermiData(dirpath=self.loc)
        if var in varlist:
            varr = data.read_var(var,kprint)
        elif var=='temp':
            den = data.read_var('den',kprint)
            press = data.read_var('e',kprint) * (gamma-1)
            varr = eqs.eos(P=press, rho=den)
        elif var=='entr':
            den = data.read_var('den',kprint)
            press = data.read_var('e',kprint) * (gamma-1)
            temp = eqs.eos(P=press, rho=den)
            nue = den/qmue/mp
            varr = temp/nue**(2./3.)

        xh = data.read_coord('xh')
        profile = slice_mesh(varr, xh, direction=dir, offset=offset)
    
        plt.plot(xh, profile)
        if xlim: plt.xlim(min(xlim),max(xlim))
        if ylim: plt.ylim(min(ylim),max(ylim))
        if xlog: plt.xscale('log')
        if ylog: plt.yscale('log')

        if(variables.get(var)): plt.ylabel(variables.get(var))
        if(direction.get(dir)): plt.xlabel(direction.get(dir))
    
        if self.dest:
            path = data.dir_path
            model = path.split('/')[-1]
            plt.savefig(f'{self.dest}sp-{var}{kprint}-{model}.jpg', dpi=300, bbox_inches='tight', pad_inches=0.02)
        else:
            plt.show()


    def display(self, var, kprint, value=None, region=(100,100), cmap='jet', nolog=False, notitle=False, flip=False):
        """ Display a 2D data using the matplotlib's pcolormesh

        Except for the 2D data output by the code, 2D temperature and entropy can be
        calculated and displayed, so far.

        Parameters:
        -----------
        var : str. Prefix of the variable output file.
        kprint : int. Index of the output file.
        value : tuple. Value range to display. If not specify, it would be the range of 2D data.
        region : tuple. The outer limit of plotting region, R-axis first. Default: (100,100)
        cmap : str. Default: 'jet'
        nolog : Bool. Default: False
        notitle : Bool. Default: False
        flip : Bool. Set True only for v_R. Default: False
        """

        rrange, zrange = region

        data = FermiData(dirpath=self.loc)
        times = data.kprint
        xh = data.read_coord('xh')
        R,z = meshgrid(xh,rrange,zrange)
        if var in varlist:
            varr = data.read_var(var,kprint)
        elif var=='temp':
            den = data.read_var('den',kprint)
            press = data.read_var('e',kprint) * (gamma-1)
            varr = eqs.eos(P=press, rho=den)
        elif var=='entr':
            den = data.read_var('den',kprint)
            press = data.read_var('e',kprint) * (gamma-1)
            temp = eqs.eos(P=press, rho=den)
            nue = den/qmue/mp
            varr = temp/nue**(2./3.)

        varr = mesh_var(den, R, flip=flip)
        if not nolog:
            varr = np.log10(varr+1e-99)

        fig = plt.figure(figsize=self.figsize)

        if value:
            pcm = plt.pcolormesh(R,z,den,cmap=cmap,vmax=max(value),vmin=min(value))
        else:
            pcm = plt.pcolormesh(R,z,den,cmap=cmap)

        # Set the attributes of plots
        plt.plot([],[], alpha=0, label=f"$t = {times[int(kprint)]}$ yr")
        plt.legend(frameon=False)

        plt.ylabel(r"$z$ (kpc)")
        plt.xlabel(r"$R$ (kpc)")

        path = data.dir_path
        model = path.split('/')[-1]
        if not notitle:
            plt.title(model)
        
        # Set the aspect ratio
        ax = plt.gca()
        ax.set_aspect('equal')

        # Add a new axes beside the plot to present colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="3%", pad=0.0)
        cb = plt.colorbar(pcm, cax=cax,orientation='horizontal')
        # Set the tick above colorbar
        cax.xaxis.tick_top()
        cax.xaxis.offsetText.set_x((1.05))

        if self.dest:
            plt.savefig(f'{self.dest}dsp-{var}{kprint}-{model}.jpg', dpi=300, bbox_inches='tight', pad_inches=0.02)
        else:
            plt.show()


    def averaged(self, var, kprint, interval=0.1, xlog=False, ylog=True, xlim=None, ylim=None):
        data = FermiData(dirpath=self.dir)

        dvol = data.get_dvolume()
        times = data.kprint

        if var in varlist:
            varr = data.read_var(var,kprint)
            # Calculate weights
            e = data.read_var('e',kprint)
            den = data.read_var('den',kprint)
            z = data.read_var('z',kprint)
            p = (gamma-1)*e
            temp = eqs.eos(P=p,rho=den)
            lbda = eqs.radcool(temp,z)
            rad = (den/mp)**2*dvol*lbda
        elif var=='temp':
            den = data.read_var('den',kprint)
            press = data.read_var('e',kprint) * (gamma-1)
            temp = eqs.eos(P=press, rho=den)
            varr = temp
            # Calculate weights
            z = data.read_var('z',kprint)
            lbda = eqs.radcool(temp,z)
            rad = (den/mp)**2*dvol*lbda
        else:
            raise ValueError(f"Variable {var} is not supported yet.")

        bins, avevar = fm.average(data, varr, interval=interval, weights=rad)
        plt.plot(bins,avevar, label=f"t={times[kprint]} yr")

        if xlim: plt.xlim(min(xlim),max(xlim))
        if xlog: plt.xscale('log')
        plt.xlabel('Radius (kpc)')
        if ylim: plt.ylim(min(ylim),max(ylim))
        if ylog: plt.yscale('log')
        plt.ylabel('Temperature (K)')
        plt.legend(frameon=False)

        if self.dest:
            path = data.dir_path
            model = path.split('/')[-1]
            plt.savefig(f'{self.dest}ave-{var}{kprint}-{model}.jpg', dpi=300, bbox_inches='tight', pad_inches=0.02)
        else:
            plt.show()



def find_nearst(arr,target):
    """get the index of nearest value

    Given a number, find out the index of the nearest element in an 1D array.

    Args:
        arr: array for searching
        target: target number
    """

    index = np.abs(arr-target).argmin()
    return index

def latex_float(f):
    float_str = "{0:.2g}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str


if __name__ == "__main__":
	fire.Fire(Image)