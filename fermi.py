#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fermi.py - tools for analyzing outputs of fermi.f
Last Modified: 2020.05.01

Copyright(C) 2020 Shaokun Xie <https://xshaokun.com>
Licensed under the MIT License, see LICENSE file for details
"""


import os
import sys
import numpy as np
import pandas as pd
from astropy import units as u
from astropy import constants as cons
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from skpy.utilities.logger import fmLogger as mylog

PARA = "Parameter: {0:<20}\t = {1:<10}"
PROP = "Property: {0:<20}\t = {1:<10}"
DIME = "Dimension: {0:<20}\t : {1:<10}"


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
            self.inp = f.readlines()
            self.bc = self.inp[0].split()[:4]
            dims = self.inp[2].split()
            self.iezone = int(dims[0])
            self.ilzone = int(dims[1])
            self.ezone = float(dims[2])
            self.izone = self.iezone + self.ilzone
            self.reso = self.ezone/self.iezone
            self.kprint = self.inp[20].split()[:-1]

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
        line = self.inp[row].split()
        para = line[col]

        return para


    def read_coord(self, var):
        """read coordination file.

        Args:
            var: str
                the prefix of coordination file.

        Returns:
            x: numpy.ndarray
                1D coordination in the unit of kpc.

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
                the variable name of output, the prefix of variable file read.
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


    def read_hist(self, var):
        """read '*c.out' history file.

        Args:
            var : string
                the variable of history. For example: 'gasmass' for 'gasmassc.out'.

        Returns:
            pandas.DataFrame
        """

        if(var == 'energy'):
            path = self.dir_path+'/energyc.out'
            return pd.read_csv(path,skiprows=6,delim_whitespace=True,index_col='tyr')
        elif(var == 'gasmass'):
            path = self.dir_path+'/gasmassc.out'
            return pd.read_csv(path,skiprows=2,delim_whitespace=True,index_col='tyr')
        else:
            raise ValueError('Unvailable name, you should check the name or add this option to the module.')


def meshgrid(coord,rrange,zrange):
    """construct mesh grid based on coordination.

    Mirror the coordination horizontally

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


def mesh_var(data, var, meshgrid):
    """construct meshgrid based on variable data.

    based on the data read by FermiData.read_var(), further constructing variable output
    to be available for matplotlib.pyplot.pcolormesh().

    Args:
        data : numpy.ndarray
            the numpy.ndarray from FermiData.read_var(var,kprint).
        var : string
            the variable name.
        meshgrid : numpy.ndarray
            the numpy.ndarray from FermiData.meshgrid(var,kprint). used to constrain the shape of var array.

    Returns:
        mesh: numpy.ndarray

    Example:
        >>> data = FermiData(dirpath='./data/fermi/')
        >>> xh = data.read_coord('xh')
        >>> Rh, zh = meshgrid(xh, 100, 100)
        >>> den1 = data.read_var('den', 1)
        >>> den1 = mesh_var(den1, 'den', xh)
    """

    zrange = meshgrid.shape[0]
    rrange = int(meshgrid.shape[1]/2)

    mylog.info(f"====> calling function [mesh_var] {var}: construct the region within {meshgrid.max()} kpc.")

    meshr = data[:zrange,:rrange]

    # the horizontal mirror of r-velocity should be in opposite direction.
    if(var == 'ur'):
        meshl = -np.fliplr(meshr)
    else:
        meshl = np.fliplr(meshr)

    mesh = np.hstack((meshl,meshr))

    mylog.info(DIME.format(f'{var}', f'z-{mesh.shape[0]}, R-{mesh.shape[1]}'))

    return mesh


def slice_mesh(data, coord, direction='z', kpc=0):
    """slice meshgrid array.

    extract data at given direction and given distance.

    Args:
        data: numpy.ndarray
            the numpy.ndarray from FermiData.read_var(var,kprint).
        coord : numpy.ndarray
            the numpy.ndarray from FermiData.read_coord(var).
        direction : string
            the direction of slice. The value can be 'z' or 'r'. Default is 'z'.
        kpc : int
            distance to the axis in unit of 'kpc'. Default is 0.

    Returns:
        data: numpy.ndarray

    Example:
        >>> data = FermiData(dirpath='./data/fermi/')
        >>> xh = data.read_coord('xh)
        >>> den1 = data.read_var('den', 1)
        >>> den1_slice = slice_mesh(den1, xh, direction='z', kpc=0)
    """

    nu = find_nearst(coord, kpc)

    n_constant = 5.155e23  # num_den_electron = den * n_constant

    if direction == 'z':
        data = data[:,nu]
        return data
    elif direction == 'r':
        data = data[nu,:]
        return data
    else:
        raise ValueError("Only 'z' and 'r' are allowed.")


def count_enclosed(coord, den, weight='mass', direction='r', interval=1):
    """ sum up the grids inside out.

    given a direction, sum up mass or volume of the grids inside out and output a profile.

    Args:
        coord: numpy.ndarray
            the numpy.ndarray from FermiData.read_coord(var).
        den: numpy.ndarray
            density of girds. the numpy.ndarray from FermiData.read_var('den',kprint).
        weight: string
            sum up 'mass' or 'volume'. Default is 'mass'.
        direction: string
            sum up along which direction. The value can be 'z', 'R', or 'r'. Default is 'r'.
        interval: int
            interval for sampling coordination in direction. Default is 5.

    Returns:
        sumup: numpy.ndarray

    Example:
        >>> data = FermiData(dirpath='./data/fermi/')
        >>> x = data.read_coord('x')
        >>> den4 = data.read_var('den', 4)
        >>> summass = count_enclosed(x, den4)
    """

    coord = coord*u.kpc.to(u.cm)
    rr, zz = np.meshgrid(coord, coord)
    dz = np.diff(zz, axis=0)[:,:-1]
    dr2 = np.diff(rr*rr)[:-1]
    dvol = np.pi*dr2*dz

    if weight == 'volume':
        mcell = dvol
    elif weight == 'mass':
        mcell = den*dvol

    if direction == 'z':
        dirtn = zz
    elif direction == 'R':
        dirtn = rr
    elif direction == 'r':
        rad = np.sqrt(zz*zz+rr*rr)
        dirtn = rad
        
    sumup = [mcell[dirtn[1::,1::]<=loc].sum() for loc in coord[1::interval]]

    return np.ndarray(sumup)

    
def average(coord, var, interval=1, weights=None):
    """average the properties of the cells within the same spherical shell

    

    """
    
    Rh, zh = np.meshgrid(coord, coord)
    rh = np.sqrt(Rh*Rh+zh*zh)

    if not rh.shape == var.shape:
        mylog.error(f"Coordinate should be grid-centric.")

    if weights is not None and weights.shape != var.shape:
        mylog.error(f"The shape of weights {weights.shape} do not match the shape of variables {var.shape}.")

    rhflt = rh.flatten()
    varflt = var.flatten()

    bins = np.arange(0, coord.max(), interval)
    indic = np.digitize(rhflt,bins)
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



# class FermiImage(Figure):
    # """ Plot 1D profile
# 
    # """
    # def __init__(self, figure_size, fontsize):
        # super(FermiImage, self).__init__(figsize=figure_size,)


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