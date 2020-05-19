#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
fermi.py - tools for analyzing outputs of fermi.f
Last Modified: 2020.05.01

Copyright(C) 2020 Shaokun Xie <https://xshaokun.com>
Licensed under the MIT License, see LICENSE file for details
"""


import numpy as np
import pandas as pd
import os

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
        zone: int
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

        with open(self.dir_path+'/fermi.inp','r') as f:
            text = f.readlines()
            dims = text[2].split()
            self.iezone = int(dims[0])
            self.ilzone = int(dims[1])
            self.ezone = float(dims[2])
            self.zone = self.iezone + self.ilzone
            self.reso = self.ezone/self.iezone
            self.kprint = text[20].split()[:-1]


    def read_xh(self):
        """read 'xhascii.out' file. 

        Returns:
            xh: numpy.ndarray
                1D coordination in the unit of kpc.

        Example:
            >>> data = FermiData(dirpath='./data/fermi/')
            >>> data.read_xh()
        """


        cmkpc=3.08e21
        filename = self.dir_path+'/xhascii.out'

        print(f"====> call method {__name__}()")
        xh = np.fromfile(filename,sep=" ")
        print(f"    shape: {xh.shape}")
        xh = xh/cmkpc
        return xh


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

        print(f"====> call method {__name__}(var= {var}, kprint={kprint})")

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
        data = data.reshape([self.zone,self.zone])
        data = data.T  # reverse index from fortran
        print(f"    shape: {data.shape}")
        print(f"    max: {dmax}, min: {dmin}")
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
        >>> xh = data.read_xh()
        >>> meshgrid(xh, 100, 100)
    """

    print(f"====> call function {__name__}(data, rrange={rrange}, zrange={zrange})")
    z = coord[np.where(coord<=zrange)]
    R = coord[np.where(coord<=rrange)]
    RR = np.hstack((-R[::-1],R))
    R,z = np.meshgrid(RR,z)
    print(f'    mesh region: [{z.max()},{R.max()}] kpc')
    print(f'    xh_mesh shape: z-{R.shape[0]} R-{R.shape[1]}')

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
        >>> xh = data.read_xh()
        >>> xh = meshgrid(xh, 100, 100)
        >>> den1 = data.read_var('den', 1)
        >>> den1 = mesh_var(den1, 'den', xh)
    """

    zrange = meshgrid.shape[0]
    rrange = int(meshgrid.shape[1]/2)

    print(f"====> calling function [mesh_var] {var}: construct the region within {meshgrid.max()} kpc.")

    meshr = data[:zrange,:rrange]

    # the horizontal mirror of r-velocity should be in opposite direction.
    if(var == 'ur'):
        meshl = -np.fliplr(meshr)
    else:
        meshl = np.fliplr(meshr)

    mesh = np.hstack((meshl,meshr))

    print(f'{var} shape: z-{mesh.shape[0]}, R-{mesh.shape[1]}')

    return mesh


def slice_mesh(data, coord, direction='z', kpc=0):
    """slice meshgrid array.

    extract data at given direction and given distance.
    
    Args:
        data: numpy.ndarray
            the numpy.ndarray from FermiData.read_var(var,kprint).
        coord : numpy.ndarray
            the numpy.ndarray from FermiData.read_xh().
        direction : string
            the direction of slice. The value can be 'z' or 'r'. Default is 'z'.
        kpc : int
            distance to the axis in unit of 'kpc'. Default is 0.

    Returns:
        data: numpy.ndarray

    Example:
        >>> data = FermiData(dirpath='./data/fermi/')
        >>> xh = data.read_xh()
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

def find_nearst(arr,target):
    """get the index of nearest value

    Given a number, find out the index of the nearest element in an 1D array.

    Args:
        arr: array for searching
        target: target number
    """


    index = np.abs(arr-target).argmin()
    return index


if __name__ == "__main__":
    import doctest
    doctest.testmod()