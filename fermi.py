import numpy as np
import pandas as pd
import os

class FermiData:
    '''
    Tools for reading output data of fermi.f

    Parameters
    ----------
    dirpath : str, optional
        Directory path to be loaded. Default is './', which assumes the you work in
        current directory.

    Attributes
    ----------
    dir_path : str
        Path to directory form which the data is read.
    iezone : int
        the number of even spaced grids.
    ilzone : int
        the numver of logarithmic spaced grids.
    ezone : float
        the length of even spaced region.
    zone : int
        the number of total grids in one direction. (two directions are the same.)
    reso : float
        the resolution for the even spaced grids.
    kprint : list
        list of time (year) for output.

    Methods
    ----------
    read_xh : numpy.ndarray
        read 'xhascii.out' file.
    read_var : numpy.ndarray
        read '*ascii.out*' meshgrid file.
    read_hist : pandas.core.frame.DataFrame
        read '*c.out' history file.
    '''

    def __init__(self, dirpath='./'):
        os.chdir(dirpath)
        self.dir_path = os.getcwd()

        with open(self.dir_path+'/fermi.inp','r') as f:
            text = f.readlines()
            dims = text[2].split()
            self.iezone = int(dims[0])
            self.ilzone = int(dims[1])
            self.ezone = float(dims[2])
            self.zone = self.iezone + self.ilzone
            self.reso = self.ezone/self.iezone
            self.kprint = text[20].split()[:-1]


    #read coordinates
    def read_xh(self):
        cmkpc=3.08e21
        filename = self.dir_path+'/xhascii.out'

        print("calling function [read_xh]")
        xh = np.fromfile(filename,sep=" ")
        print("====> xh shape:",xh.shape,"\n")
        return xh/cmkpc


    def read_var(self, var, kprint):
        '''
        read meshgrid file.

        Parameter
        ---------
        var : string
            the variable of output. It is the same as that in simulation, i.e. 'den', 'e', 'ecr', 'uz', 'ur'.

        kprint : int
            the kprint of output.

        Return
        ------
        numpy.ndarray
        '''

        if var == 'uz':
            var = 'ux'
        if var == 'ur':
            var = 'uy'

        if kprint == 0:
            filename = var + 'atmascii.out'
        else:
            filename = var + 'ascii.out' + str(kprint)

        if filename not in os.listdir(self.dir_path):
            raise KeyError('There is no file named "'+filename+'" in this directory.')

        print("calling function [read_var]: ",var,", kprint=",kprint)
        data = np.fromfile(filename,dtype=float,sep=" ")
        dmax = data.max()
        dmin = data.min()
        data = data.reshape([self.zone,self.zone])
        data = data.T  # reverse index from fortran
        print("====> ",var,kprint," shape:",data.shape)
        print("====> max:",dmax,' min:',dmin,"\n")
        return data


    def read_hist(self, var):
        '''
        read '*c.out' history file.

        Parameter
        ---------
        var : string
            the variable of history. The value can be 'energy', 'gasmass'.

        Return
        ------
        pandas.DataFrame
        '''

        if(var == 'energy'):
            path = self.dir_path+'/energyc.out'
            return pd.read_csv(path,skiprows=6,delim_whitespace=True,index_col='tyr')
        elif(var == 'gasmass'):
            path = self.dir_path+'/gasmassc.out'
            return pd.read_csv(path,skiprows=2,delim_whitespace=True,index_col='tyr')
        else:
            raise KeyError('Unvailable name, you should check the name or add this option to the module.')


def meshgrid(data,rrange,zrange):
    '''
    construct mesh grid of spatial coordinations.

    Parameter
    ---------
    zrange : float
        the range of z.

    rrange : float
        the range of R.

    Return
    ------
    numpy.ndarray tuple (R,z)
    '''

    z = data[np.where(data<=zrange)]
    R = data[np.where(data<=rrange)]
    RR = np.hstack((-R[::-1],R))
    R,z = np.meshgrid(RR,z)
    print('====> mesh region: [',z.max(),R.max(),'] kpc')
    print('====> xh_mesh shape: ','z-',R.shape[0],', R-',R.shape[1],'\n')

    return R,z

def mesh_var(data, var, meshgrid):
    '''
    generate meshgrid.

    Parameter
    ---------
    data : numpy.ndarray
        the numpy.ndarray from FermiData.read_var(var,kprint).

    var : string
        the variable of output. It is the same as that in simulation, i.e. 'den', 'e', 'ecr'.

    meshgrid : numpy.ndarray
        the numpy.ndarray from FermiData.meshgrid(var,kprint). be used to constrain the shape of var array.

    Return
    ------
    numpy.ndarray
    '''

    zrange = meshgrid.shape[0]
    rrange = int(meshgrid.shape[1]/2)

    print("calling function [mesh_var] (",var,"): construct the region within ",meshgrid.max()," kpc.")

    meshr = data[:zrange,:rrange]

    if(var == 'ur'):
        meshl = -np.fliplr(meshr)
    else:
        meshl = np.fliplr(meshr)

    mesh = np.hstack((meshl,meshr))

    print('====> ',var,' shape:','z-',mesh.shape[0],' ,R-',mesh.shape[1],'\n')

    return mesh


def slice_mesh(data, coord, direction='z', kpc=0):
    '''
    slice meshgrid array.

    Parameter
    ---------
    var : string
        the variable of output. It is the same as that in simulation, i.e. 'den', 'e', 'ecr', 'uz', 'ur'.

    coord : numpy.ndarray
        the numpy.ndarray from FermiData.read_xh().

    direction : string
        the direction of slice. The value can be 'z' or 'r'. Default is 'z'.

    kpc : int
        distance to the axis in unit of 'kpc'. Default is 0.

    Return
    ------
    1-D numpy.ndarray
    '''
    nu = find_nearst(coord, kpc)

    n_constant = 5.155e23  # num_den_electron = den * n_constant

    if direction == 'z':
        return data[nu,:]
    elif direction == 'r':
        return data[:,nu]
    else:
        raise KeyError("Only 'z' and 'r' are allowed.")

def find_nearst(arr,target):
    index = np.abs(arr-target).argmin()
    return arr[index]
