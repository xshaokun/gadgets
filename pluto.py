#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
pluto.py - tools for analyzing outputs from PLUTO
Last Modified: 2021.09.05

Copyright(C) 2021 Shaokun Xie <https://xshaokun.com>
Licensed under the MIT License, see LICENSE file for details
'''

import os
import fire
import numpy as np
import pandas as pd
import pyPLUTO.pload as pp
import matplotlib as mpl
import matplotlib.pyplot as plt
from skpy.utilities.tools import nearest
from astropy import units as u
from mpl_toolkits.axes_grid1 import make_axes_locatable


class PlutoDefConstants(object):
  ''' Physical constants in c.g.s units defined in $PLUTO_DIR/Src/pluto.h '''

  __pluto_def_constants = {
    'CONST_AH'  : 1.008     ,
    'CONST_AHe'   : 4.004     ,
    'CONST_AZ'  : 30.0      ,
    'CONST_amu'   : 1.66053886e-24  ,
    'CONST_au'  : 1.49597892e13   ,
    'CONST_c'   : 2.99792458e10   ,
    'CONST_e'   : 4.80320425e-10  ,
    'CONST_eV'  : 1.602176463158e-12,
    'CONST_G'   : 6.6726e-8     ,
    'CONST_h'   : 6.62606876e-27  ,
    'CONST_kB'  : 1.3806505e-16   ,
    'CONST_ly'  : 0.9461e18     ,
    'CONST_mp'  : 1.67262171e-24  ,
    'CONST_mn'  : 1.67492728e-24  ,
    'CONST_me'  : 9.1093826e-28   ,
    'CONST_mH'  : 1.6733e-24    ,
    'CONST_Msun'  : 2.e33       ,
    'CONST_Mearth': 5.9736e27     ,
    'CONST_NA'  : 6.0221367e23    ,
    'CONST_pc'  : 3.0856775807e18   ,
    'CONST_PI'  : 3.14159265358979  ,
    'CONST_Rearth': 6.378136e8    ,
    'CONST_Rgas'  : 8.3144598e7     ,
    'CONST_Rsun'  : 6.96e10       ,
    'CONST_sigma' : 5.67051e-5    ,
    'CONST_sigmaT': 6.6524e-25
  }

  def __init__(self):
    for key, value in self.__pluto_def_constants.items():
      setattr(self, key, value)


class PlutoFluidInfo(object):
  ''' Contain pre-defined fluid information in PLUTO '''

  known_fields = {
# Variable name :  (code_unit,             astro_unit,  dimension   , alias      )
# -----------------------------------------------------------------------------
    'rho'  :('code_density',          'g/cm**3',   'density'   , ['density'   ]),
    'vx1'  :('code_velocity',           'km/s',    'speed'     , ['velocity-1'  ]),
    'vx2'  :('code_velocity',           'km/s',    'speed'     , ['velocity-2'  ]),
    'vx3'  :('code_velocity',           'km/s',    'speed'     , ['velocity-3'  ]),
    'prs'  :('code_density*code_velocity**2',   'erg/cm**3',   'pressure'  , ['pressure'  ]),
    'speed':('code_velocity',           'km/s',    'speed'     , ['speed'     ]),
    'temp' :('u.K',               'K',       'temperature' , ['temperature' ]),
    'mass' :('code_density*code_length**3',   'Msun',    'mass'    , ['mass'    ]),
    'acc'  :('code_veloctiy**2/code_length',  '',      'acceleration', ['acceleration'])
  }

  @classmethod
  def show(cls):
    print('Field Name \t Alias')
    print('-------------------------------')
    for key, value in cls.known_fields.items():
      print(f'{key:10s} \t {value[-1]}')

  # @classmethod
  # def add_fields(cls, name, code_unit=None, astro_unit=None, dimension=None):
    # lst = (code_unit, astro_unit, dimension)
    # cls.known_fields[name] = list


class Dataset(object):
  ''' Pluto data work directory

  Args:
    w_dir (str): path to the directory where data files locate. Default is './'.
    datatype (str): type of data files. Default is 'vtk'.
    init_file (str): init file including the parameters for simulation. Default is 'pluto.ini'.

  Attributes:
    wdir (str): absolute path to the dirctory.
    init_file (str):
    datatype (str):
    filetype (str):
    endianess (str):
    geometry (str):
    ndim (int):
    code_unit (dict):
    field_list (list):
    derived_fields (list):

  Methods:
    info():
  '''

  __slots__=[
    'wdir',
    'init_file',
    'datatype',
    'filetype',
    'endianess',
    'geometry',
    'ndim',
    'code_unit',
    'field_list',
    'derived_fields',
    '__log_file',
    '__ds'
  ]

  def __init__(self, w_dir='./', datatype='vtk', init_file='pluto.ini'):
    self.wdir = os.path.abspath(w_dir) + '/'
    self.init_file = init_file
    self.datatype = datatype

    varfile = self.wdir+self.datatype+'.out'
    with open(varfile,'r') as vfp:
      varinfo = vfp.read().splitlines()
      lastline = varinfo[-1].split()
      self.filetype = lastline[4]
      self.endianess = lastline[5]
      self.field_list = lastline[6:]
    self.geometry = 'CARTESIAN'
    self.derived_fields = {}

    # Three base units and default values in pluto code
    self.code_unit={
      'code_density' : 1.0 * u.g/u.cm**3,
      'code_length'  : 1.0 * u.cm,
      'code_velocity': 1.0 * u.cm/u.s
    }

    if 'definitions.h' in os.listdir(self.wdir):
      with open(self.wdir+'definitions.h','r') as df:
        for line in df.readlines():
          if line.startswith('#define  GEOMETRY'):
            self.geometry = line.split()[-1]
          if line.startswith('#define  DIMENSION'):
            self.ndim = int(line.split()[-1])
          if line.startswith('#define  UNIT'):
            name = line.split()[1]
            expr = line.split()[-1]
            var = name.split('_')[-1]
            key = 'code_'+var.lower()
            if 'CONST' in expr:
              expr = expr.replace('CONST', 'PlutoDefConstants().CONST')
            self.code_unit[key] = eval(expr) * self.code_unit[key].unit
    else:
      print('Could not open definitions.h! \
        The values of attributes [geometry, code_units] did not update,\
        you can specifiy them later, \
        and assign units by in_code_unit() manually.')


  def __getitem__(self, index):
    # index: int or time
    ns = self._number_step(index)
    ds = Snapshot(ns, w_dir=self.wdir, datatype=self.datatype, init_file=self.init_file)
    return ds


  def info(self):
    for attr in self.__slots__:
      if hasattr(self, attr):
        print(f'{attr:15}:  {getattr(self, attr)}')


  def _number_step(self, ns):
    ''' find number step of data file

    Args:
      ns (int/float): should be a integer in default, \
          but if it is a negative integer, return the last number step or if it is a float, \
          it is assumed to be time, and return the nearst number step
    '''

    hdr = ['time','dt','Nstep']
    log_file = pd.read_table(self.wdir+self.datatype+'.out',sep=' ', usecols=[1,2,3], names=hdr)

    if type(ns) is int:
      if ns < 0:
        return log_file.index[-1]
      else:
        return ns
    elif type(ns) is float:     #  given a specific [time], find [ns] corresponding nearst existed data [time].
      return nearest(log_file['time'],ns)
    else:
      raise TypeError(f'ns({ns}) should be int or float, now it is {type(ns)}.')


class Snapshot(Dataset):
  ''' Pluto output snapshot data structure

  Args:
    ns (int/float): should be a integer in default, \
        but if it is a negative integer, return the last number step or if it is a float, \
        it is assumed to be time, and return the nearst number step
    Others refer to Dataset() class

  Attributes:
    Include all attributes of Dataset() class, besides:
    nstep: (int)
    time: (int/units.Quantity)
    dt: (float)
    is_quantity: (bool)
    index: (dict)
    coord: (dict)
    grid: (dict)
    fields: (dict)

  Methods:
    info() :
    in_code_unit():
    in_astro_unit():
    slice2d(field, x1=None, x2=None, x3=None):
    slice1d(field, x1=None, x2=None, x3=None):
    to_cart(field):
  '''

  __slots__= [
    'nstep',
    'time',
    'dt',
    'is_quantity',
    'index',
    'coord',
    'grid',
    'fields'
  ]

  def __init__(self, ns, w_dir, datatype,init_file):
    super().__init__(w_dir, datatype, init_file)

    ds = pp.pload(ns, w_dir=self.wdir, datatype=self.datatype)
    self.nstep = ds.NStep
    self.time = ds.SimTime
    self.dt = ds.Dt
    self.is_quantity = False

    grids_info = [
      'n1','n2','n3',         # number of computational cells
      'n1_tot','n2_tot','n3_tot',   # total cells including ghost cells
    ]
    self.index = {}
    for key in grids_info:
      self.index[key] = getattr(ds, key)

    coord_info = [
      'x1','x2','x3',   # cell center coordinate
      'dx1','dx2','dx3',  # cell width
      'x1r','x2r','x3r'  # cell edge coordinate
    ]
    self.coord = {}
    for key in coord_info:
      self.coord[key] = getattr(ds, key)

    self.grid = {}  # construct meshgrid
    self.grid['x1'], self.grid['x2'], self.grid['x3'] = np.meshgrid(self.coord['x1'], self.coord['x2'], self.coord['x3'])
    self.grid['dx1'], self.grid['dx2'], self.grid['dx3'] = np.meshgrid(self.coord['dx1'], self.coord['dx2'], self.coord['dx3'])
    self.grid['x1r'], self.grid['x2r'], self.grid['x3r'] = np.meshgrid(self.coord['x1r'], self.coord['x2r'], self.coord['x3r'])
    if self.ndim !=3:
      for key in self.grid:
        self.grid[key] = self.grid[key].squeeze()

    vars_info = {}
    for var in self.field_list:
      vars_info[var] = getattr(ds, var).T
    self.fields = vars_info

    # if 'definitions.h' in os.listdir(self.wdir):
      # self.in_code_unit()


  def __getitem__(self, key):
    return self.__fields_value[key]


  def info(self):
    for attr in self.__slots__:
      value = getattr(self, attr)
      if not type(value) is dict:
        print(f'{attr:15}:  {value}')


  def in_code_unit(self):
    ''' Assign code units '''

    code_density = u.def_unit('code_density', represents=self.code_unit['code_density'])
    code_length = u.def_unit('code_length', represents=self.code_unit['code_length'])
    code_velocity = u.def_unit('code_velocity', represents=self.code_unit['code_velocity'])
    # u.add_enabled_units([code_density, code_length, code_velocity])

    if self.is_quantity:  # in case units are already assigned
      for key in self.coord:
        if self.geometry == 'CARTESIAN':
          self.coord[key] = self.coord[key].to(code_length)
          self.grid[key] = self.grid[key].to(code_length)
        elif self.geometry == 'SPHERICAL':
          if '1' in key:
            self.coord[key] = self.coord[key].to(code_length)
            self.grid[key] = self.grid[key].to(code_length)
          else:
            self.coord[key] = self.coord[key].to(u.rad)
            self.grid[key] = self.grid[key].to(u.rad)
        elif self.geometry == 'POLAR':
          if '2' in key:
            self.coord[key] = self.coord[key].to(u.rad)
            self.grid[key] = self.grid[key].to(u.rad)
          else:
            self.coord[key] = self.coord[key].to(code_length)
            self.grid[key] = self.grid[key].to(code_length)

      pluto_fields = PlutoFluidInfo.known_fields
      for key in self.field_list:
        self.fields[key] = self.fields[key].to(eval(pluto_fields[key][0]))

      self.time = self.time.to(code_length/code_velocity)
      self.dt = self.dt.to(code_length/code_velocity)
    else:
      for key in self.coord:
        if self.geometry == 'CARTESIAN':
          self.coord[key] *= code_length
          self.grid[key] *= code_length
        elif self.geometry == 'SPHERICAL':
          if '1' in key:
            self.coord[key] *= code_length
            self.grid[key] *= code_length
          else:
            self.coord[key] *= u.rad
            self.grid[key] *= u.rad
        elif self.geometry == 'POLAR':
          if '2' in key:
            self.coord[key] *= u.rad
            self.grid[key] *= u.rad
          else:
            self.coord[key] *= code_length
            self.grid[key] *= code_length

      pluto_fields = PlutoFluidInfo.known_fields
      for key in self.field_list:
        self.fields[key] *= eval(pluto_fields[key][0])

      self.time *= code_length/code_velocity
      self.dt *= code_length/code_velocity

    self.is_quantity = True


  def in_astro_unit(self):
    ''' convert the units to those commonly used in astro '''

    if not self.is_quantity:  # in case not quantity, assign code_unit first
      self.in_code_unit()

    for key in self.coord:
      if self.geometry == 'CARTESIAN':
        self.coord[key] = self.coord[key].to(u.kpc)
      elif self.geometry == 'SPHERICAL':
        if '1' in key:
          self.coord[key] = self.coord[key].to(u.kpc)
          self.grid[key] = self.grid[key].to(u.kpc)
        else:
          self.coord[key] = self.coord[key].to(u.deg)
          self.grid[key] = self.grid[key].to(u.deg)
      elif self.geometry == 'POLAR':
        if '2' in key:
          self.coord[key] = self.coord[key].to(u.deg)
          self.grid[key] = self.grid[key].to(u.deg)
        else:
          self.coord[key] = self.coord[key].to(u.kpc)
          self.grid[key] = self.grid[key].to(u.kpc)

    pluto_fields = PlutoFluidInfo.known_fields
    for key in self.field_list:
      unit = u.Unit(pluto_fields[key][1])
      if unit == '':
        self.fields[key] = self.fields[key].cgs
      else:
        self.fields[key] = self.fields[key].to(u.Unit(pluto_fields[key][1]))

    self.time = self.time.to(u.yr)
    self.dt = self.dt.to(u.yr)


  def slice2d(self, field, x1=None, x2=None, x3=None):
    ''' Slice 3-D array and return 2-D array

    coordinate of one dimension should be specified

    Args:
      field (str): output variable name listed in attribute `field_list`
      x1 (float): rough coordinate (optional)
      x2 (float): rough coordinate (optional)
      x3 (float): rough coordinate (optional)

    Returns:
      arr (numpy.ndarray): in 2-D
    '''

    offset = [x1,x2,x3]
    i = 0
    for coord in offset:
      if coord is not None:  # find the direction
        dim = 'x'+str(i+1)
        if self.is_quantity:
          x = self.coord[dim].value
        else:
          x = self.coord[dim]
        offset[i] = nearest(x, coord)  # convert coord to index
        break
      i+=1
    index = str(offset).replace('None',':')
    arr = eval('self.fields[field]' + index)
    return arr


  def slice1d(self, field, x1=None, x2=None, x3=None):
    ''' Slice 3-D array and return 1-D array

    coordinates of two dimensions should be specified

    Args:
      field (str): output variable name listed in attribute `field_list`
      x1 (float): rough coordinate (optional)
      x2 (float): rough coordinate (optional)
      x3 (float): rough coordinate (optional)

    Returns:
      arr (numpy.ndarray): in 1-D
    '''

    if self.ndim == 3:
      offset = [x1,x2,x3]
    else:
      offset = [x1,x2]

    i = 0
    for coord in offset:
      dim = 'x'+str(i+1)
      if self.is_quantity:
        x = self.coord[dim].value
      else:
        x = self.coord[dim]

      if coord is not None:
        offset[i] = nearest(x, coord)
      else:
        offset[i] = None
      i+=1
    index = str(offset[::-1]).replace('None',':')
    arr = eval('self.fields[field]'+index)
    return arr


  @staticmethod
  def _from_sph_coord(r, theta, phi):
    ''' from spherical coordinate to cartesian coordinate '''
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.cos(phi)
    z = r * np.cos(theta)
    return x, y, z

  @staticmethod
  def _from_cyl_coord(r, phi, z):
    ''' from cylindrical coordinate to cartesian coordinate '''
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return x, y, z

  @staticmethod
  def _from_sph_vect(theta, phi, v_r, v_th, v_phi):
    ''' from vectors in spherical coordinate to those in cartesian coordinate '''
    v_x = v_r * np.sin(theta) * np.cos(phi) + v_th * np.cos(theta) * np.cos(phi) - v_phi * np.sin(phi)
    v_y = v_r * np.sin(theta) * np.sin(phi) + v_th * np.cos(theta) * np.sin(phi) + v_phi * np.cos(phi)
    v_z = v_r * np.cos(theta) - v_th * np.sin(theta)
    return v_x, v_y, v_z

  @staticmethod
  def _from_cyl_vect(phi, v_r, v_phi, v_z):
    ''' from vectors in cylindrical coordinate to those in cartesian coordinate '''
    v_x = v_r * np.cos(phi) - v_phi * np.sin(phi)
    v_y = v_r * np.sin(phi) + v_phi * np.cos(phi)
    return v_x, v_y, v_z


  def to_cart(self, field):
    ''' convert to cartesian coordinate system

    Args:
      field (str): 'grid' or 'velocity'

    Returns:
      tuple: the resulted three components
    '''

    x = []
    v = []
    for i in range(1,4):
      if self.index['n'+str(i)] == 1:
        xx = np.zeros_like(self.grid['x'+str(i)])
        x.append(xx)
      else:
        x.append(self.grid['x'+str(i)])
      v.append(self.fields['vx'+str(i)])

    if self.geometry == 'SPHERICAL':
      if field=='grid':
        x1, x2, x3 = self._from_sph_coord(x[0],x[1],x[2])
      elif field=='velocity':
        x1, x2, x3 = self._from_sph_vect(x[1],x[2],v[0],v[1],v[2])
    elif self.geometry == 'POLAR':
      if field=='grid':
        x1, x2, x3 = self._from_cyl_coord(x[0],x[1],x[2])
      elif field=='velocity':
        x1, x2, x3 = self._from_cyl_vect(x[1],v[0],v[1],v[2])
    else:
      raise KeyError('Only support geometry of [SPHERICAL] and [POLAR].')

    return x1, x2, x3


def to_cart(data):
  ''' convert to cartesian coordinate system

  Args:
    data (Snapshot): data needed to be converted geometry

  Returns:
    Snapshot: new Snapshot is as the same as the older one,
              except for grid and velocity arrays have been transformed to cartesian system
              (grid arrays only include cell-centered coordinates)
  '''

  ds = data
  ds.grid['x1'],  ds.grid['x2'], ds.grid['x3'] = data.to_cart('grid')
  ds.fields['vx1'],  ds.fields['vx2'], ds.fields['vx3'] = data.to_cart('velocity')

  return ds



def slice2d(data, x1=None, x2=None, x3=None):
  ''' Slice all 3-D field arrays in original data return 2-D

  coordinate of one dimension should be specified

  Args:
    data (Snapshot): 3-D data
    x1 (float): rough coordinate (optional)
    x2 (float): rough coordinate (optional)
    x3 (float): rough coordinate (optional)
  '''

  if not type(data) is Snapshot:
    raise TypeError(f'The input data is a {type(data)}, it should be Snapshot class!')

  for var in data.field_list:
    data.fields[var] = data.slice2d(var, x1=x1, x2=x3, x3=x3)


def slice1d(data, x1=None, x2=None, x3=None):
  ''' Slice all 3-D fields array in original data return 1-D

  coordinates of two dimensions should be specified

  Args:
    data (Snapshot): 3-D data
    x1 (float): rough coordinate (optional)
    x2 (float): rough coordinate (optional)
    x3 (float): rough coordinate (optional)
  '''

  if not type(data) is Snapshot:
    raise TypeError('The input data should be Snapshot class!')

  for var in data.field_list:
    data.fields[var] = data.slice1d(var, x1=x1, x2=x3, x3=x3)


# def load(fn):  # in construction
  # ''' load PLUTO simulation all outputs and return time series data '''
  # fn = os.path.expanduser(fn)
  # if any(wildcard in fn for wildcard in '[]?!*'):
    # return 0
  # return 0






class Preview(object):
  ''' Class for previewing data results.

  Args:
    wdir (str): absolute path to the dirctory.
    datatype (str): type of data files. Default is 'vtk'.
  '''

  # matplotlib setup
  mpl.rcParams['mathtext.fontset'] = 'cm'
  mpl.rcParams['font.family'] = 'serif'

  def __init__(self, wdir='./', datatype='vtk'):
    self.wdir = os.path.abspath(wdir)+'/'
    self.datatype = datatype

    self.fig = plt.figure(figsize=(5,4), tight_layout=True)


  def show(self):
    ''' show figure in prompt window '''

    return plt.show()


  def save(self, name, path='./', **kwargs):
    ''' save figure

    Args:
      name (str): filename prefix
      path (str): path to save figure. (optional) Default is './'

    Returns:
      jpg file: with name of 'name-model.jpg', model is name of dirctory
    '''

    model = self.wdir.split('/')[-2]
    plt.savefig(path+name+f'-{model}.jpg',bbox_inches='tight', pad_inches=0.02, dpi=kwargs.get('dpi',300))


  def display(self, ns, field, x1=None, x2=None, x3=None, log=True, **kwargs):
    ''' Display a 2D data using the matplotlib's pcolormesh

    Args:
      ns (int/float): should be a integer in default, \
          but if it is a negative integer, return the last number step or if it is a float, \
          it is assumed to be time, and return the nearst number step
      field: variable that needs to be displayed
      x1 (float): (optional) rough coordinate
      x2 (float): (optional) rough coordinate
      x3 (float): (optional) rough coordinate

    **kwargs:
      wdir (str): path to the directory which has the data files
      datatype (str): type of data files. Default is 'vtk'.
      vmin (float): The minimum value of the 2D array (Default : min(field))
      vmax (float): The maximum value of the 2D array (Default : max(field))
      cmap (str): color scheme of the colorbar
      title (str): Sets the title of the image.
      size (float): fontsize of title
    '''

    ds = Dataset(w_dir=kwargs.get('wdir', self.wdir), datatype=kwargs.get('datatype', self.datatype))
    ss = ds[ns]
    if ss.geometry != 'CARTESIAN':
      ss = to_cart(ss)

    if kwargs.get('in_astro_unit'):
      ss.in_astro_unit()

    offset = [x1,x2,x3]
    label = ['x1','x2','x3']

    if ss.ndim==3:
      arr = ss.slice2d(field, x1=x1, x2=x2, x3=x3)
      for i in range(3):  # find the dirction
        if offset[i] is not None:
          dim = 'x'+str(i+1)
          label.remove(dim)
          break
      x1 = ss.coord[label[0]]
      x2 = ss.coord[label[1]]
    elif ss.ndim==2:
      arr = ss.fields[field]
      x1 = ss.grid['x1']
      x2 = ss.grid['x3']

    ax1 = self.fig.add_subplot(111)
    ax1.set_aspect('equal')
    if ss.is_quantity:  # pcolormesh does not support Quantity
      x1 = x1.value
      x2 = x2.value
      arr = arr.value
    ax1.axis([np.amin(x1),np.amax(x1),np.amin(x2),np.amax(x2)])
    if log:
      pcm = ax1.pcolormesh(x1,x2,arr,vmin=kwargs.get('vmin'),vmax=kwargs.get('vmax'), \
        cmap=kwargs.get('cmap'), shading='auto', norm=mpl.colors.LogNorm())
    else:
      pcm = ax1.pcolormesh(x1,x2,arr,vmin=kwargs.get('vmin'),vmax=kwargs.get('vmax'), \
        cmap=kwargs.get('cmap'), shading='auto')

    plt.title(kwargs.get('title',f't = {ss.time:.3e}'),size=kwargs.get('size'))

    # Add a new axes beside the plot to present colorbar
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.0)
    cb = plt.colorbar(pcm, cax=cax,orientation='vertical')
    defined_fields = PlutoFluidInfo.known_fields
    if log:
      cb.ax.set_ylabel(r'$\log\;$'+defined_fields.get(field)[-1][0])
    else:
      cb.ax.set_ylabel(defined_fields.get(field)[-1][0])

    return self


  def line(self, ns, field, x1=None, x2=None, x3=None, **kwargs):
    ''' Show 1D profile

    Args:
      ns (int/float): should be a integer in default, \
          but if it is a negative integer, return the last number step or if it is a float, \
          it is assumed to be time, and return the nearst number step
      field: variable that needs to be displayed
      x1 (float): (optional) rough coordinate
      x2 (float): (optional) rough coordinate
      x3 (float): (optional) rough coordinate

    **kwargs:
      wdir (str): path to the directory which has the data files
      datatype (str): type of data files. Default is 'vtk'.
      xlog (bool): set x-axis in log scale
      ylog (bool): set x-axis in log scale
    '''

    ds = Dataset(w_dir=kwargs.get('wdir', self.wdir), datatype=kwargs.get('datatype', self.datatype))
    ss = ds[ns]
    indx = [x1,x2,x3]
    label = ['x1','x2','x3']
    for i in range(3):
      if indx[i] is None:
        dim = label[i]
        x = ss.coord[dim]
        break

    value = ss.slice1d(field,x1=x1,x2=x2,x3=x3)

    plt.plot(x, value, label=f't={ss.time:.3e}')

    plt.title(kwargs.get('title','Title'),size=kwargs.get('size'))

    if kwargs.get('xlog'): plt.xscale('log')
    if kwargs.get('ylog'): plt.yscale('log')

    plt.legend(frameon=False)

    return self


  def hist(self, *var, op=None, **kwargs):  # in construction
    ''' Preview temperal evolution stored in hist.dat file '''

    hist = pd.read_csv(self.wdir+'hist.dat', sep='\s+', index_col='t')
    hist = hist[list(var)]

    if op is None:
      ax = plt.plot(hist)
      plt.legend(ax,var)
    elif op == 'diff':
      ax = plt.plot(hist-hist.iloc[[0]].values)
      plt.legend(ax,[f'{v}-{v}(0)' for v in var])
    elif op == 'norm':
      ax = plt.plot(hist/hist.iloc[[0]].values)
      plt.legend(ax,[f'{v}/{v}(0)' for v in var])
    else:
      raise KeyError(f'Operation [{op}] has not defined yet.')

    plt.title(kwargs.get('title','Title'),size=kwargs.get('size'))
    plt.xlabel(kwargs.get('label1','Time (code unit)'),size=kwargs.get('size'))
    plt.ylabel(kwargs.get('label2','Ylabel'),size=kwargs.get('size'))

    if kwargs.get('xlog'): plt.xscale('log')
    if kwargs.get('ylog'): plt.yscale('log')

    return self


if __name__ == '__main__':
  fire.Fire(Preview)
