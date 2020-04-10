# PYMODSK

Some python modules written for simulation visualization.

## Contact

Author: Shaokun Xie

Website: https://xshaokun.github.io

## Modules

`fermi.py`: the class and functions to deal with the data of `fermi.f` (private simulation code).

`streamplot.py`: streamplot for uneven grids. [forked from [here](https://github.com/tomflannaghan/matplotlib/blob/streamplot-real-space-integrate/lib/matplotlib/streamplot.py)]


### FERMI.PY

#### Class FermiData(dirpath='./')

##### Attributes

- `self.dir_path`: location where data stored
- `self.iezone`: number of even grids
- `self.ilzone`: number of logarithmic grids
- `self.ezone`: outer boundary of even grids
- `self.zone` = number of total grids
- `self.reso` = the resolution of even grids
- `self.kprint` = time for outputs

##### Methods

- `read_xh(self)`: read coordinate file, converting the unit to kpc
- `read_var(self, var, kprint)`: read certain variable file, converting into ndarray
- `read_hist(self, var)`: read log file, converting into dataset

#### Function

- `meshgrid(data,rrange,zrange)`: construct mesh grid of spatial coordinations.
- `mesh_var(data, var, meshgrid)`: generate meshgrid of variable based on coord meshgrid
- `slice_mesh(data, coord, direction='z', kpc=0)`: slice the values of variable in certain direction
- `find_nearst(arr,target)`: given a number, find out the index of nearest element in an 1D array