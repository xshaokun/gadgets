# SKPY

**SKPY** is the short for **Python Modules by Shaokun**. It is a python package containing several modules used by myself. Part of them are written by myself, and I will illustrate its usage if you have interests. However, there are also some modules collected from the Internet. For those, I will present the original link.

So far, the package includes the following modules:
* [fermi.py](fermi): contains the class and functions used to process the data of FERMI, a private hydro simulation code
* `streamplot.py`: streamplot for uneven grids. [forked from [here](https://github.com/tomflannaghan/matplotlib/blob/streamplot-real-space-integrate/lib/matplotlib/streamplot.py)]
* [astroeqs.py](astroeqs): provides some frequently-used formulae to calculate astronomical quantities quickly.

## Installation

In Python, there are two methods to import a third-party package, both of them are used to "tell"" Python where to find this package.

The first method is the package `sys`, add the following code at the beginning of your code:

    import sys
    sys.path.append(r"path_to_your_module")

It is required every time when you use the package because the `sys.path` would be restored after the kernel turned down.

But the second method does the same thing once and for all :

Create a `pth` file, such as `mypython.pth` in which add the path of your package:

    $HOME/lib/my_python

    Then move this `mypython.pth` file to `Python_installation_directory/python*/lib/site-packages/`.

That's all : )

## Contact
If you have any questions, please feel free to ask me. My name is Shaokun Xie. The contact details are not listed here. Please go to find the right approach in somewhere and contact me : p
