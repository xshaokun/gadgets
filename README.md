# SKPY

**SKPY** is the short for **Python Modules by Shaokun**. It is a python package containing several modules used by myself. Part of them are written by myself, and I will illustrate its usage if you have interests. However, there are also some modules collected from the Internet. For those, I will present the original link.

So far, the package includes the following modules:
* [fermi.py](fermi): contains the class and functions used to process the data of FERMI, a private hydro simulation code.
* `pluto.py`: contains the class and functions used to process the data from PLUTO.
* `streamplot.py`: matplotlib streamplot for uneven grids. [forked from [here](https://github.com/tomflannaghan/matplotlib/blob/streamplot-real-space-integrate/lib/matplotlib/streamplot.py)]
* [astroeqs.py](astroeqs): provides some frequently-used formulae to calculate astronomical quantities quickly.

## Installation
First of all, download this module.

In Python, a third-party module can be installed temporarily, or permanently. 

For temporarily using a module, you should implement the following code at the beginning of your script:

    import sys
    sys.path.append(r"path_to_your_module")
    
Then the path would be removed every time after your script running.

For install a module permanently, it is better to use a package manager, or `$PYTHONPATH`.

Just like `$PATH`, open `.bashrc` file or `.zshrc` file (if you use zsh shell), and include the following command:

    export PYTHONPATH=/the/path/to/skpy:$PYTHONPATH
    
Finally, `source .bashrc` to reload shell configuration.

That's all : )

## Contact
If you have any questions, please feel free to ask me. My name is Shaokun Xie. The contact details are not listed here. Please go to find the right approach in somewhere and contact me : p
