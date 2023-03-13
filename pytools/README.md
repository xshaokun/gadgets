## pytools

Some python tools useful in other python code.

* `fermi.py`: contains the class and functions used to process the data of FERMI, a private hydro simulation code.
* `astroeqs.py`: provides some frequently-used formulae to calculate astronomical quantities quickly.
* `logger.py`: some formatted log messages
* `pluto_tools.py`: some tools  used specificaly for PLUTO simulations
* `tools.py`: some general used tools

Note that the modules here are wrapped as a standard python package `xsk_pytools` (refer to
`pyproject.toml`), which can be installed by `pip`:

    $ python -m pip install --user -e .

Then, it can be imported in your python script:

    import xsk_pytools
