import argparse

import pandas as pd

from xsk_pytools.tools import nearest, str_to_number


def output_index(logfile, ns):
    """ find the index of dataset
    Args:
        logfile (str): the log file records the information of datasets.
        ns (int/float): should be an integer in default, \
            but if it is a negative integer, return the last index, or \
            if it is a float, it is assumed to be time, and return the nearst index
    """
    hdr = ["nfile", "time", "dt", "Nstep"]
    log = pd.read_table(logfile, sep=" ", usecols=[0, 1, 2, 3], names=hdr, index_col=0)
    if isinstance(ns, int):
        if ns < 0:
            return log.index[-1]
        else:
            return ns
    elif isinstance(ns, float):
        if ns < 0:
            return log.index[-1]
        else:
            # given a specific [time], find [ns] corresponding nearst existed data [time].
            return nearest(log["time"], ns)
    else:
        raise TypeError(f"The type of {ns}({type(ns)}) should be int or float.")


class ParseIndex(argparse.Action):
    """return correct index of datasets according to given command line arguments"""

    def __call__(self, parser, namespace, values, option_string=None):
        index = []
        logfile = namespace.dtype + ".out"
        if isinstance(values, list):
            for value in values:
                index.append(output_index(logfile, value))
        else:
            index.append(output_index(logfile, values))
        setattr(namespace, self.dest, index)


class PlutoArgumentParser(argparse.ArgumentParser):
    """A argument parser for pluto where define some common used command line options"""

    def __init__(
        self,
        description=None,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    ):
        super().__init__(
            description=description,
            formatter_class=formatter_class,
        )

        self.add_argument(
            "--dtype",
            default="vtk",
            choices=["vtk", "dbl", "flt", "tab", "dbl.h5", "flt.h5"],
            help="file type of the dataset",
        )

        self.add_argument(
            "index",
            default=-1,
            nargs="*",
            type=str_to_number,
            action=ParseIndex,
            help="[int/float] index or time of a dataset",
        )

        self.add_argument(
            "-f",
            nargs="*",
            type=str,
            dest="fields",
            help="choose certain fields to plot",
        )

        self.add_argument(
            "--tag", default=None, type=str, help="specify a tag to the filename"
        )

        self.add_argument(
            "--reg",
            default=None,
            type=str,
            action="append",
            help="yt-style condition for filter data",
        )
