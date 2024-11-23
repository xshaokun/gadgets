#!/usr/bin/env python3
"""tools.py - some helpful functions.

Licensed under the MIT License, see LICENSE file for details
"""

import re

import numpy as np
from scipy.integrate import quad_vec


def nearest(arr, target):
    """get the index of nearest value

    Given a number, find out the index of the nearest element in an 1D array.

    Args:
        arr: array for searching
        target: target number
    """

    index = np.abs(arr - target).argmin()

    return index


def str_to_number(string):
    """turn a string of number into correct type"""
    # This is referred to
    # https://stackoverflow.com/questions/41668588/regex-to-match-scientific-notation
    _FLOAT_PATTERN_ = r"^[+-]?(?:0|[1-9]\d*)(?:\.\d*)$"
    _SCI_NOT_PATTERN_ = r"^[+-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+-]?\d+)$"
    _FLOAT_REGEXP_ = rf"{_FLOAT_PATTERN_}|{_SCI_NOT_PATTERN_}"
    _INT_REGEXP_ = r"^[+-]?\d+$"
    float_match = re.fullmatch(_FLOAT_REGEXP_, string)
    int_match = re.fullmatch(_INT_REGEXP_, string)
    if float_match:
        return float(string)
    elif int_match:
        return int(string)
    else:
        raise ValueError(f"Cannot identify the string as a number: {string}")


def quad_with_units(func, a, b):
    # do not support mutiple integrations yet
    uint = (func(a)).units
    ux = getattr(a, "units", 1)
    units = uint * ux

    def func_new(x):
        return func(x).v

    return quad_vec(func_new, a, b)[0] * units
