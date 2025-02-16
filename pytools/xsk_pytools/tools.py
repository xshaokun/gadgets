#!/usr/bin/env python3
"""tools.py - some helpful functions.

Licensed under the MIT License, see LICENSE file for details
"""

import re
from functools import wraps

import numpy as np
import unyt as u
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


def quad_with_units(func, a, b, epsabs=1e-30, epsrel=1e-05):
    # do not support mutiple integrations yet
    uint = getattr(func(a), "units", 1)
    ux = getattr(a, "units", 1)
    units = uint * ux
    # x = np.logspace(np.log10(a), np.log10(b), 10000)
    if uint != 1:

        def func_new(x):
            return func(x).v

        return quad_vec(func_new, a, b, epsabs=epsabs, epsrel=epsrel)[0] * units
    else:
        return quad_vec(func, a, b, epsabs=epsabs, epsrel=epsrel)[0] * units
    # return trapz(y.v, x) * units


def sanitize_quantity(quantity, units, equivalence=None):
    """make sure that the quantity is an unyt_quantity and in specific units"""
    if isinstance(quantity, str):
        quan = u.unyt_quantity.from_string(quantity)
    try:
        # for a unyt_quantity
        quan = u.unyt_quantity(quantity.value, quantity.units)
    except AttributeError as e:
        try:
            # for a tuple of (value, units)
            quan = u.unyt_quantity(*quantity)
        except (RuntimeError, TypeError):
            raise TypeError(
                "dist should be a YTQuantity or a (value, unit) tuple!"
            ) from e
    return quan.to(units, equivalence)


def check_output_unit(expected_unit, equivalence=None):
    """
    A decorator to check if the output of a function has the expected unit or equivalent units.
    The physical dimensions must match, and the units can be converted to the expected unit.

    :param expected_unit: The expected unit (as a str object).
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)

            # Check if the result is a unyt array
            if not isinstance(result, u.unyt_array):
                raise ValueError(f"The output of {func.__name__} is not a unyt array.")

            # Convert the result to the expected unit (if possible)
            try:
                result.to(expected_unit, equivalence=equivalence)
            except (u.UnitConversionError, u.InvalidUnitEquivalence) as e:
                raise u.UnitConversionError(
                    f"The output of {func.__name__} cannot be converted to the expected unit {expected_unit}. "
                    f"Error: {e}"
                ) from e

            return result

        return wrapper

    return decorator
