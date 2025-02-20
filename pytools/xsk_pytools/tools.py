#!/usr/bin/env python3
"""tools.py - some helpful functions.

Licensed under the MIT License, see LICENSE file for details
"""

import re
from functools import wraps

import numpy as np
import unyt as u
from unyt.exceptions import UnitConversionError, InvalidUnitEquivalence
from scipy.integrate import quad_vec, quad


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


def quad_with_units(func, a, b, units_str=None, epsabs=1e-30, epsrel=1e-03):
    # do not support mutiple integrations yet
    uint = getattr(arr:=func(a), "units", 1)
    if arr.size > 1:
        int_func = quad_vec
    else:
        int_func = quad
    ux = getattr(a, "units", 1)
    if units_str is not None:
        units = (1* uint * ux).to(units_str)
    else:
        units = uint * ux
    # x = np.logspace(np.log10(a), np.log10(b), 10000)
    if uint != 1:

        def func_new(x):
            return func(x).v

        return int_func(func_new, a, b, epsabs=epsabs, epsrel=epsrel)[0] * units
    else:
        return int_func(func, a, b, epsabs=epsabs, epsrel=epsrel)[0] * units


def sanitize_quantity(quantity, units, equivalence=None):
    """make sure that the quantity is an unyt_quantity and in specific units"""
    if isinstance(quantity, str):
        quan = u.unyt_quantity.from_string(quantity)
    if isinstance(quantity, float):
       # quad_vec would remove units from quantity
       return u.unyt_quantity(quantity, units)
    try:
        # for a unyt_quantity
        quan = u.unyt_quantity(quantity.value, quantity.units)
    except AttributeError as e:
        try:
            # for a tuple of (value, units)
            quan = u.unyt_quantity(*quantity)
        except (RuntimeError, TypeError):
            raise TypeError(
                f"Santinize only apply to a YTQuantity or a (value, unit) tuple, not a {type(quantity)}!"
            ) from e
    return quan.to(units, equivalence)

"""
def santinize_input_quantity(units, equivalence=None):
    def decorator(func):
        @wraps(func)
        def wrapper(quantity):
            quantity = sanitize_quantity(quantity, units, equivalence)
            return func(quantity)
"""




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
            except (UnitConversionError, InvalidUnitEquivalence) as e:
                raise RuntimeError(
                    f"The output of {func.__name__} cannot be converted to the expected unit {expected_unit}. "
                    f"Error: {e}"
                ) from e

            return result
        return wrapper
    return decorator


def integrate_with_log_segments(func, integrand, lower, upper):
    start = np.ceil(np.log10(lower))
    stop = np.floor(np.log10(upper))
    segments = np.logspace(start, stop, int(stop - start +1))
    segments = np.array([lower, *segments, upper])
    if getattr(lower, "units"):
        segments *= lower.units

    integral = 0
    for i in range(len(segments) - 1):
        # integrate within each segment
        result = func(integrand, segments[i], segments[i + 1])
        integral += result
    return integral
