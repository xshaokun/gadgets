import abc
import functools
import numbers
from collections.abc import Iterable
from typing import Union

import numpy as np
import unyt as u
from scipy.special import kv
from unyt import unyt_array, unyt_quantity
from unyt.physical_constants import c_cgs, me_cgs, qe_cgs

from .tools import quad_with_units

k43 = functools.partial(kv, 4 / 3)
k13 = functools.partial(kv, 1 / 3)

me0 = (me_cgs * c_cgs**2).to("MeV")  # rest mass of electron


class GeneralCRDist(abc.ABC):
    """
    An abstract class of cosmic ray distribution

    Requires the lower and upper energy bound (Lorentz factor or energy are both acceptable)
    and total number or energy to be normailzed.
    Subclasses only need to define the method self.distribution(gamma)

    Arguements:
        e_min (tuple/float): minimum energy (Lorentz factor) of cosmic ray
        e_max (tuple/float): maximum energy (Lorentz factor) of cosmic ray
        total_number (unyt_array/tuple): volume number density of the cosmic ray
        total_energy (unyt_array/tuple): volume energy density of the cosmic ray
        norm (float/unyt_array): additional normalization

    Attributes:
        g_min: minimum energy of cosmic ray (Lorentz factor)
        g_max: maximum energy of cosmic ray (Lorentz factor)
        e_min: minimum energy of cosmic ray
        e_max: maximum energy of cosmic ray
        total_number: volume number density of the cosmic ray
        total_energy: volume energy density of the cosmic ray
        norm: normalization of the distribution

    Methods:
        number density(gamma): return number density for a given energy
    """

    def __init__(self, e_min, e_max, total_number=None, total_energy=None, norm=1):
        self._e_min = e_min  # minimum energy of cosmic ray
        self._e_max = e_max  # maximum energy of cosmic ray
        self._total_number = total_number
        self._total_energy = total_energy
        self._norm = norm
        self._normalize_dist(total_number=total_number, total_energy=total_energy)

    def _sanitize_energy(self, energy):
        # convert input energy to unyt_quantity
        if isinstance(energy, numbers.Number):
            # given a Lorentz factor
            return energy * me0
        elif isinstance(energy, tuple):
            return unyt_quantity(*energy)
        elif isinstance(energy, unyt_quantity):
            return energy
        else:
            raise TypeError(
                f"Cannot handle type {type(energy)}, only accept scalar(Lorentz factor), tuple(freqmber, units) and unyt_quantity"
            )

    @property
    def e_min(self):
        # lower energy bound of cosmic ray
        return self._sanitize_energy(self._e_min).to("MeV")

    @property
    def e_max(self):
        # upper energy bound of cosmic ray
        return self._sanitize_energy(self._e_max).to("MeV")

    @property
    def g_min(self):
        # Lorentz factor equivalent to e_min
        return (self.e_min / me0).to("dimensionless")

    @property
    def g_max(self):
        # Lorentz factor equivalent to e_max
        return (self.e_max / me0).to("dimensionless")

    @abc.abstractmethod
    def _distribution(self, energy):
        # define the distribution of the cosmic ray by energy
        pass

    def number_density(self, gamma):
        # return the number density according to the given energy
        n = self.norm[:, None] * self._distribution(gamma)
        return n.to("cm**-3")

    def _int_number(self):
        # integrate the distribution without the normalization
        v = quad_with_units(self._distribution, self.g_min, self.g_max)
        return v

    @property
    def total_number(self):
        if self._total_number is None:
            tn = self._int_number() * self.norm
        else:
            tn = self._total_number
        return tn.to("cm**-3")

    def _int_energy(self):
        # integrate the distribution of energy without the normalization
        def integrand(gamma):
            return self._distribution(gamma) * gamma * me0

        v = quad_with_units(integrand, self.g_min, self.g_max)
        return v

    @property
    def total_energy(self):
        if self._total_energy is None:
            te = self._int_energy() * self.norm
        else:
            te = self._total_energy
        return te.to("erg/cm**3")

    def _normalize_dist(self, total_number=None, total_energy=None):
        if total_number is not None and total_energy is None:
            self._total_number = total_number
            self.norm = total_number / self._int_number() / self._norm * self._norm
        elif total_energy is not None and total_number is None:
            self._total_energy = total_energy
            self.norm = total_energy / self._int_energy() / self._norm * self._norm
        elif total_energy is None and total_number is None:
            raise ValueError(
                "Must specify one of the keyword arguments total_number or total_energy to normalize the distribution"
            )
        else:
            raise ValueError(
                "Both total_number and total_energy are specified, choose one of them"
            )


class PowerLawCR(GeneralCRDist):
    def __init__(self, index, e_min, e_max, total_number=None, total_energy=None):
        self.index = index
        super().__init__(
            e_min, e_max, total_energy=total_energy, total_number=total_number
        )

    def _distribution(self, gamma):
        return (gamma / self.g_min) ** (-self.index)


class Synchrotron:
    """
    Synchrotron emission

    Refer to
    Owen & Yang (2020)
    Dermer & Menon (2009) High energy radiation from black holes

    Arguments:
        b_field (tuple): strength of magnetic field
        cr_dist (GeneralCRDist): distribution of cosmic ray

    Attributes:
        g_min: minimum energy of cosmic ray (Lorentz factor)
        g_max: maximum energy of cosmic ray (Lorentz factor)
        e_min: minimum energy of cosmic ray
        e_max: maximum energy of cosmic ray
        fre_B: frequency of gyration

    Methods:
        freq_c(gamma): return a critical frequency for a given energy
        emissivity(freq): return emissivity for a given frequency
        spectrum(freq_list): return spectrum within given band, i.e. a list of emissivity
    """

    re = qe_cgs**2 / me0  # classical electron radius

    def __init__(self, b_field: tuple[float, str], cr_dist: GeneralCRDist):
        self.b_field = unyt_array(b_field)  # magnetic field strength
        if isinstance(cr_dist, GeneralCRDist):
            self.cr_dist = cr_dist  # distribution of cosmic ray
        else:
            raise TypeError(
                f"{type(cr_dist)} is not supported for the argument cr_dist, only a GeneralCRDist object is valid"
            )
        self.g_min = self.cr_dist.g_min  # minimum energy of cosmic ray (Lorentz factor)
        self.g_max = self.cr_dist.g_max  # maximum energy of cosmic ray (Lorentz factor)
        self.e_min = self.cr_dist.e_min.to("GeV")  # minimum energy of cosmic ray
        self.e_max = self.cr_dist.e_max.to("GeV")  # maximum energy of cosmic ray

        self.freq_B = (
            np.abs(qe_cgs) * self.b_field / (2 * np.pi * me_cgs * c_cgs)
        )  # frequency of gyration
        self._norm = 4 / 9 * np.sqrt(3) * np.pi * self.re * me_cgs * c_cgs / self.freq_B

    def freq_c(self, gamma):
        # return a critical frequency for a given energy
        return (3 / 2 * self.freq_B * gamma**2).to("Hz")

    def _f_func(self, freq, gamma):
        xc = (freq / self.freq_c(gamma) / 2).v
        return k43(xc) * k13(xc) - 3 / 5 * xc * (k43(xc) ** 2 - k13(xc) ** 2)

    def emissivity(self, freq: Union[tuple[float, str], unyt_quantity]):
        # return emissivity for a given frequency
        # integrate by CR energy
        # in units of erg/s/Hz/cm**3
        if isinstance(freq, tuple):
            freq = unyt_quantity(*freq).to("Hz")

        def integrand(gamma):
            return (
                freq**2
                * self.cr_dist._distribution(10)
                * self._f_func(freq, gamma)
                / gamma**4
            )

        emiss = quad_with_units(integrand, self.g_min, self.g_max)
        return (self._norm * emiss * self.cr_dist.norm).to("erg/s/Hz/cm**3")

    def spectrum(self, freq_list: Iterable):
        # return spectrum within given band, i.e. a list of emissivity
        # in units of erg/s/Hz/cm**3
        spec = u.unyt_array(np.zeros(freq_list.shape), "erg/s/Hz/cm**3")
        for i, f in enumerate(freq_list.to("Hz")):
            spec[i] = self.emissivity(f)
        return spec.to("erg/cm**3")
