#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""astroeqs.py - commonly used formulae.

Last Modified: 2020.05.28

Copyright(C) 2020 Shaokun Xie <https://xshaokun.com>
Licensed under the MIT License, see LICENSE file for details
"""


import numpy as np
from astropy import constants as cons
from astropy import units as u


def Ledd(mbh):
    """Eddington Luminosity

    Given Black Hole mass in the unit of Msun:
        L = 4*pi*G*M*m_p*c / sigma_T

    Return:
        L: in the unit of Lsun
    """

    L = 4 * np.pi * cons.G* mbh * cons.m_p * cons.c / cons.sigma_T
    return L.to(u.Lsun)


def MdotEdd(mbh, epsilon=0.1):
    """Eddington Accretion Rate

    Given Black Hole mass in the unit of Msun:
        Mdot = Ledd / (epsilon*c^2)
            where epsilon = 0.1 (default)
    
    Return:
        Mdot: in the unit of Msun/yr
    """

    Mdot = Ledd(mbh) / (epsilon * cons.c**2)
    return Mdot.to(u.Msun/u.yr)


def cs(gamma=5./3., mu=0.61, **kw):
    """Sound Speed

    For general equations of state: 
        cs = sqrt(\partial P / \partial rho)_s 
    The derivative is taken at constant entropy s, because sound wave travels so fast that its propagation can be approximated as adiabatic process.
    For ideal gas:
        cs = sqrt(gamma* P / rho)              (given pressure & density)
           = sqrt(gamma* k_b * T / (mu* m_p))  (given temperarature)

    Keywords:
        gamma: adiabadic index, default is 5/3
        mu: mean particle weight, default is 0.61

    **keywords:
        T: temperature in unit of K
        P: pressure in unit of erg/cm**3
        rho: density in unit of g/cm**3
    
    Return:
        cs: in the unit of km/s

    Example:
        >>> cs(T=1e6)
        150.17658kms

        >>> cs(P=3e-12, rho = 1e-26)
        223.6068kms

        >>> cs(gamma=5./3., mu=0.8, T=1e6)
        131.13606kms
    """

    if 'T' in kw:
        T = kw['T']
        cs = np.sqrt(gamma* cons.k_B * T / (mu* cons.m_p))

    else:
        try:
            P = kw['P']
            rho = kw['rho']
            cs = np.sqrt(gamma* P / rho)
        except KeyError as ke:
            print(f"The keyword {ke} is unvalid, you should input either T or (P & rho)")
        else:
            print('Such error is unexpected, so explore the cause by yourself')
        
    return cs.to(u.km/u.s)

def vkep(m, r=1.*u.kpc):
    """Keplerian velocity

    Given a point mass M and radius r:
        vkep = G * M / r
    
    Return:
        vkep: in the unit of km/s
    """

    vkep = cons.G * m / r
    return vkep.to(u.km/u.s)

def eos(mu=0.61, **kw):
    """ Equation of State

    Keywords:
        mu: mean particle weight, default is 0.61
    
    **keywords:
        T: temperature in unit of K
        P: pressure in unit of erg/cm**3
        rho: density in unit of g/cm**3

    Return:
        in the unit of cgs

    Example:
        >>> eos(T=1e6, P=3e-12)

        >>> eos(P=3e-12, rho = 1e-26)
    """
    
    coeff = (cons.k_B / mu / cons.m_p).cgs.value
    if 'T' in kw and 'P' in kw:
        result = coeff * P * T
    elif 'T' in kw and 'P' in kw:
        result = kw['P'] / kw['T'] / coeff
    elif 'P' in kw and 'rho' in kw:
        result = kw['P'] / kw['rho'] / coeff
    else:
        raise KeyError('Two values among T, rho and P shouled be given.')
    return result