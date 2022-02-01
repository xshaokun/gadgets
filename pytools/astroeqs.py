#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""astroeqs.py - commonly used formulae.

Last Modified: 2021.01.03

Copyright(C) 2020 Shaokun Xie <https://xshaokun.com>
Licensed under the MIT License, see LICENSE file for details
"""


import numpy as np
from astropy import constants as cons
from astropy import units as u
from skpy.utilities.logger import fmLogger as mylog



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

    if ('T' in kw):
        T = kw['T']
        cs = np.sqrt(gamma* cons.k_B * T / (mu* cons.m_p))
    else:
        try:
            P = kw['P']
            rho = kw['rho']
            cs = np.sqrt(gamma* P / rho)
        except KeyError as ke:
            print(f"The keyword {kw} is unvalid, you should input either T or (P & rho)")
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
    if ('T' in kw and 'P' in kw):
        result = coeff * P * T
    elif ('T' in kw and 'P' in kw):
        result = kw['P'] / kw['T'] / coeff
    elif ('P' in kw and 'rho' in kw):
        result = kw['P'] / kw['rho'] / coeff
    else:
        raise KeyError('Two values among T, rho and P shouled be given.')
    return result


def radcool(temp, zmetal):
    """ Cooling Function

    This version redefines Lambda_sd
    (rho/m_p)^2 Lambda(T,z) is the cooling in erg/cm^3 s

    Args:
        temp : temperature in the unit of K
        zmetal: metallicity in the unit of solar metallicity

    Return:
        in the unit of erg*s*cm^3
    """

    tshape = temp.shape
    tempflt = temp.flatten()
    
    qlog0 = np.zeros_like(tempflt)
    qlog1 = np.zeros_like(tempflt)

    for i, t in enumerate(tempflt):
        tlog = np.log10(t)

        # zero metal cooling coefficient Lambda_([Fe/H]=0
        if tlog>=6.1:
            qlog0[i] = -26.39 + 0.471*(np.log10(t + 3.1623e6))
        elif tlog>=4.9:
            arg = 10.**(-(tlog-4.9)/.5) + 0.077302
            qlog0[i] = -22.16 + np.log10(arg)
        elif tlog>=4.25:
            bump1rhs = -21.98 - ((tlog-4.25)/0.55)
            bump2lhs = -22.16 - ((tlog-4.9)/0.284)**2
            qlog0[i] = max(bump1rhs,bump2lhs)
        else:
            qlog0[i] = -21.98 - ((tlog-4.25)/0.2)**2

        if qlog0[i]==np.nan: mylog.warning('There is NaN.')
        
        # emission from metals alone at solar abundance
        if tlog>=5.65:
            tlogc = 5.65
            qlogc = -21.566
            qloginfty = -23.1
            p = 0.8
            qlog1[i] = qlogc -p*(tlog - tlogc)
            qlog1[i] = max(qlog1[i],qloginfty)
        else:         
            tlogm = 5.1
            qlogm = -20.85
            sig = 0.65
            qlog1[i] = qlogm - ((tlog - tlogm)/sig)**2

    qlambda0 = 10.**qlog0

    qlambda1 = 10.**qlog1

    # final cooling coefficient Lambda_sd:
    radcoolsd = qlambda0 + zmetal.flatten()*qlambda1
    radcoolsd = radcoolsd.reshape(tshape)
    

    return radcoolsd