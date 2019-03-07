#
# Example for the Ulrich (1976) flattened protostellar envelope model, 
# implemented for RADMC-3D, base on the python code in the HYPERION 
# radiative transfer package.
#
# Take care whether you specify the rho0 of dust+gas or dust only!
#
# Example
# -------
#
# nr, nth, nph = 100, 60, 1
#
# ri       = np.logspace(np.log10(rin),np.log10(rout),nr+1)
# thetai   = np.linspace(0.0,0.5*np.pi,nth+1)
# phii     = np.linspace(0.0,np.pi*2.0,nph+1)
# rc       = 0.5 * ( ri[0:nr] + ri[1:nr+1] )
# thetac   = 0.5 * ( thetai[0:nth] + thetai[1:nth+1] )
# phic     = 0.5 * ( phii[0:nph] + phii[1:nph+1] )
#
# rr, tt, ph   = np.meshgrid(rc,thetac,phic,indexing='ij')
#
# rho_env = ulrich_envelope(rr, tt, Rc=300.*au, rho0=1.165e-20, rmin=50.*au)
#
#
# Adapted 2018 Laszlo Szucs <laszlo.szucs@mpe.mpg.de>
#
# See the original license at the end of the file.
#

# import 3rd party libraries
import numpy as np

def delta_neg(r, q):

    rho = np.sqrt(-q.real ** 3)
    theta = np.arccos(r.real / rho)

    s = (rho ** (1. / 3.) * np.cos(theta / 3.)).astype(np.complex)
    s.imag = rho ** (1. / 3.) * np.sin(theta / 3.)

    t = (rho ** (1. / 3.) * np.cos(-theta / 3.)).astype(np.complex)
    t.imag = rho ** (1. / 3.) * np.sin(-theta / 3.)

    return s, t


def delta_pos(r, delta):

    dr = (r + np.sqrt(delta)).real
    s = dr ** (1. / 3.)
    neg = dr < 0.
    s[neg] = - (- dr[neg]) ** (1. / 3.)

    dr = (r - np.sqrt(delta)).real
    t = dr ** (1. / 3.)
    neg = dr < 0.
    t[neg] = - (- dr[neg]) ** (1. / 3.)

    return s, t


def cubic(c, d):
    '''
    Solve x**3 + c * x + d = 0
    '''

    c = c.astype(np.complex)
    d = d.astype(np.complex)

    q = c / 3.
    r = - d / 2.

    delta = q ** 3 + r ** 2

    pos = delta >= 0.

    s = np.zeros(c.shape, dtype=np.complex)
    t = np.zeros(c.shape, dtype=np.complex)

    if np.sum(pos) > 0:
        s[pos], t[pos] = delta_pos(r[pos], delta[pos])

    if np.sum(~pos) > 0:
        s[~pos], t[~pos] = delta_neg(r[~pos], q[~pos])

    x1 = s + t
    x2 = - (s + t) / 2. + np.sqrt(3.) / 2. * (s - t) * np.complex(0., 1.)
    x3 = - (s + t) / 2. - np.sqrt(3.) / 2. * (s - t) * np.complex(0., 1.)

    return x1, x2, x3


def solve_mu0(ratio, mu):

    x = cubic(ratio - 1., - mu * ratio)

    ambig = (x[0].imag == 0) & (x[1].imag == 0) & (x[2].imag == 0)

    v = x[0].real

    mask = ambig & (mu >= 0) & ((x[0].real >= 0)
                                & (x[1].real < 0)
                                & (x[2].real < 0))
    v[mask] = x[0][mask].real
    mask = ambig & (mu >= 0) & ((x[0].real < 0)
                                & (x[1].real >= 0)
                                & (x[2].real < 0))
    v[mask] = x[1][mask].real
    mask = ambig & (mu >= 0) & ((x[0].real < 0)
                                & (x[1].real < 0)
                                & (x[2].real >= 0))
    v[mask] = x[2][mask].real

    mask = ambig & (mu < 0) & ((x[0].real < 0)
                               & (x[1].real >= 0)
                               & (x[2].real >= 0))
    v[mask] = x[0][mask].real
    mask = ambig & (mu < 0) & ((x[0].real >= 0)
                               & (x[1].real < 0)
                               & (x[2].real >= 0))
    v[mask] = x[1][mask].real
    mask = ambig & (mu < 0) & ((x[0].real >= 0)
                               & (x[1].real >= 0)
                               & (x[2].real < 0))
    v[mask] = x[2][mask].real

    return v

def ulrich_envelope(rr,tt, Rc=4.4879e15, rho0=1.165e-20, rmin=None, rmax=None):
    '''
    Calculates the Ulrich (1976) flattened envelope density 
    distribution.

    The function and the function calls are based on the HYPERION radiative
    transfer package (Robitaille 2010)

    Parameters
    ----------
        rr:   radial coordinate in spherical coordinate system. The array is 
              typically 2 dimensional (r,theta)
        tt:   theta coordinate in spherical coordinate system, typically 
              2 dimentional (r,theta)
        Rc:   centrifugal radius (cm), default value is equal to 300 au
        rho0: density factor (g cm^-3), user should take care 
              whether gas + dust or dust only
        rmin: inner radius (cm), if not set then min(rr) is used
        rmax: outer radius (cm), if not set then max(rr) is used
    
    '''
    if not rmin:
       rmin = np.min(rr)
    if not rmax:
       rmax = np.max(rr)
    # 
    mu  = np.cos(tt)
    #
    # Find mu_0, the cosine of the angle of a streamline of infalling
    # particles at r=infinity. (original code from HYPERION)
    mu0 = solve_mu0(rr / Rc, mu)

    # Calculate the density
    rho_env = rho0 * (rr/Rc)**(-3./2.) * (1.0 + mu/mu0)**(-0.5) * \
                         (mu/mu0 + 2.0 * mu0**2 * Rc / rr)**(-1)
    # Truncate the desnity distribution
    rho_env[ rr <= rmin ] = 0.0
    rho_env[ rr >= rmax ] = 0.0

    return rho_env


#
# Copyright (c) 2012-13, Thomas P. Robitaille
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
#  * Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
