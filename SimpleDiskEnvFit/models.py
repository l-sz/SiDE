# 
# Code to set up a simple RADMC-3D disk models with envelope
#
# The code heavily relies on radmc3dPy package by Attila Juhasz:
# (https://www.ast.cam.ac.uk/~juhasz/radmc3dPyDoc/index.html)
#
# Original SimpleDiskEnvFit source available at:
# https://gitlab.mpcdf.mpg.de/szucs/SimpleDiskEnvFit
#  
# Copyright (C) 2019 Laszlo Szucs <laszlo.szucs@mpe.mpg.de>
#
# Licensed under GLPv2, for more information see the LICENSE file repository.
#

from __future__ import absolute_import
from __future__ import print_function

import os
import numpy as np

import radmc3dPy

# Internal routines
from . import ulrich_envelope as uenv

def ulrich_envelope(grid, ppar, cavity=False):
    '''
    Returns dust density distribution ([nx, ny, nz, 1] dimension) of a rotationally 
    flattened protostellar envelope. The profile is given by Ulrich (1976).
    The model parameters should be provided in ppar.  Unit of the returned array 
    is [gram/cm^3] of dust.
    
    Parameters
    ----------
    grid : radmc3dPy.grid object
           Initialized with the model boundaries.
    ppar : dict
           Dictionary provided by modPar.ppar (radmc3dPar object)
    cavity : bool
           If True, then envelope model has cavity. Default is False.
    '''
    rr, th = np.meshgrid(grid.x, grid.y, indexing='ij')
    z0 = np.zeros([grid.nx, grid.ny, grid.nz], dtype=np.float64)
    zz   = rr * np.cos(th)
    rcyl = rr * np.sin(th)

    # Envelope density array
    rho_env = np.zeros([grid.nx, grid.ny, grid.nz,1], dtype=np.float64)

    dummy = uenv.ulrich_envelope(rr, th, rho0=ppar['rho0Env'], 
                                 rmin=ppar['rTrunEnv'], Rc=ppar['r0Env'])
    
    rho_env[:,:,0,0] = dummy
    
    if cavity:
        
        rho_env = envelope_cavity(rho_env, grid, ppar)

    # Calculate the volume of each grid cell
    mass, mass3000 = computeEnvMass(grid, rho_env)
    
    return (rho_env, mass, mass3000)

def tafalla_envelope(grid, ppar, cavity=False):
    '''
    Returns protostellar envelope dust density distribution ([nx, ny, nz, 1] 
    dimension) according Tafalla et al. (2004). The model parameters should be
    provided in ppar. Unit of the returned array is [gram/cm^3] of dust.
    
    Parameters
    ----------
    grid : radmc3dPy.grid object
           Initialized with the model boundaries.
    ppar : dict
           Dictionary provided by modPar.ppar (radmc3dPar object)
    cavity : bool
           If True, then envelope model has cavity. Default is False.
    '''
    rr, th = np.meshgrid(grid.x, grid.y, indexing='ij')
    z0 = np.zeros([grid.nx, grid.ny, grid.nz], dtype=np.float64)
    zz   = rr * np.cos(th)
    rcyl = rr * np.sin(th)

    # Envelope density array
    rho_env = np.zeros([grid.nx, grid.ny, grid.nz,1], dtype=np.float64)

    dummy = ppar['rho0Env'] * 1. / (1. + (rr/ppar['r0Env'])**(-1.0*ppar['prhoEnv']))
    
    # Reduce density within rTrunEnv radius
    crit = ( rr < ppar['rTrunEnv'] ) 
    dummy[crit] = dummy[crit] * ppar['redFactEnv']

    rho_env[:,:,0,0] = dummy

    if cavity:
        
        rho_env = envelope_cavity(rho_env, grid, ppar)

    # Calculate the volume of each grid cell
    mass, mass3000 = computeEnvMass(grid, rho_env)
    
    return (rho_env, mass, mass3000)

def powerlaw_envelope(grid, ppar, cavity=False):
    '''
    Returns power law protostellar envelope dust density distribution 
    ([nx, ny, nz, 1] dimension). The model parameters should be provided in 
    ppar. Unit of the returned array is [gram/cm^3] of dust.
    
    Parameters
    ----------
    grid : radmc3dPy.grid object
           Initialized with the model boundaries.
    ppar : dict
           Dictionary provided by modPar.ppar (radmc3dPar object)
    cavity : bool
           If True, then envelope model has cavity. Default is False.
    '''
    rr, th = np.meshgrid(grid.x, grid.y, indexing='ij')
    z0 = np.zeros([grid.nx, grid.ny, grid.nz], dtype=np.float64)
    zz   = rr * np.cos(th)
    rcyl = rr * np.sin(th)
    
    # Envelope density array
    rho_env = np.zeros([grid.nx, grid.ny, grid.nz,1], dtype=np.float64)
    
    dummy = ppar['rho0Env'] * (rr/ppar['r0Env'])**ppar['prhoEnv']
    
    # Reduce density within rTrunEnv radius
    crit = ( rr < ppar['rTrunEnv'] ) 
    dummy[crit] = dummy[crit] * ppar['redFactEnv']
    
    rho_env[:,:,0,0] = dummy
    
    if cavity:
        
        rho_env = envelope_cavity(rho_env, grid, ppar)
    
    # Calculate the volume of each grid cell
    mass, mass3000 = computeEnvMass(grid, rho_env)
    
    return (rho_env, mass, mass3000)

def envelope_cavity(rho, grid, ppar, modeCav=None):
    '''
    Returns density distribution with reduced values in a cone or cylinder.
    The opening angle and reduction factors are specified in ppar.
    
    Parameters
    ----------
    rho  : array_like, float
           Density distribution of an envelope ([nx, ny, nz, 1] dimension)
    grid : radmc3dPy.grid object
           Initialized with the model boundaries.
    ppar : dict
           Dictionary provided by modPar.ppar (radmc3dPar object)
    '''
    rr, th = np.meshgrid(grid.x, grid.y, indexing='ij')
    z0 = np.zeros([grid.nx, grid.ny, grid.nz], dtype=np.float64)
    zz   = rr * np.cos(th)
    rcyl = rr * np.sin(th)
    
    # Recenter the grid; used for modeCav = 'gridcenter'
    zz_n = 0.0
    rcyl_n = ppar['rTrunEnv']
    rr_n = np.sqrt( (rcyl-rcyl_n)**2 + (zz-zz_n)**2 )
    th_n = np.arctan( (rcyl-rcyl_n) / (zz-zz_n) )
    
    dummy = rho[:,:,0,0]
    
    # Decide which cavity shape
    if modeCav is not None:
        ppar.setPar(['modeCav',str(modeCav)])
    
    # Find cavity opening angle in parameter list
    if 'thetac_deg' in ppar.keys():
        theta_cav_deg = ppar['thetac_deg']
    elif 'thetDegCav' in ppar.keys():
        theta_cav_deg = ppar['thetDegCav']
    else:
        raise ValueError('ERROR [envelope_cavity]: Cavity opening angle not given')
    
    if (ppar['modeCav'] == 'Sheehan2017'):
        crit = ( zz > (1.0*au + rcyl**(1.0)) )
    elif (ppar['modeCav'] == 'edgecenter'):
        crit = ( th_n < np.deg2rad(theta_cav_deg) )
    elif (ppar['modeCav'] == 'gridcenter'):
        crit = ( th < np.deg2rad(theta_cav_deg) )
    else:
        raise ValueError('ERROR [envelope_cavity]: Unknown cavity mode: modeCav = {:s}'.format(
            ppar['modeEnv'] ))
         
    dummy[crit] = dummy[crit] * ppar['redFactCav']
    
    rho[:,:,0,0] = dummy
    
    return rho

def slab(grid, r0=0.0, r1=0.0, H0=0.0, H1=0.0, rho0=0.0, sig0=None, smoothed=True):
    '''
    Returns the dust density distribution of a slab between r0 and r1 radius and 
    z0 and z1 in height. The slab is rectangular in Cartesian coordinates.
    SimpleDiskEnvFit, however, uses spherical coordinate system, where the cells 
    have curved shapes. Adding a slab with Hz < r will result in blocky density 
    distribution.
    
    Furthermore a sharp block-like density distribution is not realistic in most 
    protostar / protoplanetery disk cases. 
    
    The density and surface density parameters are understood as dust density. 
    The user must consider the gas-to-dust ratio manually.
    
    Due to the grid and plausibility issues, the function returns a zero density 
    array by default. Unit of the returned array is [gram/cm^3] of dust.
    
    Use this function with care!
    
    Parameters
    ----------
    grid : radmc3dPy.grid object
           Initialized with the model boundaries.
    r0    : float
            Radial coordinate of the inner edge, in unit of cm. Default is 0.0
    r1    : float
            Radial coordinate of the outer edge, in unit of cm. Default is 0.0
    H0    : float
            Height of slab above mid-plane at inner edge (r=r0), in unit of cm. 
            Default is 0.0.
    H1    : float
            Height of slab above mid-plane at outer edge (r=r1), in unit of cm. 
            If H1 = 0.0 and H0 >= 0.0, then H1 = H0, the slab is rectangular. 
            Default is 0.0.
    rho0  : float
            Dust mass density of the slab in units of g cm^-3. Default is 0.0.
    sig0  : float, optional
            Dust surface density of slab. If set the density is computed from 
            sig0 and the slab height Hz. The unit is g cm^-2. Default is None.
    smoothed : bool, optional
            If True then the vertical density distribution is smoothed with a 
            gaussian (standard deviation of H(r)). The surface density remains 
            the same regardless the parameter, however the density becomes a 
            function of z height. This helps to remove abrupt edges in the 
            density distribution, which appear because of the cartesian slab 
            geometry expressed in spherical coordinates.
            Default value is True.
    '''
    
    rr, th = np.meshgrid(grid.x, grid.y, indexing='ij')
    zz   = rr * np.cos(th)
    rcyl = rr * np.sin(th)    
    
    # Set rectangular case
    if H1 == 0.0:
        H1 = H0
        
    # Compute height slope
    if r1 > 0.0:
        slope = (H1 - H0) / (r1 - r0)
    else:
        slope = 0.0
    H = H0 + slope * rcyl
    
    if sig0 is not None:
        H_re = H.reshape((grid.nx, grid.ny, grid.nz))
        rho0 = np.ones([grid.nx, grid.ny, grid.nz], dtype=np.float64)
        rho = rho0 * sig0 / ( np.sqrt(2.*np.pi) * H_re)
    else:
        rho = np.ones([grid.nx, grid.ny, grid.nz], dtype=np.float64) * rho0
    
    rho = rho * np.exp(-0.5 * zz.reshape((grid.nx, grid.ny, grid.nz))**2 / H_re**2)
    
    # Set criterion
    crit_rad = ((rcyl >= r0) & (rcyl < r1))
    crit_z   = zz <= H
    crit = crit_rad #& crit_z
    
    # Get density
    dummy = np.zeros([grid.nx, grid.ny, grid.nz], dtype=np.float64)
    dummy[crit] = rho[crit]
    
    rho_slab = np.zeros([grid.nx, grid.ny, grid.nz,1], dtype=np.float64)
    rho_slab[:,:,:,0] = dummy
    
    return rho_slab
    
def slab_wrapper(grid, ppar):
    '''
    Extracts slab parameters from the ppar structure and generates the density 
    distribution. The density distribution, mass and surface density is returned.
    
    The slab is described by the following parameters (see docstring of slab 
    function): r0 (inner radius), r1 (outer radius), H0 (height at r0), H1 
    (height at r1), rho0 (constant density) or sig0 (constant surface density).
    
    The parameters are extracted from ppar as follows:
    
    r0 : float
         Radial location of slab inner edge. Set by 'r0_slab' keyword. If 
         not set, then r0 = 0.01au.
    r1 : float 
         Radial location of slab outer edge. Set by 'r1_slab' keyword. If not 
         set, then r1 = 1.0au.
    H0 : float
         Slab height at inner edge. Set by the hr0_slab keyword. If keyword is 
         not set, then H0 = H1.
    H1 : float
         Slab height at outer edge. Set by the hr1_slab keyword. If keyword is 
         not set, then H1 = r1.
    sig0 : float
         Constant surface density of the slab. Set by sig0_slab parameter. If it 
         is not set, then sig0 = m_slab / A_d, where A_d is the disk surface 
         area.
    rho0 : float
         Constant dust density of the slab. Set by rho0_slab parameter. If it is 
         not set, then rho0 = 0.0. Note that if sig0_slab and rho0_slab are both 
         set, then sig0_slab is used.
    '''
    params = ppar.keys()
    
    print ('Adding slab() component to the density distribution.\n'+
           'This is not recommended for general use, use it with care!')
    
    if ('r0_slab' in params):
        r0 = ppar['r0_slab']
    else:
        print ("WARN: r0_slab not set, using 0.01 au.")
        r0 = 0.01 * radmc3dPy.natconst.au
    if ('r1_slab' in params):
        r1 = ppar['r1_slab']
    else:
        print ("WARN: r1_slab not set, using 1.0 au.")
        r1 = 1.0 * radmc3dPy.natconst.au
    if ('h1_slab' in params):
        H1 = ppar['h1_slab']
    else:
        print ("WARN: h1_slab not set, using h1_slab = r1_slab.")
        H1 = r1
    if ('h0_slab' in params):
        H0 = ppar['h0_slab']
    else:
        print ("WARN: h0_slab not set, using h0_slab = h1_slab.")
        H0 = H1
    if ('sig0_slab' in params):
        sig0 = ppar['sig0_slab']
    elif ('m_slab' in params):
        print ("WARN: sig0_slab not set, computing from m_slab.")
        sig0 = ppar['m_slab'] / np.pi / (r1 - r0)**2
    else:
        print ("WARN: sig0_slab and m_slab not set, setting sig0 = 0.0.")
        sig0 = None
    if ('rho0_slab' in params):
        rho0 = ppar['rho0_slab']
    else:
        print ("WARN: rho0_slab not set, setting rho0 = 0.0.")
        rho0 = 0.0
    
    # Compute mass:
    if sig0 is not None:
        mass = sig0 * np.pi * (r1 - r0)**2
        sig = sig0
    elif rho0 is not None:
        # Note that this assumes H1 >= H0
        mass = rho0 * np.pi * (r1 - r0)**2 * (H0 + (H1-H0)/2.0)
        # Note that this assumes Sigma is constant with r
        sig = mass / np.pi / (r1 - r0)**2
    else:
        mass = 0.0
        sig = None

    rho_slab = slab(grid, r0=r0, r1=r1, H0=H0, H1=H1, rho0=rho0, sig0=sig0,
                    smoothed=True)
    
    return (rho_slab, mass, sig)
    
def flaring_disk(grid, ppar):
    '''
    Returns flaring disk dust density distribution. The distribution is described 
    by the following equations:
    
       hp(r) = hr0 * (r/r0)**plh * r
       
       sigma(r) = sigma0 * (r/r0)**plsig
       
       rho(r,z) = sigma(r) / (hp(r) * sqrt(2*pi) * exp(-0.5 * z**2/hp**2)
    
    Unit of the returned array is [gram/cm^3] of dust.
    
    Parameters
    ----------
    grid : radmc3dPy.grid object
           Initialized with the model boundaries.
    ppar : dict
           Dictionary provided by modPar.ppar (radmc3dPar object)
    '''
    rr, th = np.meshgrid(grid.x, grid.y, indexing='ij')
    z0 = np.zeros([grid.nx, grid.ny, grid.nz], dtype=np.float64)
    zz   = rr * np.cos(th)
    rcyl = rr * np.sin(th)
    
    # Disk density array
    rho_disk = np.zeros([grid.nx, grid.ny, grid.nz,1], dtype=np.float64)
    
    # Calculate the pressure scale height as a function of r, phi
    hp = np.zeros([grid.nx, grid.ny, grid.nz], dtype=np.float64)
    
    hp[:,:,0] = ppar['hrdisk'] * (rcyl/ppar['hrpivot'])**ppar['plh'] * rcyl

    # Determine Sigma(r)
    sigma = np.zeros([grid.nx, grid.ny, grid.nz], dtype=np.float64)
    
    if ('rsig' in ppar.keys()):
        rsig = ppar['rsig']
    else:
        rsig = ppar['rdisk']
    if ('sig0' in ppar.keys()):
        sig0 = ppar['sig0']
    else:
        sig0 = 1.0  # we normalise it later to the requested mass

    dum1 = sig0 * (rcyl/rsig)**ppar['plsig1']

    # Broken power law outside of rdisk
    crit = ( rcyl > ppar['rdisk'] )
    sig_rdisk = sig0 * (ppar['rdisk']/rsig)**ppar['plsig1']
    if ppar['plsig2'] > -20.0:
        dum1[crit] = sig_rdisk * (rcyl[crit]/ppar['rdisk'])**ppar['plsig2']
    else:
        dum1[crit] = 0.0
    
    # Within Rin
    crit = ( rcyl < ppar['rin'] )
    dum1[crit] = 0.0
    
    # Adding the smoothed inner rim
    if ('srim_rout' in ppar.keys()) and ('srim_plsig' in ppar.keys()):
        sig_srim = 1.0 * (ppar['srim_rout']*ppar['rin'] / rsig)**ppar['plsig1']
        dum2 = sig_srim * (rcyl / (ppar['srim_rout']*ppar['rin']))**ppar['srim_plsig']
        p = -5.0
        dum = (dum1**p + dum2**p)**(1./p)
    else:
        dum = dum1

    sigma[:,:,0] = dum
    
    # Calculate rho_disk(r,theta) density
    dum = sigma[:,:,0] / (hp[:,:,0] * np.sqrt(2.0*np.pi)) * np.exp(-0.5 * zz[:,:]**2 / hp[:,:,0]**2)
    
    # Copy to all z coordinates
    for iz in range(grid.nz):
        rho_disk[:,:,iz,0] = dum

    # Calculate the volume of each grid cell
    vol  = grid.getCellVolume()

    # Calculate the mass in rho_disk and scale the density to get back the
    # desired disk mass (['mdisk'] parameter) *only if ['mdisk'] is set*:
    if ('mdisk' in ppar.keys()):
        mass = (rho_disk[:,:,:,0]*vol).sum(0).sum(0).sum(0)
        rho_disk = rho_disk * (ppar['mdisk']*0.5/mass) 
        # Note that: * 0.5 -> we only consider the upper half of the disk
        # Recompute the disk mass for consistency check

    mass = (2.0*rho_disk[:,:,:,0]*vol).sum(0).sum(0).sum(0)

    # Determine/check Sigma(r) 
    dens = rho_disk[:,:,0,0]
    vol = vol[:,:,0]
    surf_mass = np.zeros(grid.nx)
    surf_area = np.zeros(grid.nx)
    for ix in range(grid.nx):
        lind = rcyl >= grid.xi[ix]
        uind = rcyl < grid.xi[ix+1]
        ind = np.where(lind == uind)
        surf_mass[ix] = np.sum(2.*dens[ind] * vol[ind])
        surf_area[ix] = np.pi * (grid.xi[ix+1]**2 - grid.xi[ix]**2)
    sig_rsig = np.interp(rsig,grid.x,surf_mass/surf_area)

    rhodust = np.array(rho_disk)

    return (rhodust, mass, sig_rsig)

def computeEnvMass(grid, rho_env):
    '''
    Compute envelope total dust mass and dust mass within 3000 au.
    
    Parameters
    ----------
    grid : radmc3dPy.grid object
           Initialized with the model boundaries.
    rho  : array_like, float
           Density distribution of an envelope ([nx, ny, nz, 1] dimension)
    '''
    au = radmc3dPy.natconst.au
    vol  = grid.getCellVolume()
    
    mass = (2.0*rho_env[:,:,:,0]*vol).sum(0).sum(0).sum(0)
    mass3000 = ( (2.0*rho_env[:,:,:,0]*vol
                  )[(grid.x<=3000.*au),:,:]).sum(0).sum(0).sum(0)
    
    return (mass, mass3000)
    
def ISradField(grid, ppar, G=1.7, show=False):
    '''
    Computes the spectral energy distribution of the Interstellar Radiation 
    Field, based on Draine (1978) and Black (1994) prescription.
    If requested then written to RADMC-3D external_source.inp file and 
    displayed on screen.

    Note that G only scales the FUV component, described by Draine and not 
    the both components.
    
    Parameters
    ----------
    grid : radmc3dPy.grid object
           Initialized with the model boundaries.
    ppar : dict
           Dictionary provided by modPar.ppar (radmc3dPar object)
    ppar : float
           Scale parameter of the interstellar radiation field. Default is G0, 
           the Draine field.
    '''
    
    if 'G' in ppar.keys():
        G = ppar['G']
    
    cc = radmc3dPy.natconst.cc   # speed of light in cgs
    kk = radmc3dPy.natconst.kk   # Boltzmann constant in cgs
    hh = radmc3dPy.natconst.hh   # Planck constant in cgs

    erg2eV = 6.242e11  # conversion from electronvolt to erg
    
    # Get the wavelength and frequency ranges of the model from grip object
    wav = grid.wav
    nwav = grid.nwav
    freq = grid.freq
    nfreq = grid.nfreq
    eV = hh * freq * erg2eV   # erg -> eV

    # Create the black body components of the Black (1994) radiation field:
    # see: https://ui.adsabs.harvard.edu/#abs/1994ASPC...58..355B/abstract
    p = [0.0, 0.0, 0.0, 0.0, -1.65, 0.0]
    T = [7500.0,4000.0,3000.0,250.0,23.3,2.728]
    W = [1.0e-14, 1.0e-13, 4.0e-13, 3.4e-09, 2.0e-4, 1.0]
    lp = [0.4,0.75,1.0,1.0,140.0,1060.0]
    
    IBlack = np.zeros(nwav)

    for i in range(len(p)):
        exppart = np.array(hh * freq / kk / T[i])
        exppart= np.clip(exppart,0,500)
        IBlack = IBlack + ( 2.*hh*freq**3. / cc**2. * (wav / lp[i])**p[i]    \
                * W[i] / (np.exp(exppart, dtype=np.float64)-1))
    # don't worry about the warnings about division by zero, 
    # it comes from the (np.exp(hh * freq / kk / T[i])-1) part.
    
    # Create the Draine (1978) radiation field using his original formula, 
    # see eq. 11 in https://ui.adsabs.harvard.edu/#abs/1978ApJS...36..595D/abstract
    # in photons cm^-2 s^-1 sr^-1 eV^-1
    IDraineEv = (1.658E6 * eV) - (2.152e5 * eV**2) + (6.919E3 * eV**3)
    # in erg cm^-2 s^-1 sr^-1 Hz^-1
    IDraineErg = IDraineEv * hh**2 * freq * erg2eV                      
    
    # scale the FUV Draine field
    IDraine = IDraineErg * G/1.7
    # The formula was designed on the 5 eV (0.24 micron) to 13.6 eV (0.09 micron) range,
    # limit the calculated intensities
    IDraine[wav < 0.09117381] = 0.0
    IDraine[wav > 0.24799276] = 0.0
    IBlack[wav < 0.24799276] = 0.0           # limit the Black field as well below 0.24 micron

    # Combine the expressions for the different wavelength ranges:
    Iisrf = IBlack + IDraine
    #
    # Plot the results if asked
    #
    # Measurements to overplot from Allen book ISRF, unit: erg s-1 cm-2 mum-1 
    if show:
        wavObs = np.array([0.091, 0.1, 0.11, 0.13, 0.143, 0.18,  \
                            0.2, 0.21, 0.216, 0.23, 0.25, 0.346,  \
                            0.435, 0.55, 0.7, 0.9, 1.2, 1.8, 2.2, \
                            2.4, 3.4, 4, 5, 12, 25, 60, 100, 200, \
                            300, 400, 600, 1000])
        freqObs_hz = cc / (wavObs / 1e4)
        FlamObs  = np.array([1.07e-2, 1.47e-2, 2.04e-2, 2.05e-2, 1.82e-2, \
                            1.24e-2, 1.04e-2, 9.61e-3, 9.17e-3, 8.25e-3, \
                            7.27e-3, 1.3e-2, 1.5e-2, 1.57e-2, 1.53e-2,   \
                            1.32e-2, 9.26e-3, 4.06e-3, 2.41e-3, 1.89e-3, \
                            6.49e-4, 3.79e-4, 1.76e-4, 1.7e-4, 6.0e-5,   \
                            4.6e-5, 7.3e-5, 2.6e-5, 5.4e-6, 1.72e-6,     \
                            3.22e-6, 7.89e-6])
        InuObs = (wavObs / 1e4)**2. / cc * FlamObs / (4.*3.14) * 1e4
        # Plot   
        plt.xscale('log')
        plt.yscale('log')
        plt.plot(wavObs, InuObs * freqObs_hz, 'ro')
        plt.plot(wav, Iisrf * freq, color='black')
        plt.xlabel('wavelength [$\mu$m]')
        plt.ylabel('intensity [erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$ sr$^{-1}$]')
        plt.show()
  
    return Iisrf
