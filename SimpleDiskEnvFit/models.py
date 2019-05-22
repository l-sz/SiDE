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
    Returns density distribution ([nx, ny, nz, 1] dimension) of a rotationally 
    flattened protostellar envelope. The profile is given by Ulrich (1976).
    The model parameters should be provided in ppar.
    
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
    z0 = np.zeros([grid.nx, grid.nz, grid.nz], dtype=np.float64)
    zz   = rr * np.cos(th)
    rcyl = rr * np.sin(th)

    # Envelope density array
    rho_env = np.zeros([grid.nx, grid.ny, grid.nz,1], dtype=np.float64)

    dummy = uenv.ulrich_envelope(rr, th, rho0=ppar['rho0Env'] / 
                                            ppar['dusttogas'], 
                                            rmin=ppar['rTrunEnv'], 
                                            Rc=ppar['r0Env'])
    
    rho_env[:,:,0,0] = dummy
    
    if cavity:
        
        rho_env = envelope_cavity(rho_env, grid, ppar)

    # Calculate the volume of each grid cell
    mass, mass3000 = computeEnvMass(grid, rho_env)
    
    return (rho_env, mass, mass3000)

def tafalla_envelope(grid, ppar, cavity=False):
    '''
    Returns protostellar envelope density distribution ([nx, ny, nz, 1] 
    dimension) according Tafalla et al. (2004). The model parameters should be
    provided in ppar.
    
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
    z0 = np.zeros([grid.nx, grid.nz, grid.nz], dtype=np.float64)
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
    Returns power law protostellar envelope density distribution 
    ([nx, ny, nz, 1] dimension). The model parameters should be provided in 
    ppar.
    
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
    z0 = np.zeros([grid.nx, grid.nz, grid.nz], dtype=np.float64)
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
    z0 = np.zeros([grid.nx, grid.nz, grid.nz], dtype=np.float64)
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
    elif 'tetDegCav' in ppar.keys():
        theta_cav_deg = ppar['tetDegCav']
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

def flaring_disk(grid, ppar):
    '''
    Returns flaring disk density distribution. The distribution is described by 
    the following equations:
    
       hp(r) = hr0 * (r/r0)**plh * r
       
       sigma(r) = sigma0 * (r/r0)**plsig
       
       rho(r,z) = sigma(r) / (hp(r) * sqrt(2*pi) * exp(-0.5 * z**2/hp**2)
    
    Parameters
    ----------
    grid : radmc3dPy.grid object
           Initialized with the model boundaries.
    ppar : dict
           Dictionary provided by modPar.ppar (radmc3dPar object)
    '''
    rr, th = np.meshgrid(grid.x, grid.y, indexing='ij')
    z0 = np.zeros([grid.nx, grid.nz, grid.nz], dtype=np.float64)
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

    rhodust = np.array(rho_disk) * ppar['dusttogas']

    return (rhodust, mass, sig_rsig)

def computeEnvMass(grid, rho_env):
    '''
    Compute envelope total mass and mass within 3000 au.
    
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
    p = [0., 0., 0., 0., -1.65, 0.]
    T = [7500.,4000.,3000.,250.,23.3,2.728]
    W = [1e-14, 1e-13, 4e-13, 3.4e-09, 2e-4, 1.]
    lp = [0.4,0.75,1.,1.,140.,1060.]
    
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
  
