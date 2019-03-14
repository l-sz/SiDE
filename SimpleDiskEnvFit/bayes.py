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

import SimpleDiskEnvFit.main as main
import radmc3dPy.natconst as nc

import numpy as np
import os

def lnpriorfn(p, par_ranges):
    """
    Uniform prior probability function
    """

    for i in range(len(p)):
        if p[i] < np.min(par_ranges[i]) or p[i] > np.max(par_ranges[i]):
            return -np.inf

    jacob = -p[0]       # jacobian of the log transformation

    return jacob

def lnpostfn(p, p_ranges, parname, ppar, main_dir, visdata, 
             dpc=1.0, incl=45., impar=None, verbose=False):
    """
    Log of posterior probability function
    """
    # Model ID
    rand = np.random.randint(0,99999)

    lnprior = lnpriorfn(p, p_ranges)  # apply prior
    if not np.isfinite(lnprior):
        return -np.inf

    if len(p) != len(parname):
        raise ValueError('ERROR [lnpostfn()]: len(p) [{}] != \
                         len(parname) [{}]'.format( len(p), len(parname) ))

    # Update parameters
    for i in range(len(parname)):
        
        # Special cases
        if parname[i] == 'mdisk':
            val = 10.0**p[i] * nc.ms
        elif parname[i] == 'rho0Env':
            val = 10**p[i]
        elif parname[i] == 'rdisk':
            val = p[i] * nc.au
        elif parname[i][0:6] == 'agrain':
            val = 10.0**p[i]
        else:
            val = p[i]
        
        # Set the model parameters
        if parname[i] in ppar.ppar.keys():
            ppar.setPar([parname[i], "{:10.6E}".format(val)])
        elif parname[i] =='agrain_env':
            tmp = ppar.ppar['agraincm']
            tmp[1] = val
            ppar.setPar(['agraincm', "{}".format(tmp)])
        elif parname[i] =='agrain_disk':
            tmp = ppar.ppar['agraincm']
            tmp[0] = val
            ppar.setPar(['agraincm', "{}".format(tmp)])
        elif parname[i] == 'dpc':
            dpc = val
        elif parname[i] == 'incl':
            incl = val
        else:
            raise ValueError('ERROR [lnpostfn]: unknown \
                parameter [{}]'.format(parname[i]))
        
        if verbose:
            print ("INFO [{:06}]: {:s} is set to {}".format(rand,
                                                            parname[i],
                                                            val))
    
    # Generate model folder name
    folder = "{}/mod_".format('.')
    for i in range(len(parname)):
        folder = "{:s}{:.2s}{:10.6E}_".format(folder,parname[i],p[i])
    folder = "{:s}{:06}".format(folder,rand)

    if verbose:
        print ('INFO [{:06}]: saving model to {:s}'.format(rand, folder))

    # Check/set image parameters
    if impar is None:
        impar = {'wav':1000.,'dpc':dpc,'incl':incl}
    else:
        if type(impar) == list:
            for ip in impar:
                ip['dpc'] = dpc
                ip['incl'] = incl
        else:
            impar['dpc'] = dpc
            impar['incl'] = incl

    if verbose:
        print ('INFO [{:06}]: using dpc={} and incl={}'.format(rand, dpc, incl))
    
    # Check visdata


    # compute the model brightness profile
    
    mod = main.radmc3dModel(modpar=ppar, folder=folder, main_dir=main_dir, 
                            ID = rand)
    mod.write2folder()

    mod.runModel(impar=impar, mctherm=True, nphot_therm=100000, verbose=verbose)
    # Use the correct distance
    if type(impar) == list:
        dpc_vis = impar[0]['dpc']
    else:
        dpc_vis = impar['dpc']
    mod.getVis(visdata, dpc=dpc_vis)

    chi2 = -0.5 * np.sum(mod.chi2) + lnprior
    
    if verbose:
        print ("INFO [{:06}]: model ch^2 = {:10.6E}".format(rand, chi2))

    return chi2
