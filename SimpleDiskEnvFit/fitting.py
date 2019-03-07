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

def lnpriorfn(p, par_ranges):
    """
    Uniform prior probability function
    """

    for i in range(len(p)):
        if p[i] < np.min(par_ranges[i]) or p[i] > np.max(par_ranges[i]):
            return -np.inf

    jacob = -p[0]       # jacobian of the log transformation

    return jacob

def lnpostfn(p, p_ranges, ppar, main_dir, dpc, u, v, Re, Im, w):
    """
    Log of posterior probability function
    """

    lnprior = lnpriorfn(p, p_ranges)  # apply prior
    if not np.isfinite(lnprior):
        return -np.inf

    # unpack the parameters
    mdisk, rhoenv = p
    
    rand = np.random.randint(0,30000+1)

    folder = "{}/mod_md{:10.6E}_re{:10.6E}_{:06}".format(main_dir.strip(), mdisk, rhoenv, rand)

    mdisk = 10.**mdisk        # convert from log to real space
    rhoenv = 10.**rhoenv
    
    print (ppar.ppar['mdisk'], ppar.ppar['rho0Env'] )
    
    ppar.setPar(['mdisk', '{:06.2E}*ms'.format(mdisk), ' Disk mass', 'Disk parameters'])
    ppar.setPar(['rho0Env', '{:06.2E}'.format(rhoenv), ' New central density g/cm^3 dust density volume', 
                 'Envelope parameters'])

    print (ppar.ppar['mdisk'], ppar.ppar['rho0Env'] )

    # compute the model brightness profile
    
    print ("\n{}\n".format(folder))
    
    mod = radmc3dModel(modpar=ppar, folder=folder, main_dir=main_dir, ienv=True, idisk=True)
    mod.write2folder()

    impar = [{'npix':512,'wav':1100.,'sizeau':6000,'incl':60},
             {'npix':512,'wav':3000.,'sizeau':6000,'incl':60}]
    
    visdata = {'u':u, 'v':v, 'Re':Re, 'Im':Im, 'w':w, 'wav':1100.}
    
    
    mod.runModel(impar=impar, mctherm=True, nphot_therm=100000)
    mod.getVis(visdata, dpc=dpc)

    chi2 = mod.chi2
    
    print (-0.5 * chi2 + lnprior, chi2, lnprior)

    return -0.5 * chi2 + lnprior
