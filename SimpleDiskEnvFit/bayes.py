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

from . import main
import radmc3dPy.natconst as nc
from galario import deg, arcsec

import numpy as np
import os

def lnpriorfn(p, par_ranges):
    """
    Uniform prior probability function.
    
    If p (parameters) are outside of parameter range set prior, then return -Inf,
    else return negative scalar.
    
    Parameters
    ----------
    p : list
        Prior probability is returned for these parameters. Normally this is 
        provided by the emcee sampler.
    par_ranges : list or list of lists
        Prior constraints. Assumption of parameter ranges. Should contain a 
        range for each parameters.
    """

    for i in range(len(p)):
        if p[i] < np.min(par_ranges[i]) or p[i] > np.max(par_ranges[i]):
            return -np.inf

    jacob = -p[0]       # jacobian of the log transformation

    return jacob

def lnpostfn(p, p_ranges, parname, modpar, resource_dir, uvdata,
             dpc=1.0, incl=45., PA=0.0, dRA=0.0, dDec=0.0,
             idisk=True, ienv=True, icav=False, impar=None, 
             verbose=False, cleanModel=False, binary=False,
             chi2_only=True, galario_check=False, time=False):
    """
    Log of posterior probability function.
    
    Parameters
    ----------
    p   :   list
            Prior probability is returned for these parameters. Normally this is 
            provided by the emcee sampler.
    par_ranges : list or list of lists
            Prior constraints. Assumption of parameter ranges. Should contain a 
            range for each parameters.
    parname : list of string
            Names of fitted parameters following the radmc3dPy.modPar and 
            getParam() conventions. This has to have the same number of elements 
            as p.
    modpar : radmc3dPy.radmc3dPar class object
            Containing the base parameter values. These are common in all models 
            within a run. The fitted parameters are updated according the values 
            of p.
    resource_dir : sting
            Path (absolute or relative) to the folder containing additional 
            files (e.g. dust opacity or lnk files) that are needed to create 
            the model. Defaults to {SIMPLEDISKENVFIT_HOME}/lnk_files.
    uvdata : dict or list of dict 
            Containing observed visibility data. The 'u', 'v', 'Re', 'Im', 
            'w' and 'wav' keywords need to be defined.
    dpc  :  float
            Distance to object in unit of parsec, Default is 1.0.
    incl :  float
            Model inclination in image, Default is 45.0.
    PA   :  float, optional
            Position angle in radian. Default is 0.0.
    dRA  :  float, optional
            Offset in RA in radian. Default is 0.0.
    dDec :  float, optional
            Offset in Dec in radian. Default is 0.0.
    idisk : bool
            Include disk component in model. Default is True.
    ienv  : bool
            Include envelope component in model. Default is True.
    icav  : bool
            Include envelope cavity in model. Default is False.
    impar : dict or list of dict, optional
            Image parameter(s). Known keywords are listed in the runImage()
            method description. At least the wavelength (wav keyword) must 
            be set for each images. Default is None.
    verbose : bool, optional
            If True, then print summary of model parameters to standard 
            output. Runtime INFO messages are also printed to standard 
            output. Default is False.
    cleanModel : bool, optional
            If True, then delete the RADMC-3D model folder from disk after the 
            posterior probability estimation. In this case model files are not 
            stored.
    binary  : bool, optional
            If True, then RADMC3D will use binary I/O, if False then use 
            ASCII I/O. Binary I/O may improve computation speed and reduce 
            disk space usage when models are kept (i.e. cleanModel is not 
            called).
    chi2_only : bool
            If True then the synthetic visibility itself is not computed and 
            stored (a zero value array is stored instead). The chi2 is still 
            computed and stored. Set this to True when running MCMC fitting 
            in order to improve speed. Default is True.
    galario_check : bool
            Check whether image and dxy satisfy Nyquist criterion for 
            computing the synthetic visibilities in the (u, v) locations 
            provided (see galario documentation). Default is False.
    time :  bool, optional
            Prints function runtime information. Useful for profiling.
            Default is False.
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
        elif parname[i] in ['rdisk','r0Env','rTrunEnv']:
            val = 10**p[i] * nc.au
        elif parname[i][0:3] == 'gsm': # gsmin or gsmax
            val = 10.0**p[i]
        else:
            val = p[i]
        
        # If rTrunEnv is not a fit parameter then set it equal to rdisk.
        if parname[i] == 'rdisk':
            modpar.setPar("rTrunEnv", "{}".format(val))

        # Set the model parameters
        if parname[i] in modpar.ppar.keys():
            if type(val) is list:
                modpar.setPar([parname[i], "{}".format(val)])
            else:
                modpar.setPar([parname[i], "{:10.6E}".format(val)])
        elif parname[i] =='gsmax_env':
            tmp = modpar.ppar['gsmax']
            tmp[1] = val
            modpar.setPar(['gsmax', "{}".format(tmp)])
        elif parname[i] =='gsmax_disk':
            tmp = modpar.ppar['gsmax']
            tmp[0] = val
            modpar.setPar(['gsmax', "{}".format(tmp)])
        elif parname[i] =='gsmin_env':
            tmp = modpar.ppar['gsmin']
            tmp[1] = val
            modpar.setPar(['gsmin', "{}".format(tmp)])
        elif parname[i] =='gsmin_disk':
            tmp = modpar.ppar['gsmin']
            tmp[0] = val
            modpar.setPar(['gsmin', "{}".format(tmp)])
        elif parname[i] == 'dpc':
            dpc = val
        elif parname[i] == 'incl':
            incl = val
        elif parname[i] == 'PA':
            PA = val * deg
        elif parname[i] == 'dRA':
            dRA = val * arcsec
        elif parname[i] == 'dDec':
            dDec = val * arcsec
        else:
            raise ValueError('ERROR [lnpostfn]: unknown \
                parameter [{}]'.format(parname[i]))

        if verbose:
            print ("INFO [{:06}]: {:s} is set to {}".format(rand,
                                                            parname[i],
                                                            val))

    # Generate model folder name
    model_dir = "{}/mod_".format('.')
    for i in range(len(parname)):
        model_dir = "{:s}{:.4s}{:0.3E}_".format(model_dir,parname[i],p[i])
    model_dir = "{:s}{:06}".format(model_dir,rand)

    if verbose:
        print ('INFO [{:06}]: saving model to {:s}'.format(rand, model_dir))

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
    
    # Check uvdata


    # compute the model brightness profile    
    mod = main.radmc3dModel(modpar=modpar, model_dir=model_dir, 
                            resource_dir=resource_dir, ID=rand,
                            binary=binary, idisk=idisk, ienv=ienv, 
                            icav=icav)
    mod.write2folder()

    mod.runModel(impar=impar, mctherm=True, nphot_therm=100000, verbose=verbose,
                 time=time)

    # Use the correct distance
    if type(impar) == list:
        dpc_vis = impar[0]['dpc']
    else:
        dpc_vis = impar['dpc']
    mod.getVis(uvdata, dpc=dpc_vis, PA=PA, dRA=dRA, dDec=dDec, chi2_only=
               chi2_only, galario_check=galario_check, time=time, 
               verbose=verbose)

    # Delete model folder from disk if requested
    if cleanModel:
        mod.cleanModel()

    chi2 = -0.5 * np.sum(mod.chi2/mod.nvis) + lnprior
    
    if verbose:
        print ("INFO [{:06}]: model ch^2 = {:10.6E}".format(rand, chi2))

    return chi2
