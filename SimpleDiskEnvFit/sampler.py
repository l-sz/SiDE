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
import sys
import pickle
import numpy as np

from emcee import EnsembleSampler
from emcee.utils import MPIPool
import galario
import radmc3dPy.natconst as nc

from . import bayes
from . import main
from . import tools

__all__ = ['run_mcmc']

def run_mcmc(main_dir, uvdata, paramfile='model_param.inp', nthreads=8, 
             nwalkers=40, nsteps=300, nburnin=100, use_mpi=False, verbose=False, 
             resume=False, sloppy=False, chain_file='chain.dat', 
             restart_file='chain.dat', impar=None, parname=None, p_ranges=None, 
             p0=None, kwargs=None):
    '''
    Computes posteriori probabilities of parametrised Class 0/I models given 
    a set of observational constraints using SimpleDiskEnvFit.

    The observational constraints and parameter ranges are hard-coded in the 
    function. The code assumes that elias29_params.inp, Elias29uvt_270.txt 
    and Elias29uvt_94.txt are in the folder where the function is executed.

    The function save the MCMC chains, parameter ranges, constraints and 
    metadata to a binary pickle file (elias29_mcmc_save.p) and displays the 
    posterior likelihood distribution in a "corner plot" (elias29_mcmc.pdf).

    Parameters
    ----------
    main_dir : string
             Directory containing input and output files
    nthreads : int
             Number of threads used in multiprocessing mode. This is ignored 
             in MPI mode. Default is 8.
    nwalkers : int
             Number of walkers used. Default is 40.
    nsteps   : int
             Number of walker steps in the main run. Default is 300.
    nburnin  : int
             Number of walker steps in initial "burn-in" run. Default is 100.
    use_mpi  : bool
             Use MPI pools instead of python threads. Useful for running 
             on computer clusters using multiple nodes. Default is False.
    resume   : bool
             If True then resume MCMC chain from file and continue sampling 
             the posterior distribution. nwalkers should not change between 
             runs. Default is False.
    sloppy   : bool
             If True then RADMC-3D relaxed the sub-pixel refinement criterion 
             when raytracing. This may reduce runtime. Please don not forget to 
             check best fit models with higher accuracy before publishing results.
             Default is False.
    chain_file : string, optional
             Chain (parameters and probabilities) are stored in this file. 
             The file can be used to restart or continue MCMC sampling.
             Meaning of columns: walker index (1), parameter value (n), 
             log probability (1).
             If file already exists then output is automatically renamed, 
             this is done in order not to overwrite previous results and 
             the restart_file (see below).
             Default is "chain.dat".
    restart_file: string, optional
             When restarting (resume = True), then results from previous 
             run are read from restart_file. If resume parameter is set 
             True, then file must exist.
             Default is "chain.dat".
    '''
    if use_mpi:
        pool = MPIPool()
        if not pool.is_master():
            os.chdir(main_dir)
            pool.wait()
            sys.exit(0)
    else:
        pool = None

    # Change to main_dir in order to find the UV obs. data files
    current_dir = os.path.realpath('.')
    os.chdir(main_dir)

    print ("INFO [{:06}]: nthreads [{}], nwalkers [{}], nburnin [{}], nsteps [{}]".format(0, 
                                             nthreads, nwalkers, 
                                             nburnin, nsteps))
    print ("INFO [{:06}]: USE_MPI is {}".format(0, use_mpi))

    # Check for minimum function call arguments
    if not uvdata:
        raise ValueError('uvdata must be provided!')
    
    if paramfile is None:
        print ('WARN [{:06}]: paramfile not provided, using defaults!'.format(0))
    
    if impar is None:
        print ('WARN [{:06}]: impar is not provided, computing parameters from uvdata!'.format(0))

    if parname is None:
        raise ValueError('parname must be provided!')
    elif type(parname) is not list:
        raise TypeError('parname must be a list!')
        
    if p_ranges is None:
        raise ValueError('p_ranges must be provided!')
    elif type(p_ranges) is not list:
        raise TypeError('p_ranges must be a list!')
    elif len(p_ranges) != len(parname):
        raise ValueError('len(p_ranges)[{:d}] != len(parname[{:d}])'.format(
            len(p_ranges,len(parname))))
    
    if p0 is None:
        raise ValueError('p0 must be provided!')

    # Read parameter file 
    par = main.getParams(paramfile=paramfile)

    # Set parameters for bayes.lnpostfn() function
    if kwargs is None:
        kwargs = {'dpc': 125., 'incl': 60., 'verbose': verbose, 
                'PA':0.0, 'dRA':0.0, 'dDec':0.0,
                'idisk':True, 'ienv':True, 'icav':False, 'islab':False,
                'cleanModel': True, 'binary': True, 'chi2_only':True, 
                'galario_check':False, 'time':True }

    # Set image parameters if not provided
    if impar is None:
        impar = []
        
        if type(uvdata) is dict:
            uvdata = [uvdata]
            
        for dset in uvdata:
            wav_ = dset['wav']
            wav_m = wav_ * 1.0e-6
            npix_, dpix_ = galario.double.get_image_size(dset['u']/wav_m, 
                                                         dset['v']/wav_m)
            dpix_au = dpix_ / galario.arcsec * kwargs['dpc'] 
            sizeau_ = npix_ * dpix_au

            impar.append({'npix':npix_, 'wav':wav_, 'sizeau':sizeau_, 
                          'incl':kwargs['incl']})
            print ('''INFO [{:06}]: visibility dataset found: npix = {}, sizeau = {:.2f}, wav = {:.2f}'''.format(0,npix_,sizeau_,wav_))

    # Set sloppynes
    for ip in impar: ip['sloppy'] = sloppy

    # Update kwargs keys if needed
    kwargs['verbose'] = verbose
    kwargs['impar'] = impar

    # Number of fitted parameters
    ndim = len(p_ranges)

    # initialize the walkers with an ndim-dimensional Gaussian ball
    if resume:
        resume_data = tools.read_chain_ascii(restart_file)
        if nwalkers != resume_data.nwalkers:
            raise ValueError('ERROR: walker number does not match resume file.')
        pos = []
        for pv in resume_data.chain[:,-1,:]:
            pos.append(pv)
        lnprob0 = resume_data.lnprob[:,-1]
    else:
        pos = [p0 + 1.0e-2*np.random.randn(ndim) for i in range(nwalkers)]
        lnprob0 = None

    nsteps += nburnin   # set total steps in chain

    ## Create chain_file, if already exists then rename output
    while os.path.isfile(chain_file):
        counter = 0
        selem = chain_file.split('.')
        main_name = ''.join(selem[0:-1])
        try:
            selem2 = main_name.split('_')
            counter = int(selem2[-1]) + 1
            main_name = ''.join(selem2[0:-1])
        except:
            pass
        chain_file = '{}_{}.{}'.format(main_name,counter,selem[-1])

    # Create chain file and write header
    f = open(chain_file, "w")
    f.write('# nwalkers = {:3d}, nthreads = {:3d}, nsteps = {:5d}, MPI = {}\n'.format(
               nwalkers, nthreads, nsteps, use_mpi))
    f.write('# i_walker {}  lnprob\n'.format(''.join(
                                        np.vectorize(" %s".__mod__)(parname))))
    f.close()

    # Create and run sampler
    sampler = EnsembleSampler(nwalkers, ndim, bayes.lnpostfn,
                          args=[p_ranges, parname, par, main_dir, uvdata],
                          kwargs=kwargs, threads=nthreads, pool=pool)

    print ("INFO [{:06}]: RUN {} main steps".format(0,nsteps))
    print ("INFO [{:06}]: status info at every 100 steps:".format(0))

    f = open(chain_file, "a")

    for i, step in enumerate(sampler.sample(pos, iterations=nsteps, 
                                            lnprob0=lnprob0)):
        position = step[0]
        lnprob = step[1]
        for k in range(nwalkers):
            posstr = ''.join(np.vectorize("%12.5E ".__mod__)(position[k]))
            f.write("{:04d} {:s}{:12.5E}\n".format(k, posstr, lnprob[k]))
        f.flush()
        # Print progress info
        if (i+1) % 100 == 0:
            print("INFO [{:06}]: {:5.1%} done".format(0,float(i+1) / nsteps))

    f.close()

    print ("INFO [{:06}]: DONE {} main steps".format(0,nsteps))

    # Close pool
    if use_mpi:
        pool.close()

    # Extract results
    chain = sampler.chain[:, :, :]
    accept_frac = sampler.acceptance_fraction
    lnprob = sampler.lnprobability

    if resume:
        chain = np.concatenate( (resume_data['chain'], chain), axis=1 )
        nsteps = nsteps + resume_data['nsteps']

    # Save results
    results = {'chain': chain, 'accept_frac':accept_frac, 'lnprob':lnprob, 
               'parname':parname, 'p_ranges':p_ranges, 'p0':p0, 'ndim':ndim, 
               'nwalkers':nwalkers, 'nthreads':nthreads, 'nsteps':nsteps, 
               'nburnin':nburnin, 'uvdata':uvdata, 'impar':impar}

    # Save chain and metadata
    # Note that protocol=2 needed for python 2/3 compatibility
    pickle.dump( results, open( 'elias29_mcmc_save.p', 'wb' ), protocol=2 )

    # Return
    os.chdir(current_dir)
    return results
