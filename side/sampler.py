# 
# Code to set up a simple RADMC-3D disk models with envelope
#
# The code heavily relies on radmc3dPy package by Attila Juhasz:
# (https://www.ast.cam.ac.uk/~juhasz/radmc3dPyDoc/index.html)
#
# Original SimpleDiskEnvFit (SiDE) source available at:
# https://github.com/l-sz/SiDE
#  
# Copyright (C) 2019 Laszlo Szucs <laszlo.szucs@mpe.mpg.de>
#
# Licensed under GLPv2, for more information see the LICENSE file repository.
#

from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import gc
import pickle
import numpy as np

import mpi4py
from emcee import EnsembleSampler
from emcee.utils import MPIPool
import galario
import radmc3dPy.natconst as nc

from . import bayes
from . import main
from . import tools

__all__ = ['run_mcmc']

def run_mcmc(main_dir, uvdata, paramfile='model_param.inp', nthreads=1, 
             nthreads_openmp=1, nwalkers=40, nsteps=300, nburnin=0, use_mpi=False, 
             loadbalance=False, verbose=False, resume=False, sloppy=False, 
             chain_file='chain.dat', restart_file='chain.dat', impar=None, 
             parname=None, p_ranges=None, p0=None, p_form=None, p_formprior=None, 
             p_sigma=None, debug=False, kwargs=None, dpc=1.0, incl=60.):
    '''
    Computes posteriori probabilities of parametrised Class 0/I models given 
    a set of observational constraints and radiative transfer model parameters.

    The observational constraints (visibility data) must be provided in the 
    uvdata argument. The radiative transfer model parameters must be provided 
    in the paramfile. The fitted parameters (specified in parname argument) 
    will be changed during the MCMC procedure. The initial values of the fitted 
    parameters are provided in the p0 argument. All other model parameters are 
    as set in the paramfile.
    
    In version 0.1.1 only uniform prior is possible (1 within the parameter range,
    0 outside of the range). The ranges of the uniform prior are given in the 
    p_ranges argument.

    The MCMC chains are saved in a Python readable pickle file (chain.p) and 
    in an ASCII file (chain.dat). The pickle file contains additional information 
    (e.g. uniform prior ranges, observational constraints, meta-data), but it is 
    only written at the successful finish of the MCMC run. The ASCII output 
    saves only the current parameters of the chains and the model likelihood, 
    but it is updated after each MCMC step. This file can be used to restart 
    runs that ended before completion (e.g. because the allocated time run out 
    on the cluster).
    
    The resulting chain and meta-data is also returned as a dictionary.

    Parameters
    ----------
    main_dir : string
             Directory containing input and output files.
    uvdata   : dictionary or list of dictionaries
             Input observed visibility data at a single or multiple wavelengths.
             The chi^2 of the models is computed compared to the uvdata datasets.
             The dictionary should contains the u, v, Re, Im, w, wav keywords. 
    paramfile: string
             Name of the radmc3dModel parameter file. The parameter file is 
             necessary. Default is 'model_param.inp'.
    nthreads : int
             Number of threads used in multiprocessing mode. In MPI mode the 
             parameter is ignored. Default is 1.
    nthreads_openmp : int
             Number of OpenMP threads used by RADMC-3D and galario. It is used 
             both in MPI and multiprocessing modes. nthreads_openmp should not 
             be larger than the total number of CPU threads. Default is 1.
    nwalkers : int
             Number of walkers used. Default is 40.
    nsteps   : int
             Number of walker steps in the main run. Default is 300.
    nburnin  : int
             Number of walker steps in initial "burn-in" run. Default is 0.
    use_mpi  : bool
             Use MPI pools instead of python threads. Useful for running 
             on computer clusters using multiple nodes. Default is False.
    loadbalance : bool
             When the MPI mode is used and the runtime of individual log-probability
             function calls vary significantly and ntask > Ncpu, then setting this
             parameter to True may improve the overall computational speed. 
             Default is False.
    verbose  : bool
             If True then write detailed information messages to the standard 
             output. Default is False.
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
    impar    : dict or list of dict, optional
             Image parameter(s). Known keywords are listed in the runImage()
             method description. At least the wavelength (wav keyword) must 
             be set for each images. Default is None.
    parname  : list, string
             Names of fitted parameters (string). The known parameter names are 
             those given in the parameter file. Required to run the fitting. 
             Default is None.
    p_range  : list of lists
             Uniform prior parameter ranges. For each parameter it should contain 
             a two element list with the minimum and maximum values. Within these 
             ranges the prior probability is 1, outside it is 0. Must have as 
             many elements at p0.
             Default is None.
    p0       : list
             Initial values of the fitted parameters. Should have exactly as many 
             elements as the parname list. The p0 values should be within the 
             corresponding p_range.
             Default is None.
    p_form   : list
             Sets whether p[i] is logarithmic (i.e. that val = 10**p[i]) or linear
             (val = p[i]). Must have as many elements as p0.
             Default is None.
    p_formprior : list
             Sets the functional form of the prior probability distribution. It 
             should be set to 'normal' or 'uniform' for a Gaussian or rectangular 
             distribution, respectively. Must have exactly as many elements as p0.
             Default is None.
    p_sigma  : list
             Width of the Gaussian function. Must have exactly as many elements 
             as p0. If p_formprior[i] is 'uniform', then p_sigma[i] is not used.
             Default is None.
    debug    : bool
             Passes debug argument to the emcee module. If set then more information 
             is written to standard output about the MPI processes.
             Default is False.
    kwargs   : dict
             Dictionary containing keyword arguments for the lnpostfn() function.
             For details see the docstring of bayes.lnpostfn(). 
             Important: the dpc, incl, PA, dRA, dDec parameters given as an 
             argument to lnpostfn() overwrite the corresponding values given in 
             impar. If kwargs is not set, then these will be overwritten by the 
             lnpostfn() default arguments!
             Default is None.
    dpc      : float
             Distance to the modelled object in parsecs. If dpc is not defined 
             in kwargs, then this value is used.
             Default is 1.0.
    incl     : float
             Inclination of the model in degrees. If incl is not a fit parameter 
             or not set in kwargs, then this value is used.
             Default is 60.
    '''
    if use_mpi:
        version= mpi4py.MPI.Get_version()
        comm = mpi4py.MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
        universe_size=comm.Get_attr(mpi4py.MPI.UNIVERSE_SIZE)
        
        print ('INFO [{:06}]: MPI: {} (version), {} (size), {} (rank), {} (universe)'.format(
                            0, version, size, rank, universe_size))
        
        pool = MPIPool(comm=comm, debug=debug, loadbalance=loadbalance)
        
        if not pool.is_master():
            os.chdir(main_dir)
            pool.wait()
            sys.exit(0)

        nthreads = pool.size
    else:
        pool = None

    # Change to main_dir in order to find the UV obs. data files
    current_dir = os.path.realpath('.')
    os.chdir(main_dir)

    print ("INFO [{:06}]: nthreads [{}], nthreads_openmp [{}], nwalkers [{}], nburnin [{}], nsteps [{}]".format(0, 
                                             nthreads, nthreads_openmp, nwalkers, 
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

    if p_form is None:
        raise ValueError('p_form must be provided!')

    if p_formprior is None:
        raise ValueError('p_formprior must be provided!')
    
    if p_sigma is None:
        raise ValueError('p_sigma must be provided!')

    # Read parameter file 
    par = main.getParams(paramfile=paramfile)

    # Set parameters for bayes.lnpostfn() function
    if kwargs is None:
        print ('WARN [{:06}]: kwargs is not provided, using defaults!'.format(0))
        print ('WARN [{:06}]: using dpc  = {:6.2f}'.format(0, dpc))
        print ('WARN [{:06}]: using incl = {:6.2f}'.format(0, incl))
        
        kwargs = {'dpc': dpc, 'incl': incl, 'verbose': verbose, 
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
    kwargs['nthreads'] = nthreads_openmp

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
    f.write('# nwalkers = {:3d}, nthreads = {:3d}, nthreads_openmp = {:3d}, nsteps = {:5d}, MPI = {}\n'.format(
               nwalkers, nthreads, nthreads_openmp, nsteps, use_mpi))
    f.write('# i_walker {}  lnprob\n'.format(''.join(
                                        np.vectorize(" %s".__mod__)(parname))))
    f.close()

    # Create and run sampler
    sampler = EnsembleSampler(nwalkers, ndim, bayes.lnpostfn,
                          args=[p_form, p_ranges, p_formprior, p0, p_sigma, 
                                parname, par, main_dir, uvdata],
                          kwargs=kwargs, threads=nthreads, pool=pool)

    print ("INFO [{:06}]: RUN {} main steps".format(0,nsteps))
    print ("INFO [{:06}]: status info at every 100 steps:".format(0))

    f = open(chain_file, "a")

    for step in sampler.sample(pos, iterations=nsteps, lnprob0=lnprob0):
        position = step[0]
        lnprob = step[1]
        # Write restart file
        for k in range(nwalkers):
            posstr = ''.join(np.vectorize("%12.5E ".__mod__)(position[k]))
            f.write("{:04d} {:s}{:12.5E}\n".format(k, posstr, lnprob[k]))
        f.flush()
        # Run garbage collection
        gc.collect()

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
               'parname':parname, 'p_ranges':p_ranges, 'p0':p0, 'p_form':p_form, 
               'p_formprior':p_formprior, 'p_sigma':p_sigma, 'ndim':ndim, 
               'nwalkers':nwalkers, 'nthreads':nthreads, 'nsteps':nsteps, 
               'nburnin':nburnin, 'uvdata':uvdata, 'impar':impar}

    # Save chain and metadata
    # Note that protocol=2 needed for python 2/3 compatibility
    pickle.dump( results, open( 'chain.p', 'wb' ), protocol=2 )

    # Return
    os.chdir(current_dir)
    return results
