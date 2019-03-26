import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys

import SimpleDiskEnvFit

from emcee import EnsembleSampler
from emcee.utils import MPIPool
import corner
from galario import arcsec, deg

'''
TODO: save current position and probability to file. This should be readable by 
      the code to resume computation (useful when the cluster job runtime is limited 
      to 1 day).
      To avoid too frequent writes, store results and write after every 10 or 100 
      steps. (need to be tested)
      Individual models are not useful add option to delete already used model.
'''

def run_mcmc(main_dir, nthreads=8, nwalkers=40, nsteps=1000, nburnin=100,
             plot=True, use_mpi=False, resume=False, chain_file='chain.dat'):
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
            Number of walkers used. Default is 20.
    nsteps  : int
            Number of walker steps in the main run. Default is 300.
    nburnin : int, optional
            Number of walker steps in initial "burn-in" run. Default is 100.
    plot    : bool, optional
            Plot the posterior likelihood distributions. Default is True.
    use_mpi : bool, optional
            Use MPI pools instead of python threads. Useful for running 
            on computer clusters using multiple nodes. Default is False.
    resume  : bool, optional
            If True then resume MCMC chain from file and continue sampling 
            the posterior distribution. nwalkers should not change between 
            runs. Default is False.
    chain_file : string, optional
            Chain (parameters and probabilities) are stored in this file. 
            The file can be used to restart or continue MCMC sampling.
            Meaning of columns: walker index (1), parameter value (n), 
            log probability (1).
            If file already exists then output is automatically renamed.
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

    # Read parameter file 
    par = SimpleDiskEnvFit.getParams(paramfile='elias29_params.inp')

    # Read observational constraints
    u1, v1, Re1, Im1, w1 = np.require(np.loadtxt('Elias29uvt_270.txt', 
                                                 unpack=True), requirements='C')
    u2, v2, Re2, Im2, w2 = np.require(np.loadtxt('Elias29uvt_94.txt', 
                                                 unpack=True), requirements='C')
    
    # The Elias 29 data is noisy above 50 klambda, remove these from fitting
    use1 = np.where(np.hypot(u1,v1)/(1.1e-3) <= 5.0e4)   # Select only data below 50 klambda
    use2 = np.where(np.hypot(u2,v2)/(3.3e-3) <= 5.0e4)   # Select only data below 50 klambda

    # Bundle visibility data
    visdata = [{'u':u1[use1], 'v':v1[use1], 'Re':Re1[use1], 'Im':Im1[use1], 
                'w':w1[use1], 'wav':1100.},
               {'u':u2[use2], 'v':v2[use2], 'Re':Re2[use2], 'Im':Im2[use2], 
                'w':w2[use2], 'wav':3000.}]

    # Set image parameters
    impar = [{'npix':512,'wav':1100.,'sizeau':11000,'incl':67.},
             {'npix':512,'wav':3000.,'sizeau':11000,'incl':67.}]

    # Set parameters for bayes.lnpostfn() function
    kwargs = {'dpc': 125., 'incl': 67., 'impar': impar, 'verbose': True, 
              'PA':0.0, 'dRA':0.48*arcsec, 'dDec':0.98*arcsec}

    # If projection parameters are not known
    #parname = ['mdisk','rho0Env','gsmax_disk','gsmax_env','PA',
               #'dRA', 'dDec']
    #p_ranges = [[-10., -2.],    # log disk mass [solar mass]
                #[-23., -19.],   # log envelope density [g/cm**3]
                #[-6., 0.],      # log disk grain size [cm]
                #[-6., 0.]],     # log envelope grain size [cm]
                #[0., 180.],     # position angle [deg]
                #[-2., 2.],      # offset in RA [arcsec]
                #[-2., 2.]]      # offset in Dec [arcsec]
    ## initial guess for the parameters
    #p0 = [-5, -20, -4., -4., 0., 0.48, 0.98] # 4 parameters for the model + 3 (PA, dRA, dDec)

    # Projection parameters already known
    parname = ['mdisk','rho0Env','gsmax_disk','gsmax_env']
    p_ranges = [[-10., -2.],    # log disk mass [solar mass]
                [-23., -19.],   # log envelope density [g/cm**3]
                [-6., 0.],      # log disk grain size [cm]
                [-6., 0.]]      # log envelope grain size [cm]
    # initial guess for the parameters
    p0 = [-5, -20, -4., -4.]

    # Number of fitted parameters
    ndim = len(p_ranges)

    # initialize the walkers with an ndim-dimensional Gaussian ball
    if resume:
        raise ValueError('resume = True not yet implemented!')
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
    sampler = EnsembleSampler(nwalkers, ndim, SimpleDiskEnvFit.bayes.lnpostfn,
                          args=[p_ranges, parname, par, main_dir, visdata],
                          kwargs=kwargs, threads=nthreads, pool=pool)

    print ("INFO [{:06}]: RUN {} main steps".format(0,nsteps))

    for step in sampler.sample(pos, iterations=nsteps, lnprob0=lnprob0):
        position = step[0]
        lnprob = step[1]
        print (step[2])
        f = open(chain_file, "a")
        for k in range(nwalkers):
            posstr = ''.join(np.vectorize("%12.5E ".__mod__)(position[k]))
            f.write("{:04d} {:s}{:12.5E}\n".format(k, posstr, lnprob[k]))
        f.close()

    print ("INFO [{:06}]: DONE {} main steps".format(0,nsteps))

    # Close pool
    if use_mpi:
        pool.close()

    chain = sampler.chain[:, :, :]

    # Save results
    results = {'chain': chain, 'parname':parname, 'p_ranges':p_ranges, 'p0':p0,
               'ndim':ndim, 'nwalkers':nwalkers, 'nthreads':nthreads, 
               'nsteps':nsteps, 'nburnin':nburnin, 'visdata':visdata, 
               'impar':impar}

    # Save chain and metadata
    # Note that protocol=2 needed for python 2/3 compatibility
    pickle.dump( results, open( 'elias29_mcmc_save.p', 'wb' ), protocol=2 )

    if plot:
        plot_corner(main_dir, results=results, nburnin=nburnin, save=True, 
                    figname='elias29_mcmc.pdf', show=False)

    # Return
    os.chdir(current_dir)
    return results

def plot_corner(main_dir, results=None, nburnin=0, range=None, show=True, 
                save=True, figname='corner.pdf', full_range=False):
    '''
    Plots the posteriori distribution of the fitted parameters.

    Parameters
    ----------
    main_dir : string
            Main folder containing the observational constrain data, 
            optical constants, parameter file, model run folders and 
            <result_data>.p file containing the results of the fitting.
    results : dict, optional
            Dictionary output from run_mcmc() function.
    nburnin : int, optional
            Number of initial burn-in steps. These are not shown in figure.
    range   : list, optional
            Plotting ranges of each parameters given in a list of lists. The 
            number of bundled 2 element list (min, max) must be equal to the 
            number of fitted parameters (see also full_range). If not set, 
            then corner routine decided.
    show    : bool, optional
            Show corner plot on screen. Set this to False in non-interactive 
            script. Default is True.
    save    : bool, optional
            Save corner plot to supported file format (e.g. pdf or png). The 
            filename, including format is set by figname argument. Default is 
            True.
    figname : string, optional
            Filename of corner plot figure. Used if save argument is True.
            If no file path is specified, then figure is saved to main_dir. 
            Default is corner.pdf.
    full_range : bool, optional
            Show the complete fitting range (as contained in the results 
            dictionary). If the range is broad and the models are localised 
            this may lead to warning messages and hard to read figure. Default 
            is False.
    '''
    current_dir = os.path.realpath('.')
    os.chdir(main_dir)

    # Restore results from file
    if results is None:
        results = pickle.load( open( 'elias29_mcmc_save.p','rb' ) )

    os.chdir(current_dir)

    chain = results['chain']

    # Determine nsteps
    nstep = chain.shape[1]
    nstep = nstep - nburnin

    # Get samples and ranges
    samples = results['chain'][:, -nstep:, :].reshape((-1, results['ndim']))

    if range is None and full_range:
        range = results['p_ranges']

    fig1 = corner.corner(samples, labels=results['parname'],
                         show_titles=True, quantiles=[0.16, 0.50, 0.84],
                         label_kwargs={'labelpad':20, 'fontsize':0}, 
                         fontsize=8, range=range)
    if save:
        plt.savefig(figname)

    if show:
        plt.show()

    return

if __name__ == "__main__":
    current_dir = os.path.realpath('.')
    run_mcmc(current_dir+'/elias29', use_mpi=True)
