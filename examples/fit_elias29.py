import os
import pickle
import numpy as np
from matplotlib.pyplot import savefig

import SimpleDiskEnvFit

from emcee import EnsembleSampler
from emcee.utils import MPIPool
import corner


def run_mcmc(main_dir, nthreads=8, nwalkers=20, nsteps=300, nburnin=100,
             plot=True, use_mpi=False):
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
      main_dir  -  directory containing input and output files
      nthreads  -  number of threads used (default 8)
      nwalkers  -  number of walkers used (default 20)
      nsteps    -  number of walker steps in the main run (default 300)
      nburnin   -  number of walker steps in initial "burn-in" run (default 100)
      plot      -  plot the posterior likelihood distributions (default True)
      use_mpi   -  use MPI pools instead of python threads. Useful for running 
                   on computer clusters using multiple nodes (default False)
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

    par = SimpleDiskEnvFit.getParams(paramfile='elias29_params.inp')

    u1, v1, Re1, Im1, w1 = np.require(np.loadtxt('Elias29uvt_270.txt', 
                                                 unpack=True), requirements='C')
    u2, v2, Re2, Im2, w2 = np.require(np.loadtxt('Elias29uvt_94.txt', 
                                                 unpack=True), requirements='C')

    visdata = [{'u':u1, 'v':v1, 'Re':Re1, 'Im':Im1, 'w':w1, 'wav':1100.},
               {'u':u2, 'v':v2, 'Re':Re2, 'Im':Im2, 'w':w2, 'wav':3000.}]

    impar = [{'npix':512,'wav':1100.,'sizeau':6000,'incl':60},
             {'npix':512,'wav':3000.,'sizeau':6000,'incl':60}]

    parname = ['mdisk','rho0Env','gsmax_disk','gsmax_env','PA',
               'dRA', 'dDec']
    p_ranges = [[-20., -2.],    # log disk mass [solar mass]
                [-23., -19.],   # log envelope density [g/cm**3]
                [-6., 0.],      # log disk grain size [cm]
                [-6., 0.],      # log envelope grain size [cm]
                [0., 180.],     # position angle [deg]
                [-2., 2.],      # offset in RA [arcsec]
                [-2., 2.]]      # offset in Dec [arcsec]
    
    ndim = len(p_ranges)

    kwargs = {'dpc': 125., 'incl': 60., 'impar': impar, 'verbose': True}

    sampler = EnsembleSampler(nwalkers, ndim, SimpleDiskEnvFit.bayes.lnpostfn,
                          args=[p_ranges, parname, par, main_dir, visdata],
                          kwargs=kwargs, threads=nthreads, pool=pool)

    # initial guess for the parameters
    p0 = [-5, -20, -4., -4., 60., 0., 0.] #  4 parameters for the model + 3 (PA, dRA, dDec)

    # initialize the walkers with an ndim-dimensional Gaussian ball
    pos = [p0 + 1.0e-2*np.random.randn(ndim) for i in range(nwalkers)]

    # Burn-in
    nburnin = np.max([1,nburnin])
    print ("INFO [{:06}]: RUN {} burn-in steps".format(0,nburnin))
    pos, prob, state = sampler.run_mcmc(pos, nburnin)
    print ("INFO [{:06}]: DONE {} burn-in steps".format(0,nsteps))
    
    # Main run
    print ("INFO [{:06}]: RUN {} main steps".format(0,nsteps))
    pos, prob, state = sampler.run_mcmc(pos, nsteps, rstate0=state, lnprob0=prob)
    print ("INFO [{:06}]: DONE {} main steps".format(0,nsteps))

    chain = sampler.chain[:, :, :]

    # Save results
    results = {'chain': chain, 'parname':parname, 'p_range':p_ranges, 'p0':p0,
               'ndim':ndim, 'nwalkers':nwalkers, 'nthreads':nthreads, 
               'visdata':visdata, 'impar':impar}
    
    # Close pool
    if use_mpi:
        pool.close()
    
    # Save chain and metadata
    # Note that protocol=2 needed for python 2/3 compatibility
    pickle.dump( results, open( 'elias29_mcmc_save.p', 'wb' ), protocol=2 )
    
    if plot:
        plot_corner(main_dir, results=results, nburnin=nburnin, save=True, 
                    figname='elias29_mcmc.pdf')
    
    # Return
    os.chdir(current_dir)
    return results

def plot_corner(main_dir, results=None, nburnin=0, save=True, figname='corner.pdf'):
    '''
    Plots the posteriori distribution of the fitted parameters.
    
    Parameters
    ----------
      main_dir  -  Main folder containing the observational constrain data, 
                   optical constants, parameter file, model run folders and 
                   <result_data>.p file containing the results of the fitting.
      results   -  Dictionary output from run_mcmc() function (optional)
      save      -  Save figure to pdf file (default True)
    '''
    current_dir = os.path.realpath('.')
    os.chdir(main_dir)
    
    if results is None:
        results = pickle.load( open( 'elias29_mcmc_save.p','rb' ) )
    
    samples = results['chain'][:, -nburnin:, :].reshape((-1, results['ndim']))
    
    fig1 = corner.corner(samples, labels=results['parname'],                                      
                         show_titles=True, quantiles=[0.16, 0.50, 0.84],
                         label_kwargs={'labelpad':20, 'fontsize':0}, 
                         fontsize=8)
    if save:
        plt.savefig(figname)
    
    os.chdir(current_dir)
    
    return

if __name__ == "__main__":
    current_dir = os.path.realpath('.')
    run_mcmc(current_dir+'/elias29', use_mpi=True)
