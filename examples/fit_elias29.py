import os
import sys
import numpy as np
from galario import arcsec, deg

from side import tools
from side import run_mcmc

if __name__ == "__main__":
    
    # Directory of main program
    current_dir = os.path.realpath('.')
    
    # Set working directory (this contains the parameter file, optical constants 
    # file, the restart / result files and the computer models. Assumed to be 
    # relative to current_dir.
    work_dir = 'elias29'
    
    # Fiducial model parameter file name
    paramfile = 'elias29_params.inp'
    
    #
    # Interpret command line arguments (if any)
    #
    restart = False
    if len(sys.argv) > 1:
        if sys.argv[1] == 'restart':
            restart = True
            #
            # Choose the most recent chain file in work_dir
            #
            list_of_chains = glob.glob(work_dir + '/chain*.dat')
            if len(list_of_chains) > 0:
                restart_file = max(list_of_chains, key=os.path.getctime)
            else:
                restart_file = work_dir + '/chain.dat'
 
    # If second argument is set, then use it as the chain file name
    if len(sys.argv) > 2:
        restart_file = sys.argv[2]

    
    #
    # Initialize observed visibility data
    #
    
    # Read in visibility data
    u1, v1, Re1, Im1, w1 = np.require(np.loadtxt('{}/Elias29uvt_270.txt'.format(work_dir), 
                                    unpack=True), requirements='C')
    u2, v2, Re2, Im2, w2 = np.require(np.loadtxt('{}/Elias29uvt_94.txt'.format(work_dir), 
                                    unpack=True), requirements='C')
        
    # The Elias 29 data is noisy above 50 klambda, remove these from fitting
    use1 = np.where(np.hypot(u1,v1)/(1.1e-3) <= 5.0e4)   # Select only data below 50 klambda
    use2 = np.where(np.hypot(u2,v2)/(3.3e-3) <= 5.0e4)   # Select only data below 50 klambda

    # Bundle visibility data
    #
    # Note: PA, dRA and dDec is overwritten by values (if any) provided in kwargs.
    #       If PA, dRA and dDec are fitted, then values specified here are also 
    #       overwritten.
    visdata = [{'u':u1[use1], 'v':v1[use1], 'Re':Re1[use1], 'Im':Im1[use1], 
                'w':w1[use1], 'wav':1100., 'PA':0.0, 'dRA':0.48*arcsec,
                'dDec':0.98*arcsec},
            {'u':u2[use2], 'v':v2[use2], 'Re':Re2[use2], 'Im':Im2[use2], 
                'w':w2[use2], 'wav':3000., 'PA':0.0, 'dRA':0.0, 'dDec': 0.0}]

    #
    # Set image and runtime parameters
    #

    # Set image parameters
    #
    # Important: set position angle in the visdata dictionary
    impar = [{'npix':512,'wav':[1100.,3000.],'sizeau':11000}]

    # Set parameters for bayes.lnpostfn() function
    #
    # Important: if the inclination is not a fitted parameter then it should be 
    # defined in kwargs.
    kwargs = {'dpc': 125., 'incl': 67., 'impar': impar, 'verbose': True, 
            'idisk':True, 'ienv':True, 'icav':False,
            'cleanModel': True, 'binary': True, 'chi2_only':True, 
            'galario_check':False, 'time':True }


    #
    # Fitted parameters and priori ranges
    #

    # Choose fitting parameters and initial values
    parname = ['mdisk','rho0Env','gsmax_disk','gsmax_env']
    p_ranges = [[-10., -2.],   # log disk dust mass [solar mass]
               [-23., -19.],   # log envelope dust density [g/cm**3]
               [-6., 0.],      # log disk grain size [cm]
               [-6., 0.]]      # log envelope grain size [cm]

    # initial guess for the parameters
    p0 = [-5, -20, -4., -4.]

    p_form = ['log', 'log', 'log', 'log']
    p_formprior = ['uniform', 'uniform', 'uniform', 'uniform']
    p_sigma = [0., 0., 0., 0., 0., 0.]

    #
    # Run the MCMC sampler
    #
    # Resume from restart file
    if restart:
        results = run_mcmc(current_dir+'/{}'.format(work_dir), visdata, 
                        paramfile=paramfile, use_mpi=True, verbose=True, 
                        impar=impar, parname=parname,
                        nwalkers=100, nsteps=300, nburnin=0,
                        p_ranges=p_ranges, p0=p0, p_form=p_form, 
                        p_formprior=p_formprior, p_sigma=p_sigma, 
                        kwargs=kwargs, resume=True, restart_file=restart_file)

    # Start new run
    else:
        results = run_mcmc(current_dir+'/{}'.format(work_dir), visdata, 
                        paramfile=paramfile, use_mpi=True, 
                        verbose=True, impar=impar, parname=parname, 
                        nwalkers=100, nsteps=300, nburnin=0,
                        p_ranges=p_ranges, p0=p0, p_form=p_form, 
                        p_formprior=p_formprior, p_sigma=p_sigma, kwargs=kwargs)
    
    print ("Note that mass and density parameters always refer to the dust component!")
    
    print ("All Done!")
