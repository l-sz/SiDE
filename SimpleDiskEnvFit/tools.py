import corner
import numpy as np
import matplotlib.pyplot as plt
import pickle

plt.style.use('classic')

class emcee_chain():
    '''
    Class containing MCMC chain data and associated methods.
    '''
    
    nwalkers = None
    ndim = None
    nsteps = None
    parname = None
    p_ranges = None
    chain = None
    lnprob = None
    accept_frac = None
    p0 = None
    nthreads = None
    nburnin = None
    # TODO: may be deleted
    visdata = None
    impar = None
    
    def __init__(self, chain, parname, ndim=None, nwalkers=None, nsteps=None, 
                 p_ranges=None, lnprob=None, **kwargs):
        '''
        Initialise emcee_chain() instance.
        '''
        self.chain = chain
        self.parname = parname
        
        if ndim is None:
            self.ndim = chain.shape[2]
        else:
            self.ndim = ndim
        if nwalkers is None:
            self.nwalkers = chain.shape[0]
        else:
            self.nwalkers = nwalkers
        if nsteps is None:
            self.nsteps = chain.shape[1]
        else:
            self.nsteps = nsteps
        # if prior range is not given then guess it from chain
        if p_ranges is None:
            p_ranges = []
            for i in range(self.ndim):
                tmp = chain[:,:,i]
                p_ranges.append([tmp.min(),tmp.max()])
        else:
            self.p_ranges = p_ranges
        # Extract additional meta-data
        if 'accept_frac' in kwargs.keys():
            self.accept_frac = kwargs['accept_frac']
        if 'p0' in kwargs.keys():
            self.p0 = kwargs['p0']
        if 'nthreads' in kwargs.keys():
            self.nthreads = kwargs['nthreads']
        if 'nburnin' in kwargs.keys():
            self.nburnin = kwargs['nburnin']
        if 'visdata' in kwargs.keys():
            self.visdata = kwargs['visdata']
        if 'impar' in kwargs.keys():
            self.impar = kwargs['impar']
    
    def save(self, filename='chain_data.p'):
        
        results = {'chain': self.chain, 'accept_frac':self.accept_frac, 
                   'lnprob':self.lnprob, 'parname':self.parname, 
                   'p_ranges':self.p_ranges, 'p0':self.p0, 'ndim':self.ndim, 
                   'nwalkers':self.nwalkers, 'nthreads':self.nthreads, 
                   'nsteps':self.nsteps, 'nburnin':self.nburnin, 
                   'visdata':self.visdata, 'impar':self.impar}
        
        # Use protocol 2 for python 2/3 compatibility
        pickle.dump( results, open( filename, 'wb' ), protocol=2 )
        
        return
        
    def save_txt(self, filename='chain_data.txt'):
        '''
        '''
        print ('Function not implemented yet.')
        return
        
    def plot_chain(self, show=True, save=True, figname='walkers.pdf'):
        '''
        Plots the path taken by walkers for each fitted parameters.
        
        Parameters
        ----------
        '''
        fig, ax = plt.subplots(self.ndim, 1, sharex=True)
        
        # loop over parameters
        for p in range(self.ndim):
            ax[p].set_ylabel(self.parname[p])
            
            # path of individual walkers
            for w in range(self.nwalkers):
                ax[p].plot(self.chain[w,:,p],'b-', linewidth=1)
            
            # initial guess
            #ax[p].plot([0,nstep],[par[p],par[p]],'g-', linewidth=2)
        ax[-1].set_xlabel('step number')
            
        return

    def plot_corner(self, nburnin=0, range=None, full_range=False, show=True, 
                    save=True, figname='corner.pdf'):
        '''
        Plots the posteriori distribution of the fitted parameters.

        Parameters
        ----------
        nburnin : int
                Number of initial burn-in steps. These are not shown in figure.
                Default is 0.
        range   : list, optional
                Plotting ranges of each parameters given in a list of lists. The 
                number of bundled 2 element list (min, max) must be equal to the 
                number of fitted parameters (see also full_range). If not set, 
                then corner routine decided.
        full_range : bool, optional
                Show the complete fitting range (as contained in the results 
                dictionary). If the range is broad and the models are localised 
                this may lead to warning messages and hard to read figure. Default 
                is False.
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
        '''
        # Determine nsteps
        nstep = self.nsteps
        nstep = nstep - nburnin

        # Get samples and ranges
        samples = self.chain[:, -nstep:, :].reshape((-1, self.ndim))

        if range is None and full_range:
            range = self.p_ranges

        fig1 = corner.corner(samples, labels=self.parname,
                            show_titles=True, quantiles=[0.16, 0.50, 0.84],
                            label_kwargs={'labelpad':20, 'fontsize':0}, 
                            fontsize=8, range=range)
        if save:
            plt.savefig(figname)

        if show:
            plt.show()

        return

def read_chain_txt(filename='chain.txt'):
    '''
    Reads data from chain file and prepares pos0 and lnprob0 for restarting 
    the MCMC run. It also returns an array with previous results to be merged 
    with new probability estimations for plotting.

    Some metadata is returned (nthreads, nwalker, nstep, MPI).
    '''
    
    # Read file header
    f = open(filename, 'r')
    header = f.readlines()[0:2]
    f.close()
    #
    # Get parameter names
    for i in range(2):
        header[i] = header[i].replace('\n','').replace('# ','')
    parname = header[1].split()[1:-1]
    # 
    # Get control parameters
    chain_data0 = parse_karg(header[0].replace('\n',
                   '').replace('# ','').split(','))

    # Read data
    data = np.loadtxt(filename)
    
    iwalker = np.int32( data[:,0] )
    pos0 = data[:,1:-1]
    lnprob0 = data[:,-1]
    
    # Determine MCMC parameters
    nwalkers = iwalker.max() + 1
    ndim = pos0.shape[1]
    nsteps = np.int32(pos0.shape[0] / nwalkers)

    chain = np.empty((nwalkers, nsteps, ndim))
    lnprob = np.empty((nwalkers,nsteps))
    
    for i in range(nwalkers):
        loc = np.where( i == iwalker )
        chain[i,:,:] = pos0[loc,:]
        lnprob[i,:] = lnprob0[loc]
    
    # Compare set and final control parameters
    
    if chain_data0['nwalkers'] != nwalkers:
        print (chain_data0['nwalkers'], nwalkers)
        raise ValueError('nwalkers in header does not match chain size.')

    if chain_data0['nsteps'] != nsteps:
        print ('WARN: nsteps in header ({}) does not match chain size ({}), interrupted MCMC.'.format(nsteps, 
                                                    chain_data0['nsteps']))
        
    chain_data = {'lnprob': lnprob, 'nwalkers': nwalkers, 'nsteps': nsteps, 
                  'ndim': ndim,'use_mpi': chain_data0['MPI']}
    
    return emcee_chain(chain, parname, kwargs=chain_data)

def read_chain_pickle(filename='chain.p'):
    '''
    Reads data from chain file and prepares pos0 and lnprob0 for restarting 
    the MCMC run. It also returns an array with previous results to be merged 
    with new probability estimations for plotting.

    Some metadata is returned (nthreads, nwalker, nstep, MPI).
    '''
    
    # Test whether file exists
    
    # Try to read pickle file (try compatibility mode if read fails)
    try:
        data = pickle.load(open(filename,'rb'))
    except:
        print ("WARN: pickle file cannot be read, trying compatibility mode.")
        data_ = pickle.load(open(filename,'rb'), fix_imports=True, 
                           encoding='bytes')
        # workaround for string encoding
        data = {}
        for k in data_.keys():
            data[k.decode('utf8')] = data_[k]
        del data_
    
    # Convert byte to string if needed
    parname = []
    for pn in data['parname']:
        if type(pn) == bytes:
            parname.append(pn.decode('utf8'))
        else:
            parname.append(pn)
    
    if 'lnprob' in data.keys():
        lnprob = data['lnprob']
    else:
        lnprob = None
    if 'nwakers' in data.keys():
        nwakers = data['nwalkers']
    else:
        nwalkers = None
    if 'nsteps' in data.keys():
        nsteps = data['nsteps']
    else:
        nsteps = None
    if 'use_mpi' in data.keys():
        use_mpi = data['use_mpi']
    else:
        use_mpi = None
    # Check for additional meta-data
    if 'accept_frac' in data.keys():
        accept_frac = data['accept_frac']
    else:
        accept_frac = None
    if 'p0' in data.keys():
        p0 = data['p0']
    else:
        p0 = None
    if 'nthreads' in data.keys():
        nthreads = data['nthreads']
    else:
        nthreads = None
    if 'nburnin' in data.keys():
        nburnin = data['nburnin']
    else:
        nburnin = None
    if 'visdata' in data.keys():
        visdata = data['visdata']
    else:
        visdata = None
    if 'impar' in data.keys():
        impar = data['impar']
    else:
        impar = None
    
    chain_data = {'lnprob':lnprob, 'nwalkers':nwalkers, 'nsteps': nsteps, 
                  'ndim':data['ndim'], 'use_mpi':use_mpi, 
                  'accept_frac':accept_frac, 'p0':p0, 'nthreads':nthreads, 
                  'nburnin':nburnin, 'visdata':visdata, 'impar':impar}
    
    return emcee_chain(data['chain'], parname, kwargs=chain_data)

def parse_karg(string_list):
    '''
    Converts list of strings given in 'a = b' form to dictionary, where the key 
    is 'a' and the value is eval(b).
    '''
    dic = {}
    for s in string_list:
        data = s.split('=')
        dic[data[0].strip()] = eval(data[1].strip())
    return dic
