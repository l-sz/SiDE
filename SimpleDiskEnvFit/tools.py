#
# Tools for working with MCMC chain data.
#
# This file is part of the SimpleDiskEnvFit project.
# https://gitlab.mpcdf.mpg.de/szucs/SimpleDiskEnvFit
#
# Copyright (C) 2019 Laszlo Szucs <laszlo.szucs@mpe.mpg.de>
# Licensed under GLPv2, for more information see the LICENSE file repository.
#

from __future__ import absolute_import
from __future__ import print_function

import corner
import numpy as np
import matplotlib.pyplot as plt
import pickle

__all__ = ['emcee_chain','read_chain_ascii','read_chain_pickle']

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
                 p_ranges=None, **kwargs):
        '''
        Initialise emcee_chain() class instance.
        
        Parameters
        ----------
        chain   : array-like, float
                Array with (nwalkers, nsteps, ndim) dimension containing parameter 
                combinations explored by the MCMC walkers.
                The chain must be specified in each emcee_chain instance.
        parname : list, str
                Names of fitted parameters. ndim element list of string. The 
                list must be ordered to match the parameter order of the 3rd 
                dimension of the chain array.
        ndim    : int, optional
                Number of fitted parameters. If not set, then it is determined 
                from the shape of chain.
        nwalkers: int, optional
                Number of MCMC walkers used. If not set, then it is determined 
                from the shape of chain.
        nsteps  : int, optional
                Number of MCMC steps taken by the walkers. If not set, then it is 
                determined from the shape of chain.
        p_range : list, float, optional
                Fitted parameter range. p_range is a ndim element list, each 
                element being a two element list. These give the minimum and 
                maximum prior assumption on the fitted parameter.
                This may be used in plotting routines.
        **kwargs: dict, optional
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
        if 'lnprob' in kwargs.keys():
            self.lnprob = kwargs['lnprob']
    
    def __add__(self, other):
        '''
        Return the combined content of two emcee_chain class objects.
        
        Note: this operation is not commutative and not associative. It is assumed 
              that in c = a+b, a precedes b in "time" (walker location).
        '''
        if isinstance(other, emcee_chain):
            if self.parname != other.parname:
                raise ValueError('Parameter names do not match. Aborting!')
            
            if self.nwalkers != other.nwalkers:
                raise ValueError('Number of walkers does not match. Aborting!')
            
            ndim = self.ndim
            nwalkers = self.nwalkers
            parname = self.parname
            nsteps = self.nsteps + other.nsteps
            
            # Assume that p_ranges are the same.
            p_ranges = self.p_ranges
            
            # TODO: if accept_frac in self or other is None the set up NaN array and combined
            
            # TODO: 
            nthreads = self.nthreads
            nburnin = self.nburnin
            # Take initial parameters from self.
            if self.p0 != other.p0:
                print ('WARN: p0 parameter differs, using p0 from first term.')
            p0 = self.p0
            #
            if self.visdata != other.visdata:
                print ('WARN: visdata parameter differs, using visdata from first term.')
            visdata = self.visdata
            #
            if self.impar != other.impar:
                print ('WARN: impar parameter differs, using impar from first term.')
            impar = self.impar
            #
            # Combine lnprob
            lnprob = self.lnprob, other.lnprob
            if self.lnprob is None and other.lnprob is None:
                lnprob = None
            elif self.lnprob is None:
                lnprob_self = np.ones(self.nsteps) * np.nan
                lnprob = np.concatenate(lnprob_self, other.lnprob)
            elif other.lnprob is None:
                lnprob_other = np.ones(other.nsteps) * np.nan
                lnprob = np.concatenate(self.lnprob, lnprob_other)
            else:
                lnprob = np.concatenate((self.lnprob, other.lnprob),axis=1)
            #
            # Combine chain data
            chain = np.concatenate((self.chain, other.chain),axis=1)
        else:
            raise TypeError('Cannot add {} to emcee_chain instance.'.type(other))
        
        chain_data = {'lnprob': lnprob, 'nwalkers': nwalkers, 'nsteps': nsteps, 
                  'ndim': ndim, 'p0':p0, 'visdata':visdata, 'impar':impar}
    
        return emcee_chain(chain, parname, **chain_data)
    
    def save(self, filename='chain_data.p'):
        '''
        Saves emcee_chain data to pickle file.
        
        The output file is Python 2/3 compatible and is written to the local 
        directory.
        
        Parameters
        ----------
        filename    : str, optional
                    Name of output file. Default is chain_data.p.
        '''
        results = {'chain': self.chain, 'accept_frac':self.accept_frac, 
                   'lnprob':self.lnprob, 'parname':self.parname, 
                   'p_ranges':self.p_ranges, 'p0':self.p0, 'ndim':self.ndim, 
                   'nwalkers':self.nwalkers, 'nthreads':self.nthreads, 
                   'nsteps':self.nsteps, 'nburnin':self.nburnin, 
                   'visdata':self.visdata, 'impar':self.impar}
        
        # Use protocol 2 for python 2/3 compatibility
        pickle.dump( results, open( filename, 'wb' ), protocol=2 )
        
        return
        
    def save_ascii(self, filename='chain_data.dat'):
        '''
        Saves emcee_chain data to ascii file.
        
        The output file is written to the local directory.
        
        Parameters
        ----------
        filename    : str, optional
                    Name of output file. Default is chain_data.txt.
        '''
        print ('Function not implemented yet.')
        return
        
    def plot_chain(self, show=True, save=True, gamma=1.0, alpha_floor=0.1,
                   xlim=None, ylim=None, xscale='linear', yscale='linear', 
                   figname='walkers.pdf', **kwargs):
        '''
        Plots the path taken by walkers for each fitted parameters.
        
        Parameters
        ----------
        show    : bool, optional
                Show walker path figure on screen. Set this to False in 
                non-interactive script. Default is True.
        save    : bool, optional
                Save walker path figure plot to supported file format (e.g. 
                pdf or png). The file name, including format is set by 
                figname argument. Default is True.
        gamma   : float, optional
                Sets the scaling between line transparency and lnprob (posterior
                probability of the model. gamma > 1 one emphesises the chains 
                with the highest likelihood, gamma = 0 show all chains with 
                solid, non-transparent line. Default value is 1.0.
        alpha_floor : float, optional
                Lower limit on transparency value. This is used to make sure 
                all models are visible on the figure. Default value is 0.1.
        xlim    : array-like, optional 
                Set x-axis plotting limits. Two element array like object is 
                expected. Default is None.
        ylim    : array-like, optional
                Set y-axis plotting limits. Two element array like object is 
                expected. Default is None.
        xscale  : string, optional
                Set the scaling of x axis. Recognised values: "linear", "log", 
                "symlog", "logit". Default is "linear".
        yscale  : string, optional
                Set the scaling of y axis. Recognised values: "linear", "log", 
                "symlog", "logit". Default is "linear".
        figname : string, optional
                File name of walker path figure. Used if save argument is True.
                If no file path is specified, then figure is saved to the current 
                directory. Default is walkers.pdf.
        **kwargs : Line2D properties, optional
                Keyword argument will be passed to matplotlib.pyplot.plot().
        '''
        fig, ax = plt.subplots(self.ndim, 1, sharex=True, figsize=(6,3*self.ndim))
        
        if self.lnprob is None:
            lnprob_max = 1.0
            lnprob = np.ones((self.nwalkers,1))
        else:
            lnprob_max = self.lnprob.max()
            lnprob = self.lnprob
        # Scale lnprob if it is too small (to avoid division by zero)
        if lnprob_max < -300.:
            scale = abs(lnprob_max)
        else:
            scale = 1.0
        
        # loop over parameters
        for p in range(self.ndim):
            ax[p].set_ylabel(self.parname[p])
            
            # path of individual walkers
            for w in range(self.nwalkers):
                chain_max = lnprob[w,:].max()
                alpha = max([(10**(chain_max/scale) / 10**(lnprob_max/scale))**gamma, 
                            alpha_floor])
                ax[p].plot(self.chain[w,:,p],'b-', alpha=alpha, **kwargs)
                # Plotting limits
                if xlim:
                    ax[p].set_xlim(xlim)
                if ylim:
                    ax[p].set_ylim(ylim)
                # Axis scale
                if xscale:
                    ax[p].set_xscale(xscale)
                if yscale:
                    ax[p].set_yscale(yscale)
            
            # initial guess
            #ax[p].plot([0,nstep],[par[p],par[p]],'g-', linewidth=2)
        ax[-1].set_xlabel('step number')
        
        if save:
            plt.savefig(figname)

        if show:
            plt.show()
        
        return

    def plot_lnprob(self, show=True, save=True, gamma=1.0, alpha_floor=0.1,
                    xlim=None, ylim=None, xscale='linear', yscale='linear',
                    figname='posterior.pdf', **kwargs):
        '''
        Plots the posterior probability of models explored by walkers.
        
        Parameters
        ----------
        show    : bool, optional
                Show corner plot on screen. Set this to False in non-interactive 
                script. Default is True.
        save    : bool, optional
                Save corner plot to supported file format (e.g. pdf or png). The 
                file name, including format is set by figname argument. Default is 
                True.
        gamma   : float, optional
                Sets the scaling between line transparency and lnprob (posterior
                probability of the model. gamma > 1 one emphesises the chains 
                with the highest likelihood, gamma = 0 show all chains with 
                solid, non-transparent line. Default value is 1.0.
        alpha_floor : float, optional
                Lower limit on transparency value. This is used to make sure 
                all models are visible on the figure. Default value is 0.1.
        xlim    : array-like, optional 
                Set x-axis plotting limits. Two element array like object is 
                expected. Default is None.
        ylim    : array-like, optional
                Set y-axis plotting limits. Two element array like object is 
                expected. Default is None.
        xscale  : string, optional
                Set the scaling of x axis. Recognised values: "linear", "log", 
                "symlog", "logit". Default is "linear".
        yscale  : string, optional
                Set the scaling of y axis. Recognised values: "linear", "log", 
                "symlog", "logit". Default is "linear".
        figname : string, optional
                File name of corner plot figure. Used if save argument is True.
                If no file path is specified, then figure is saved to the current 
                directory. Default is posterior.pdf.
        **kwargs : Line2D properties, optional
                Keyword argument will be passed to matplotlib.pyplot.plot().
        '''
        if self.lnprob is None:
            raise ValueError('lnprob class variable is not set.')
        
        fig, ax = plt.subplots(1, 1, sharex=True)
        
        lnprob_max = self.lnprob.max()
        # Scale lnprob if it is too small (to avoid division by zero)
        if lnprob_max < -99.:
            scale = abs(lnprob_max)
        else:
            scale = 1.0
        
        for w in range(self.nwalkers):
            chain_max = self.lnprob[w,:].max()
            alpha = max([(10**(chain_max/scale) / 10**(lnprob_max/scale))**gamma, 
                         alpha_floor])
            ax.plot(self.lnprob[w,:],'-', alpha=alpha, **kwargs)
        ax.set_xlabel('step number')
        ax.set_ylabel('$ln$(P)')
        
        # Plotting limits
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        # Axis scale
        if xscale:
            ax.set_xscale(xscale)
        if yscale:
            ax.set_yscale(yscale)
                    
        if save:
            plt.savefig(figname)

        if show:
            plt.show()
        
        return

    def plot_corner(self, nburnin=0, range=None, full_range=False, show=True, 
                    save=True, figname='corner.pdf', **kwargs):
        '''
        Plots the posteriori distribution of the fitted parameters.

        Method uses the corner package. Parameters can be passed to the corner 
        package using kwargs.

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
                file name, including format is set by figname argument. Default is 
                True.
        figname : string, optional
                File name of corner plot figure. Used if save argument is True.
                If no file path is specified, then figure is saved to the current 
                directory. Default is corner.pdf.
        **kwargs : keyword arguments, optional
                Keyword argument will be passed to corner.corner().
        '''
        # Determine nsteps
        if nburnin < 0:
            print ("WARN: nburnin < 0: all steps are plotted!")
        nstep = self.nsteps
        nstep = nstep - nburnin

        # Get samples and ranges
        samples = self.chain[:, -(nstep):, :].reshape((-1, self.ndim))

        if range is None and full_range:
            range = self.p_ranges

        fig1 = corner.corner(samples, labels=self.parname,
                            show_titles=True, quantiles=[0.16, 0.50, 0.84],
                            label_kwargs={'labelpad':20, 'fontsize':0}, 
                            fontsize=8, range=range, **kwargs)
        if save:
            plt.savefig(figname)

        if show:
            plt.show()

        return

def read_chain_ascii(filename='chain.dat'):
    '''
    Reads ascii format chain data to emcee_chain class instance.
    
    Use this method to read (in)complete chain files, written during execution.
    Particularly useful for continuing incomplete MCMC chains.
    
    Parameters
    ----------
    filename    : str
                Name of input file containing the chain data. Default is 
                chain.txt.

    Returns
    -------
    emcee_chain class instance, created from data in input file.
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
    
    return emcee_chain(chain, parname, **chain_data)

def read_chain_pickle(filename='chain.p'):
    '''
    Reads binary (pickle) format chain data to emcee_chain class instance.
    
    Use this method to analyse complete chain files.
    
    Parameters
    ----------
    filename    : str
                Name of input file containing the chain data. Default is 
                chain.txt.

    Returns
    -------
    emcee_chain class instance, created from data in input file.
    
    Note: method tries compatibility mode if pickle read fails. (may occur on 
    Python 3.7)
    '''
    # Test whether file exists
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
    
    return emcee_chain(data['chain'], parname, **chain_data)

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
