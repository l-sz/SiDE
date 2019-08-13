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
import time
import copy
import numpy as np
from shutil import copyfile, rmtree
from timeit import default_timer as timer

import radmc3dPy
import radmc3dPy.natconst as nc

import galario
import uvplot

# Internal routines
from . import models
from . import runner

__all__ = ['radmc3dModel','getParams']

global module_dir

module_dir = os.path.dirname(sys.modules['SimpleDiskEnvFit'].__file__)

class radmc3dModel:
    '''
    Interface for creating, running and analysing RADMC-3D models.
    
    The class provides storage, read-in and write-out routines, runner interface 
    to call RADMC-3D as a child process and interface to galario code to translate 
    images to the complex visibility space. 
    
    Plotting routines should be added in the future.
    '''
    ID = None
    verbose = None
    
    # radmc3dPy objects
    modpar = None
    grid = None
    data = None
    radsource = None
    opac = None
    opac_files = None
    use_binary = None
    
    # Model specific data
    model_dir = None
    resource_dir = None
    
    # Computed model parameters
    mdisk = 0.0
    m_slab = 0.0
    sig0_slab = 0.0
    menv = 0.0
    menv3000 = 0.0
    sig0 = 0.0
    Lstar = 0.0
    Iisrf = None
    
    # Result data
    image = None
    SED = None
    vis_mod = None
    vis_inp = None
    chi2 = None
    nvis = None           # number of visibility points, used for normalisation
   
    # Fitting parameters
    prediction = None
    
    # runner
    rrun = None
    
    def __init__(self, modpar=None, model_dir=None, resource_dir=None, 
                 idisk=True, islab=False, ienv=True, icav=False, iext=False,
                 write2folder=False, ID=None, verbose=False, binary=False):
        '''
        Self contained RADMC-3D model class. 
        
        The class provides and easy interface to create, write and run RADMC-3D 
        protoplanetary disk and envelope dust continuum models. Model parameters 
        are defined in a radmc3dPy.modPar class object (see getParams() method).
        Model components (disk, envelope, cavity) might be activated and 
        deactivated at model initialization or with the updateModel() method.
        
        If model_dir parameter is set then it overwrites the value specified in 
        modPar parameter list. If model_dir is not set at initialization, then 
        the value from modPar is used. If model_dir is set neither as parameter 
        nor it is contained in modpPar, then it defaults to '.' current folder 
        (using absolute path).
        
        Additional files needed to create the model (e.g. dust opacity file or 
        complex index of refraction (.lnk) file) should be found in resource_dir.
        If resource_dir is not set, then search the {SIMPLEDISKENVFIT_HOME}/lnk_files
        folder for the requested files.
        
        Parameters
        ----------
        modpar : radmc3dPy.radmc3dPar class object
                Containing the model parameter values.
        model_dir : string
                Path (absolute or relative) to the folder where the model is stored 
                and/or written. Files needed to be written to disk for the 
                RADMC-3D computation are stored here. model_dir is created when 
                write2folder() class method is called.
        resource_dir : sting
                Path (absolute or relative) to the folder containing additional 
                files (e.g. dust opacity or lnk files) that are needed to create 
                the model. Defaults to {SIMPLEDISKENVFIT_HOME}/lnk_files.
        idisk : bool
                If True, then include disk in model. Default is True.
        islab : bool,
                If True, then inlculde slab density distribution in model. 
                Note that islab and idisk should not be set True at the same 
                time! Default is False.
        ienv :  bool
                If True, then include envelope in model. Default is True.
        icav :  bool
                If True, then envlope has a cavity. Default is False.
        iext :  bool, optional
                Irradiate model by external radiation field (1 Draine field 
                strength). See Draine (1978) and Black (1994). Default is False.
        write2folder : bool, optional
                If True then use write2folder() class method to save model 
                at initialization to the destination folder. Default is False.
        ID :    integer, optional
                Set model ID directly by user. If None, then a randomly generated 
                integer in teh (0,99999) range is used. Default is None.
        verbose : bool, optional
                If True, then print summary of model parameters to standard 
                output. Runtime INFO messages are also printed to standard 
                output. Default is False.
        binary  : bool, optional
                If True, then RADMC3D will use binary I/O, if False then use 
                ASCII I/O. Binary I/O may improve computation speed and reduce 
                disk space usage when models are kept (i.e. cleanModel is not 
                called).
        '''
        # Set model ID
        if ID is None:
            self.ID = np.random.randint(0,99999)
        else:
            self.ID = ID
            
        # Set verbosity
        self.verbose = verbose
        
        # Use binary RADMC3D I/O?
        self.binary = binary
        
        # Create empty model
        self.grid = radmc3dPy.analyze.radmc3dGrid()
        self.data = radmc3dPy.analyze.radmc3dData(self.grid)
 
        # Set parameters
        if modpar is not None:
            self.modpar = modpar
        else:
            self.modpar = getParams()
        
        # Set model directory
        if model_dir is not None:
            self.model_dir = model_dir
        elif "model_dir" in self.modpar.ppar.keys():
            self.model_dir = self.modpar.ppar["model_dir"]
        else:
            self.model_dir = os.path.realpath('.')
        
        # Set resource directory
        if resource_dir is None:
            self.resource_dir = '{}/{}'.format(module_dir,'/lnk_files')
        else:
            self.resource_dir = resource_dir
          
        # Update model_dir parameter in modpar
        self.modpar.setPar(["model_dir", "'{}'".format(str(self.model_dir)), 
                            " model folder path",
                            "Model carolina"])
        
        # Update control parameters
        self.modpar.setPar(['idisk',str(idisk),' Include disk in model?',
                                'Disk parameters'])
        
        self.modpar.setPar(['islab',str(islab),' Include slab in model?',
                                'Slab parameters'])
        
        self.modpar.setPar(['ienv',str(ienv),' Include envelope in model?',
                            'Envelope parameters'])
        
        self.modpar.setPar(['icav',str(icav), 
                            ' Include envelope cavity in model?',
                            'Envelope parameters'])
            
        self.modpar.setPar(['iext',str(iext),'Include external radiation?',
                            'Radiation sources'])
        # Update output format parameter
        if self.binary is True:
            self.modpar.setPar(['rto_style', '3'])
        else:
            self.modpar.setPar(['rto_style', '1'])

        # Set model grid
        self._setGrid()

        # Set radiation source
        self._setRadSources()
        
        # Set opacity data
        self.opac = radmc3dPy.analyze.radmc3dDustOpac()
        if 'dustkappa_ext' in self.modpar.ppar.keys():
            self.opac_files = self.modpar.ppar['dustkappa_ext']
        self.computeOpac()

        # Set model density distribution
        self._setDustDensity()
        
        # Write data to model_dir
        if write2folder:
            self.write2folder()
        
        # Write out model parameters
        if verbose:
            self.infoModelParams()
        
        # Done

    def readModel(self, model_dir=None):
        '''
        Read model from folder (parameter file, grid, density)
        
        Parameters
        ----------
        model_dir : string
                Path to model to be read.
        '''
        print ('WARN [{:06}]: This function is not implemented yet!'.format(ID))
        
        return 0

    def updateModel(self, modpar=None, model_dir=None, verbose=False):
        '''
        Updates model density distribution and location according to
        set parameters.
        
        See also switchComponents() method.
        
        Parameters
        ----------
        modpar : radmc3dPy.radmc3dPar class object
                Containing the new or updated model parameter values.
        model_dir : string
                New model folder name.
        verbose : bool, optional
                If True then write parameter summary to screen. Default is False.
        '''
        self.__init__(modpar=modpar, model_dir=model_dir, verbose=verbose)
        
        return 0
        
    def _setRadSources(self):
        '''
        Compute and store stellar and optional interstellar radiation sources.
        
        Black body spectrum is assumed for the stellar radiation. The optional 
        interstellar radiation (ISRF) has a strength of the Draine field and 
        the spectrum given by Draien (1978) and Black (1994)
        
        ISRF computed only if modpar class variable contains the 'iext' keyword 
        and the value is set to True. (see __init__)
        '''
        
        # Stellar radiation
        self.Lstar = (self.modpar.ppar['rstar'][0]/nc.rs)**2 * \
                     (self.modpar.ppar['tstar'][0]/5772.)**4
    
        self.radsource = radmc3dPy.analyze.radmc3dRadSources(ppar=self.modpar.ppar, 
                                                             grid=self.grid)
        self.radsource.getStarSpectrum(ppar=self.modpar.ppar)
        
        # External irradiation
        if self.modpar.ppar['iext']:
            self.Iisrf = models.ISradField(self.grid, self.modpar.ppar)
        
        return 0

    def _setGrid(self):
        '''
        Creates wavelength and spatial grids using internal radmc3dPy functions.
        '''
        
        # Create the wavelength grid
        self.grid.makeWavelengthGrid(ppar=self.modpar.ppar)
        
        # Create the spatial grid
        self.grid.makeSpatialGrid(ppar=self.modpar.ppar)
        
        return 0

    def _setDustDensity(self):
        '''
        Computes dust density distributions.
        
        Disk and envelope density may be treated as separate (ngpop==2) or 
        combined (ngpop!=2) species, depending on the ngpop parameter in modpar 
        class variable.
        
        The disk component is a parametric hydrostatic disk model (see models.py).
        
        Three possible envelope density structures may be used: Ulrich1976, 
        Tafalla2004 and powerlaw (see model.py). Optionally, the envelope may 
        include a cavity.
        
        Model parameters are stored in the modpar class variable.
        '''
        ppar = self.modpar.ppar   # in order to shorten code
   
        # Compute density distributions
        rho_disk_dust = np.zeros([self.grid.nx, self.grid.ny, self.grid.nz, 1], 
                                 dtype=np.float64)
        rho_env_dust = np.zeros([self.grid.nx, self.grid.ny, self.grid.nz, 1], 
                                dtype=np.float64)
        rho_slab_dust = np.zeros([self.grid.nx, self.grid.ny, self.grid.nz, 1], 
                                dtype=np.float64)
        
        if ppar['idisk'] == True:
            
            rho_disk_dust, self.mdisk, self.sig0 = \
                models.flaring_disk(self.grid, ppar=ppar)

        if ppar['islab'] == True:
                
            rho_slab_dust, self.m_slab, self.sig0_slab = \
                models.slab_wrapper(self.grid, ppar=ppar)

        if ppar['ienv'] == True:
            
            if (ppar['modeEnv'] == 'Ulrich1976'):
                rho_env_dust, self.menv, self.menv3000 = \
                    models.ulrich_envelope(self.grid, ppar=ppar, 
                                           cavity=ppar['icav'])
            elif (ppar['modeEnv'] == 'Tafalla2004'):
                rho_env_dust, self.menv, self.menv3000 = \
                    models.tafalla_envelope(self.grid, ppar=ppar, 
                                            cavity=ppar['icav'])
            elif (ppar['modeEnv'] == 'powerlaw'):
                rho_env_dust, self.menv, self.menv3000 = \
                    models.powerlaw_envelope(self.grid, ppar=ppar, 
                                             cavity=ppar['icav'])
            else:
                raise ValueError('Unknown envelope mode: modeEnv = {:s}'.format(    
                    ppar['modeEnv'] ))
            
        # Add up density contributions
        if 'ngpop' in ppar.keys():
            ngpop = ppar['ngpop']
        else:
            ngpop = 1
            print ("WARN [{:06}]: ngpop not defined in parameter file, using {:2} dust species".format(self.ID, ngpop))
        
        if ngpop == 2 and ppar['idisk'] and ppar['ienv']:
            rhodust = np.zeros([self.grid.nx, self.grid.ny, self.grid.nz, ngpop])
            rhodust[:,:,:,0] = rho_disk_dust[:,:,:,0]
            rhodust[:,:,:,1] = rho_env_dust[:,:,:,0]
            if ppar['islab']:
                if self.verbose:
                    print('INFO [{:06}]: Adding slab to envelope.'.format(self.ID))
                rhodust[:,:,:,1] += rho_slab_dust[:,:,:,0]
            self.data.rhodust = rhodust
        elif ngpop == 3 and ppar['idisk'] and ppar['islab'] and ppar['ienv']:
            rhodust = np.zeros([self.grid.nx, self.grid.ny, self.grid.nz, ngpop])
            rhodust[:,:,:,0] = rho_disk_dust[:,:,:,0]
            rhodust[:,:,:,1] = rho_env_dust[:,:,:,0]
            rhodust[:,:,:,2] = rho_slab_dust[:,:,:,0]
            self.data.rhodust = rhodust
        else:
            self.data.rhodust = rho_env_dust + rho_disk_dust
        
        return 0
        
    def _writeSourceExternal(self, fname='external_source.inp'):
        '''
        Writes external radiation file to disk
        
        Parameters
        ----------
        fname : string (default external_source.inp)
                Name of output file
        '''
        if self.verbose:
            print('INFO [{:06}]: Writing {}'.format(self.ID, fname))
            
        if self.grid is None:
            raise ValueError('_writeSourceExternal called before grid is set.')
        
        nwav = self.grid.nwav
        
        wfile = open(fname, 'w')
        wfile.write('%d\n'%2)     # this is the format, should be 2 always
        wfile.write('%d\n'%nwav)  # number of wavelength
        for ilam in range(nwav):  # first wavelength then Inu ISRF
            wfile.write('%.9e\n'%self.grid.wav[ilam])
        for ilam in range(nwav):
            wfile.write('%.9e\n'%self.Iisrf[ilam])
        wfile.close()
        
        return 0

    def switchComponents(self, idisk=None, ienv=None, icav=None, iext=None,
                         write=True):
        '''
        For existing models this function switched ON/OFF model components 
        (disk or envelope) and writes the new density distribution to disk.
        The equivalent self.modpar parameters are also changed. 
        
        Parameters
        ----------
        idisk : bool
                Include disk in model.
        ienv :  bool   
                Include envelope in model.
        icav :  bool, optional
                Include envelope cavity in model (only if ienv=True).
        iext :  bool, optional
                Include external radiation.
        write : bool, optional
                Write new density distribution to model_dir folder. Default is 
                True.
        '''
        recompute = False

        if idisk != None:
            self.modpar.setPar(['idisk',str(idisk),' Include disk in model?',
                                'Disk parameters'])
            recompute = True
        if ienv != None:
            self.modpar.setPar(['ienv',str(idisk),' Include envelope in model?',
                                'Envelope parameters'])
            recompute = True
        if icav != None:
            self.modpar.setPar(['icav',str(icav), 
                                ' Include envelope cavity in model?',
                                'Envelope parameters'])
            recompute = True
            
        if iext !=None:
            self.modpar.setPar(['iext',str(iext),'Include external radiation?',
                                'Radiation sources'])

        if recompute:
            self._setDustDensity()
            
        if write:
            self.write2folder()
            
        return 0

    def write2folder(self, write_param=False):
        '''
        Write RADMC-3D model to model_dir folder.

        Parameters
        ----------
        write_param : bool, optional
                If True overwrite parameter file in folder, given that file 
                already exists. Note that parameter file is written regardless, 
                if it is not already present in folder. Default is False.
        '''
        current_dir = os.path.realpath('.')

        # Create folder if necessary
        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir)
        os.chdir(self.model_dir)

        # Suppress radmc3dPy prints
        blockPrint()

        # Frequency grid
        self.grid.writeWavelengthGrid(old=False)
        # Spatial grid
        self.grid.writeSpatialGrid(old=False)
        # Internal radiation field
        self.radsource.writeStarsinp(old=False)

        # Dust density distribution
        self.data.writeDustDens(binary=self.binary, old=False)

        # radmc3d.inp
        radmc3dPy.setup.writeRadmc3dInp(modpar=self.modpar)

        # Master dust opacity file
        self.opac.writeMasterOpac(ext=self.opac_files, 
                             scattering_mode_max=
                             self.modpar.ppar['scattering_mode_max'], 
                             old=False)

        # if parameter file doesn't exist write as well.
        if (not os.path.isfile('problem_params.inp')) or write_param:
            self.modpar.writeParfile('problem_params.inp')

        # Enable print messages on standard output
        enablePrint()

        if self.modpar.ppar['iext']:
            self._writeSourceExternal()

        os.chdir(current_dir)

        #
        # The following methods take care of finding self.model_dir themselves
        #
        # Write opacity if it was computed 
        self.writeOpac()
        # Copy opacity files if they are specified in parameters
        self.copyOpac(path=self.resource_dir)

        return 0

    def infoModelParams(self):
        '''
        Print model parameters (grid, star, disk, envelope) and computed values 
        (disk and envelope mass) to standard output.
        '''
        ppar = self.modpar.ppar
    
        # Stellar properties:
        print ("\nINFO [{:06}]: Model parameters\n".format(self.ID))
        print("\nStellar properties:\n")
        print(" T_star = {:.2f} K".format(ppar['tstar'][0]))
        print(" R_star = {:.2f} Rsol = {:.2f} AU".format(ppar['rstar'][0]/nc.rs, 
                                                        ppar['rstar'][0]/nc.au))
        print(" L_star = {:.2E} Lsol".format(self.Lstar))
   
        # Grid parameters
        print("\nGrid parameters:\n")
        print(" coordinate system: {}".format(ppar['crd_sys']))
        print(" x coordinate boundaries: {} in AU".format(np.array(ppar['xbound']) 
                                                          / nc.au))
        print(" nx = {}".format(ppar['nx']))
        print(" y coordinate boundaries: {} in rad".format(ppar['ybound']))
        print(" ny = {}".format(ppar['ny']))
        if ('zbound' in ppar.keys()):
            print(" z coordinate boundaries: {}".format(ppar['zbound']))
            print(" nz = {}".format(ppar['nz']))
        else:
            print(" z coordinate is not activated (2D model)!")
        if ('xres_nlev' in ppar.keys()):
            print(" Refinement along x axis:")
            print(" in {} steps".format(ppar['xres_nstep']))
        if (ppar["ngs"] == 2) and ppar["idisk"] and ppar["ienv"]:
            print(" Different opacity in disk and envelope!")
   
        # Wavelength grid
        print("\nWavelength grid:\n")
        print(" Wavelength ranges {} in micron".format(ppar['wbound']))
        print(" Bin number in range {}".format(ppar['nw']))
    
        # Envelope parameters:
        print("\nEnvelope parameters:\n")
        if ppar["ienv"] == True:
            print(" {:s} envelope is included!".format(ppar['modeEnv']))
            print(" Dust density at {:.2f} AU = {:.2E} g/cm^3".format(ppar['r0Env'] 
                                                                 / nc.au, 
                                                                 ppar['rho0Env']))
            if  (ppar['modeEnv'] != 'Ulrich1976'):
                print(" Density power law index = {:.2f}".format(ppar['prhoEnv']))
                print(" Truncation radius = {:.2f} AU".format(ppar['rTrunEnv'] / nc.au))
                print(" Density reduction fact. within tranc. radius = {:.2E}".format(   \
                                                         ppar['redFactEnv']))
            if ppar['icav'] == True:
                print("\n Model contains cavity with parameters:")
                print(" Cavity mode: {:s}".format(ppar['modeCav']))
                if 'thetac_deg' in ppar.keys():
                    thetDegCav = ppar['thetac_deg']
                elif 'thetDegCav' in ppar.keys():
                    thetDegCav = ppar['thetDegCav']
                print(" Opening angle = {:.2f} deg".format(thetDegCav))
                print(" Density reduction factor = {:.2E}".format(ppar['redFactCav']))
                
            print("\n Menv (dust, <3000 au) = {:.2E} Msol".format(self.menv3000/nc.ms))
            print(" Menv (dust, final) = {:.2E} Msol".format(self.menv/nc.ms))
      
        else:
            print(" *NO* envelope is included!")
    
        # Disk parameters:
        print("\nDisk parameters:\n")
        if ppar["idisk"] == True:
            print(" Disk is included in the model!")
            if ('mdisk' in ppar.keys()): 
                print("Mdisk (dust) = {:.2E} Msol".format(ppar['mdisk'] / nc.ms))
            else:
                print(" Mdisk is not set, using Sig(0)")
                if ('sig0' in ppar.keys()): 
                    print(" Sig0(dust,Rsig) = {:.2E} g/cm^2".format(ppar['sig0']))
                else:
                    raise ValueError("Keyword sig0 is not set!")
                if ('rsig' in ppar.keys()): 
                    print(" Rsig = {:.2f} AU".format(ppar['rsig'] / nc.au))
                else: 
                    print(" Rsig = {:.2f} AU".format(ppar['rdisk'] / nc.au))
            print(" Rin = {:.2f} AU".format(ppar['rin'] / nc.au))
            print(" Rdisk = {:.2f} AU".format(ppar['rdisk'] / nc.au))
            print(" Hrpivot = {:.2f} AU".format(ppar['hrpivot'] / nc.au))
            print(" Hr(Hrpivot) = {:.2E}".format(ppar['hrdisk']))
            print(" Power law of H = {:.2f}".format(ppar['plh']))
            print(" Power law of surface density = {:.2f}".format(ppar['plsig1']))
            print(" Dust-to-gas = {:.2E} (not used)".format(ppar['dusttogas']))
        
            # Print the final mass and the sig0
            print(" Mdisk (dust, final) = {:.2E} Msol".format(self.mdisk/nc.ms))
            print(" Sig0(rsig) (dust, final) = {:.2E} g/cm^2".format(self.sig0))
        else:
            print(" *NO* disk is included!")
        
        return 0
    
    def copyOpac(self, path=None):
        '''
        Copy opacity file(s) from resource_dir folder to model_dir folder.
        
        Parameters
        ----------
        path :  string
                Absolute or relative path to the main model folder containing the 
                opacity files. If not set, then use resource_dir class variable.
        '''
        if path is None:
            path = self.resource_dir
        
        current_dir = os.path.realpath('.')
        os.chdir(self.model_dir)

        par = self.modpar.ppar

        if 'dustkappa_ext' in par.keys():

            if par['dustkappa_ext'] == self.opac_files:

                for ext in par['dustkappa_ext']:
                    fname = "dustkappa_{}.inp".format(ext.strip())
            
                    try:
                        copyfile("{}/{}".format(path,fname), "./{}".format(fname) )
                    except:
                        print( "ERROR [{:06}] : copyOpac(): failed to copy {}".format(self.ID, fname) )

        else:
            # Nothing to do
            pass

        os.chdir(current_dir)
        
        return 0
        
    def writeOpac(self):
        '''
        Write dust opacity data to model_dir. The dust opacity should be first 
        computed using the computeOpac() class method. 
        
        The opacity data should be stored in the opac class variable and the 
        output file extension (in the form dustkappa_xxx.inp) is stored in the 
        opac_files class variable (list if multiple species are used).
        '''
        current_dir = os.path.realpath('.')
        os.chdir(self.model_dir)
        
        # Start only if opacity is stored in object
        if self.opac.wav:
            
            ngpop = self.modpar.ppar['ngpop']
            
            for i in range(ngpop):
                if self.verbose:
                    print ('INFO [{:06}]: Writing dustkappa_{}.inp'.format(self.ID, self.opac_files[i]))
                self.opac.writeOpac(ext=self.opac_files[i], idust=i)
        
        else:
            
            print ('WARN [{:06}]: No dust opacity found. Run computeOpac() first!')
            pass
        
        os.chdir(current_dir)
        
        return 0
        
    def runModel(self, bufsize=500000, nthreads=1, radmc3dexec=None,
                 mctherm=True, noscat=None, nphot_therm=None, nphot_scat=None,
                 nphot_mcmono=None, impar=None, verbose=None, time=False,
                 get_tdust=True):
        '''
        Run Monte Carlo dust radiation transport to determine dust temperature 
        and / or compute image according impar dictionary.
        
        Parameters
        ----------
        bufsize : int
                buffer size in bytes used for the communication with the RADMC-3D 
                child process. If large images are computed then it needs to be 
                increased. Default is 500000.
        nthreads : int
                Number of threads used in the thermal Monte Carlo calculation. 
                Default is 1.
        readmc3dexec : string
                Path to the RADMC3D binary. If not set, then assume it to be 
                reachable in $PATH.
        mctherm : bool, optional
                If True, then run thermal Monte Carlo computation. Default is 
                True.
        noscat : int, optional
                Switch off scattering process even if dust opacity file contains 
                scattering cross sections. If None, then option from radmc3d.inp 
                is used. Default is None.
        nphot_therm : int, optional
                Set number of thermal photon packages. If None, then option from 
                radmc3d.inp is used. Default is None.
                Note: as of RADMC-3D version 0.41 setting nphot_therm in child 
                mode is not supported. If code crashes, set this to None and 
                adjust nphot_therm in the radmc3dPy parameter file.
        nphot_scat : int, optional
                Set number of photon packages used for scattering. If None, then 
                option from radmc3d.inp is used. Default is None.
        nphot_mcmono : int, optional
                Set number of photon packages in monochromatic MC simulation. 
                If None, then option from radmc3d.inp is used. Default is None.
        impar : dict or list of dict, optional
                Image parameter(s). Known keywords are listed in the runImage()
                method description. At least the wavelength (wav keyword) must 
                be set for each images. Default is None.
        get_tdust : bool
                If True then read the dust temperature to data class variable.
                If only image, SED and/or chi2 is needed the set to False.
                Default is True.
        verbose : bool, optional
                Print INFO messages to standard output. If set to None, then 
                use verbose class variable. Default is None.
        time :  bool, optional
                Prints function runtime information. Useful for profiling.
                Default is False.
        '''
        # If verbosity not set then use global class variable
        if verbose is None:
            verbose = self.verbose
        
        current_dir = os.path.realpath('.')
        os.chdir(self.model_dir)
        
        # Initialize radmc3dRunner
        self.rrun = runner.radmc3dRunner(model_dir=self.model_dir, bufsize=bufsize,
                                         nthreads=nthreads, radmc3dexec=None, 
                                         ID=self.ID, verbose=verbose)
        
        # Compute dust temperature
        if mctherm:
            self.rrun.runMCtherm(noscat=noscat, 
                                 nphot_therm=nphot_therm,
                                 nphot_scat=nphot_scat, 
                                 nphot_mcmono=nphot_mcmono,
                                 verbose=verbose, time=time)
        
        # Compute image(s)
        if impar is not None:
            
            if type(impar) == dict:
                impar = [impar]
        
            self.image = []
            
            for ip in impar:
                img = self.rrun.getImage(verbose=verbose, time=time, **ip)
                if img.nwav == 1:
                    self.image.append(img)
                else:
                    # If multi wavelength image, then unpack
                    for jp in range(img.nwav):
                        tmp = copy.deepcopy(img)
                        tmp.nwav = 1
                        tmp.nfreq = 1
                        tmp.wav = np.array([img.wav[jp]])
                        tmp.image = img.image[:,:,jp].reshape((img.nx,img.ny,1))
                        tmp.imageJyppix = img.imageJyppix[:,:,jp].reshape((img.nx,img.ny,1))
                        self.image.append(tmp)

        # TODO: Compute SED(s) if needed

        # Terminate radmc3dRunner
        self.rrun.terminate(verbose=verbose)
        
        # Read dust temperature, if necessary
        if get_tdust:
            blockPrint()
            self.data.readDustTemp(binary=self.binary)
            enablePrint()
        
        os.chdir(current_dir)
        
        return 0

    def getVis(self, uvdata, dpc=1.0, PA=None, dRA=None, dDec=None, chi2_only=False, 
               galario_check=False, verbose=None, time=False):
        '''
        Compute visibility of previously computed images and their chi2 
        compared to observations, using the Galario library.
        
        When multiple visibility datasets are provided, then equal number of 
        corresponding images need to exist in the self.image class variable.
        The order of visibility datasets and the order of images must match.
        
        Parameters
        ----------
        uvdata : dict or list of dict 
                Containing observed visibility data. The 'u', 'v', 'Re', 'Im', 
                'w' and 'wav' keywords need to be defined.
        dpc  :  float
                Distance to object in unit of parsec, Default is 1.0.
        PA   :  float or list of floats, optional
                Position angle in radian. If multiple images are processed, then 
                each image may use a different PA value, if a list is provided. 
                The order used is the same as in the uvdata list. The PA value 
                provided in uvdata will be overruled by the value provided as 
                an argument in the function call. If PA is not set in uvdata 
                and PA is None in the function call, then PA = 0.0 is used.
                Default is None.
        dRA  :  float or list of floats, optional
                Offset in RA in radian. If multiple images are processed, then 
                each image may use a different dRA value, if a list is provided. 
                The order used is the same as in the uvdata list. The dRA value 
                provided in uvdata will be overruled by the value provided as 
                an argument in the function call. If dRA is not set in uvdata 
                and dRA is None in the function call, then dRA = 0.0 is used.
                Default is None.
        dDec :  float or list of floats, optional
                Offset in Dec in radian. If multiple images are processed, then 
                each image may use a different PA value, if a list is provided. 
                The order used is the same as in the uvdata list. The dDec value 
                provided in uvdata will be overruled by the value provided as 
                an argument in the function call. If dDec is not set in uvdata 
                and dDec is None in the function call, then dDec = 0.0 is used.
                Default is None.
        chi2_only : bool
                If True then the synthetic visibility itself is not computed and 
                stored (a zero value array is stored instead). The chi2 is still
                computed and stored. Set this to True when running MCMC fitting 
                in order to improve speed. Default is False.
        galario_check : bool
                Check whether image and dxy satisfy Nyquist criterion for 
                computing the synthetic visibilities in the (u, v) locations 
                provided (see galario documentation). Default is False.
        verbose : bool, optional
                Print INFO messages to standard output. Default is False.
        time :  bool, optional
                Prints function runtime information. Useful for profiling.
                Default is False.
        '''
        # If verbosity not set then use global class variable
        if verbose is None:
            verbose = self.verbose

        if time:
            start = timer()

        if self.image is None:

            print ("WARN [{:06}]: No images stored in object!".format(self.ID)) 
            pass

        else:

            if type(uvdata) == dict:
                uvdata = [uvdata]

            n_uvdata = len(uvdata)

            # Set default offsets
            if type(PA) is list:
                n_PA = len(PA)
                if n_PA != n_uvdata:
                    raise ValueError('PA list ({}) and uvdata ({})element numbers do not match!'.format(
                        n_PA,n_uvdata))
            if type(dRA) is list:
                n_dRA = len(dRA)
                if n_dRA != n_uvdata:
                    raise ValueError('dRA list ({}) and uvdata ({})element numbers do not match!'.format(
                        n_dRA,n_uvdata))
            if type(dDec) is list:
                n_dDec = len(dDec)
                if dDec != n_uvdata:
                    raise ValueError('dDec list ({}) and uvdata ({})element numbers do not match!'.format(
                        n_dDec,n_uvdata))

            # Set galario threads to 1, parallelisation is done by emcee and MPI
            galario.double.threads(1)

            for i in range(len(uvdata)):
                
                uv = uvdata[i]
                
                # extract uv data from dictionary
                u = np.ascontiguousarray(uv['u'])
                v = np.ascontiguousarray(uv['v'])
                Re = np.ascontiguousarray(uv['Re'])
                Im = np.ascontiguousarray(uv['Im'])
                w = np.ascontiguousarray(uv['w'])
                wav = uv['wav']
                
                uv_keywords = uv.keys()
                
                # Set offsets
                if PA is not None:
                    if type(PA) is list:
                        PA_use = PA[i]
                    else:
                        PA_use = PA
                elif 'PA' in uv_keywords:
                    PA_use = uv['PA']
                else:
                    PA_use = 0.0
                if dRA is not None:
                    if type(dRA) is list:
                        dRA_use = dRA[i]
                    else:
                        dRA_use = dRA
                elif 'dRA' in uv_keywords:
                    dRA_use = uv['dRA']
                else:
                    dRA_use = 0.0
                if dDec is not None:
                    if type(dDec) is list:
                        dDec_use = dDec[i]
                    else:
                        dDec_use = dDec
                elif 'dDec' in uv_keywords:
                    dDec_use = uv['dDec']
                else:
                    dDec_use = 0.0

                # Use always the nth image for the nth visibility data set
                iim = i
                
                # Warn if the wavelength does not match
                wav_im = self.image[iim].wav[0]
                if wav != wav_im:
                     print ('WARN [{:06}]: {}th image and visibility data wavelength does not match! ({}, {})'.format(self.ID, i, wav, wav_im))

                if self.vis_mod is None:
                    self.vis_mod = []
                if self.vis_inp is None:
                    self.vis_inp = []
                if self.chi2 is None:
                    self.chi2 = np.empty(0)
                if self.nvis is None:
                    self.nvis = np.empty(0)

                wle = wav * 1.0e-6

                imJyppix = self.image[iim].imageJyppix[:,:,0] / dpc**2

                dxy = self.image[iim].sizepix_x / nc.au / dpc / 3600. * galario.deg

                print ("Image: {} micron, PA: {}, dRA: {}, dDec: {}".format(wav, 
                                                                            PA_use,
                                                                            dRA_use,
                                                                            dDec_use))

                # Compute visibility
                if chi2_only:
                    vis = np.zeros_like(u, dtype=np.complex)
                else:
                    vis = galario.double.sampleImage( imJyppix, dxy, u/wle, 
                                                     v/wle, dRA=dRA_use, 
                                                     dDec=dDec_use, PA=PA_use,
                                                     check=galario_check)

                # Store visibility (or 0 array)
                self.vis_mod.append( uvplot.UVTable(uvtable=[u, v, vis.real,
                                     vis.imag, w], wle=wle,
                                     columns=uvplot.COLUMNS_V0) )
                self.vis_inp.append( uvplot.UVTable(uvtable=[u, v, Re,
                                     Im, w], wle=wle,
                                     columns=uvplot.COLUMNS_V0) )

                # Compute chi2
                chi2 = galario.double.chi2Image( imJyppix, dxy, u/wle, v/wle, 
                                                 Re, Im, w, dRA=dRA_use, 
                                                 dDec=dDec_use, PA=PA_use, 
                                                 check=galario_check )
                # Number of (u,v) pairs
                nvis = u.shape[0]

                self.chi2 = np.append(self.chi2, chi2)
                self.nvis = np.append(self.nvis, nvis)

        if time:
            end = timer()
            dt = end-start

        if verbose and time:
            print ('INFO [{:06}]: Visibility and/or chi^2 computed in {:.2f} s!'.format(
                            self.ID, dt))
        elif verbose:
            print ('INFO [{:06}]: Visibility and/or chi^2 computed!'.format(
                self.ID))

        return 0

    def cleanModel(self):
        '''
        Delete model directory
        '''

        current_dir = os.path.realpath('.')

        if current_dir != self.model_dir:

            rmtree(self.model_dir)

        else:
            print("WARN [{:06}]: Folder cannot be deleted!".format(self.ID))

        os.chdir(current_dir)

        return 0

    def computeOpac(self):
        '''
        Compute dust opacity on the fly according to parameters defined in the 
        self.modpar object.

        The code search the lnk file in the current folder first. If not found 
        then it looks for it in the resource_dir folder.

        Currently single grain size is supported and the code requires the 
        Fortran implementation of the Mie scattering code in radmc3dPy.
        '''
        par = self.modpar.ppar

        if self.grid is None:
            raise ValueError('ERROR [computeOpac()]: set up model grid first!')
        else:
            lamcm = self.grid.wav * 1.0e-4

        if 'lnk_fname' in par.keys():

            lnk_fname = par['lnk_fname']

            if isinstance(lnk_fname, str):
                lnk_fname = [lnk_fname]

            if 'gdens' in par.keys():
                matdens = par['gdens']
                if type(matdens) is not list:
                    matdens = [matdens]
            else:
                matdens = [3.0]
                print ("WARN [{:06}]: gdens not defined in parameter file, \
                        using {:.2} g/cm^3".format(self.ID, matdens[0]))

            if 'gsmin' in par.keys():
                gsmin = par['gsmin']
                if type(gsmin) is not list:
                    gsmin = [gsmin]
            else:
                gsmin = [0.1 * 1.0e-4] # 0.1 micron converted to cm
                print ("WARN [{:06}]: gsmin not defined in parameter file, \
                           using {:.2} cm".format(self.ID, gsmin[0]))

            if 'gsmax' in par.keys():
                gsmax = par['gsmax']
                if type(gsmax) is not list:
                    gsmax = [gsmax]
            else:
                gsmax = [0.1 * 1.0e-4] # 0.1 micron converted to cm
                print ("WARN [{:06}]: gsmax not defined in parameter file, \
                           using {:.2} cm".format(self.ID, gsmax[0]))

            if 'ngs' in par.keys():
                ngs = par['ngs']
                if type(ngs) is not list:
                    ngs = [ngs]
            else:
                ngs = [1]
                print ("WARN [{:06}]: ngs not defined in parameter file, \
                           using {:.2} grain sizes".format(self.ID, ngs[0]))

            if 'gsdist_powex' in par.keys():
                gsdist_powex = par['gsdist_powex']
                if type(gsdist_powex) is not list:
                    gsdist_powex = [gsdist_powex]
            else:
                gsdist_powex = [-3.5]    # assumed ISM value
                print ("WARN [{:06}]: gsdist_powex not defined in parameter file, \
                           using {:.2}".format(self.ID, gsdist_powex[0]))

            if 'ngpop' in par.keys():
                ngpop = par['ngpop']
            else:
                ngpop = 1
                print ("WARN [{:06}]: ngpop not defined in parameter file, \
                           using {:2} dust species".format(self.ID, ngpop))

            # If multiple grain populations are computed, then make sure that all 
            # required input list has enough elements:
            if ngpop > 1:
                if len(lnk_fname) == 1: lnk_fname = lnk_fname * ngpop
                if len(matdens) == 1  : matdens = matdens * ngpop
                if len(gsmax) == 1    : gsmax = gsmax * ngpop
                if len(gsmin) == 1    : gsmin = gsmin * ngpop
                if len(ngs) == 1      : ngs = ngs * ngpop
                if len(gsdist_powex) == 1 : gsdist_powex = gsdist_powex * ngpop

            # Loop over grain populations
            for i in range(ngpop):

                # Loop variables
                lnk = lnk_fname[i]
                ngb = ngs[i]
                amin = gsmin[i]
                amax = gsmax[i]
                mtd = matdens[i]
                pla = gsdist_powex[i]

                # Grain sizes
                agr = np.logspace(np.log10(amin), np.log10(amax), ngb)

                # Compute weighting factors
                pwgt = (pla - 2.0) / 3.0 + 2.0
                mgra = (4.0/3.0*np.pi) * agr**3 * mtd
                dum  = (mgra / mgra[0])**pwgt
                mwgt = dum / np.sum(dum)

                # Find file in local or main directory
                if os.path.isfile(lnk):
                    fname = lnk
                else:
                    fname = "{}/{}".format(self.resource_dir,lnk)

                kabs = np.zeros_like(lamcm)
                ksca = np.zeros_like(lamcm)
                gsca = np.zeros_like(lamcm)

                # Loop over grain sizes
                for j in range(ngb):

                    opac_tmp = radmc3dPy.miescat.compute_opac_mie(fname=fname,
                                                             matdens=mtd,
                                                             agraincm=agr[j],
                                                             lamcm=lamcm,
                                                             extrapolate=True)

                    kabs += mwgt[j] * opac_tmp['kabs']
                    ksca += mwgt[j] * opac_tmp['kscat']
                    gsca += mwgt[j] * opac_tmp['gscat']

                # Save grain population data
                self.opac.kabs.append( kabs )
                self.opac.ksca.append( ksca )
                self.opac.wav.append( opac_tmp['lamcm'] * 1.0e4 )
                self.opac.nwav.append( len(opac_tmp['lamcm']) )
                self.opac.freq.append( nc.cc / opac_tmp['lamcm'] )
                self.opac.nfreq.append( len(opac_tmp['lamcm']) )
                self.opac.scatmat.append( False )
                self.opac.phase_g.append( gsca )
                self.opac.idust.append( i )

                self.opac.ext.append("{:.5s}_amin_{:06.3E}_amax_{:06.3E}_pl{:04.2}".format(lnk,amin,amax,pla))

            # Save computed opacities
            self.opac_files = self.opac.ext

        else:
            # No lnk_fname specified, continue
            pass

        return 0

def getParams(paramfile=None):
    '''
    Set default parameter values and/or read user defined parameter from file.
    
    Similar to radmc3dPar readPar and loadDefaults methods. The default 
    parameters are tuned for the FAUST program.
    
    Parameters
    ----------
    paramfile : string, optional
                Name of parameter file (usually problem_params.inp). If file is 
                not set (None) then load defaults. Default is None.
    '''

    # Read the parameters from the problem_params.inp file 
    modpar = radmc3dPy.analyze.radmc3dPar()

    if paramfile is not None:
        modpar.readPar(fname=paramfile)
    else:
        # Set radmc3dPy default
        #modpar.loadDefaults()
        # Set SimpleDiskEnv defaults

        # Radiation sources
        modpar.setPar(['pstar','[0.0, 0.0, 0.0]', 
                       ' Position of the star(s) (cartesian coordinates)', 
                       'Radiation sources'])
        modpar.setPar(['mstar', '[3.0*ms]', 
                       ' Mass of the star(s)', 
                       'Radiation sources'])
        modpar.setPar(['rstar','[5.9*rs]', 
                       ' Radius of the star(s)', 
                       'Radiation sources'])
        modpar.setPar(['tstar','[4786.0]', 
                       ' Effective temperature of the star(s) [K]', 
                       'Radiation sources'])
        modpar.setPar(['staremis_type','["blackbody"]', 
                       ' Stellar emission type ("blackbody", "kurucz", "nextgen")', 
                       'Radiation sources'])
        modpar.setPar(['iext','False','Include external radiation?',
                       'Radiation sources'])
        modpar.setPar(['G','1.7','ISRF strength in units of G0',
                       'Radiation sources'])

        # Grid parameters
        modpar.setPar(['crd_sys', "'sph'", 
                       ' Coordinate system used (car/cyl)', 
                       'Grid parameters']) 
        modpar.setPar(['nx', '[30,100]', 
                       ' Number of grid points in the first dimension (to switch off this dimension set it to 0)', 
                       'Grid parameters']) 
        modpar.setPar(['ny', '80', 
                       ' Number of grid points in the second dimension (to switch off this dimension set it to 0)', 
                       'Grid parameters'])
        modpar.setPar(['nz', '0', 
                       ' Number of grid points in the third dimension (to switch off this dimension set it to 0)', 
                       'Grid parameters'])
        modpar.setPar(['xbound', '[1.0*au, 1.05*au, 100.*au]', 
                       ' Boundaries for the x grid', 'Grid parameters'])
        modpar.setPar(['ybound', '[0.0, pi/2.]', 
                       ' Boundaries for the y grid', 
                       'Grid parameters'])
        modpar.setPar(['zbound', '[0.0, 2.0*pi]', 
                       ' Boundraries for the z grid', 
                       'Grid parameters'])
        modpar.setPar(['nw', '[50, 150, 100]', ' Number of points in the \
                       wavelength grid', 'Grid parameters'])
        modpar.setPar(['wbound', '[0.1, 7.0, 25.0, 1e4]', 
                       ' Boundaries for the wavelength grid', 
                       'Grid parameters'])

        # Dust opacity
        modpar.setPar(['lnk_fname', '"astro_sill_draine2003.lnk"', ' ', 
                       'Dust opacity'])
        modpar.setPar(['gdens', '3.5', 
                       ' Bulk density of the materials in g/cm^3', 'Dust opacity'])
        modpar.setPar(['gsmin', '1.0e-5', ' Minimum grain size [cm]', 'Dust opacity'])
        modpar.setPar(['gsmax', '1.0e-5', ' Maximum grain size [cm]', 'Dust opacity'])
        modpar.setPar(['ngs', '1', ' Number of grain size bins', 'Dust opacity'])
        modpar.setPar(['gsdist_powex', '1', ' Power law index of grain size distribution', 'Dust opacity'])
        modpar.setPar(['ngpop', '1', ' Number of grain populations', 'Dust opacity'])
        modpar.setPar(['ngs', '1', ' Number of grain size bins', 'Dust opacity'])

        # Code parameters
        modpar.setPar(['scattering_mode_max', '0', 
                       ' 0 - no scattering, 1 - isotropic scattering, 2 - anizotropic scattering', 'Code parameters'])
        modpar.setPar(['istar_sphere', '1', 
                       ' 1 - take into account the finite size of the star, 0 - take the star to be point-like', 'Code parameters'])
        modpar.setPar(['itempdecoup', '1', 
                       ' Enable for different dust components to have different temperatures', 'Code parameters'])
        modpar.setPar(['tgas_eq_tdust', '1', 
                       ' Take the dust temperature to identical to the gas temperature', 'Code parameters'])
        modpar.setPar(['modified_random_walk', '1', 
                       ' Switched on (1) and off (0) modified random walk', 'Code parameters'])
        modpar.setPar(['rto_style', '1', 
                       ' Space-dependent output format (1) ASCII, F77 unformatted (2) and off (3) binary', 'Code parameters'])
        
        # Envelope parameters
        modpar.setPar(['ienv','True',' Include envelope in model?',
                       'Envelope parameters'])
        modpar.setPar(['icav','False',' Include envelope cavity in model?'
                       'Envelope parameters'])
        modpar.setPar(['bgdens', '0.0e0', ' Background density (g/cm^3)', 
                       'Envelope parameters'])
        modpar.setPar(['dusttogas', '1.0e-2', ' Dust-to-gas mass ratio', 
                       'Envelope parameters'])
        modpar.setPar(['modeEnv', "'Ulrich1976'", 
                       " Choose envelope model, options: ['Ulrich1976','Tafalla2004','powerlaw']", 
                       'Envelope parameters'])
        modpar.setPar(['rho0Env', '4.e-20', 
                       ' New central density g/cm^3 dust density volume', 
                       'Envelope parameters'])
        modpar.setPar(['r0Env', '300.0*au', 
                       " Flattening radius in 'Tafalla2004' or centrifugal radius in 'Ulrich1976' models", 'Envelope parameters'])
        modpar.setPar(['rTrunEnv', '30.0*au', ' Truncation radius', 
                       'Envelope parameters'])
        modpar.setPar(['redFactEnv', '1.0e-2', 
                       ' Density is reduced by this factor if r < rTrunEnv', 
                       'Envelope parameters'])

        # Disk parameters
        modpar.setPar(['idisk','True',' Include disk in model?',
                       'Disk parameters'])
        modpar.setPar(['mdisk', '0.01*ms', ' Disk mass', 
                       'Disk parameters'])
        modpar.setPar(['rin', '1.0*au', ' Inner disk radius', 
                       'Disk parameters'])
        modpar.setPar(['rdisk', '50.0*au', ' Outer disk radius', 
                       'Disk parameters'])
        modpar.setPar(['hrdisk', '0.1', 
                       ' Ratio of the pressure scale height over radius at hrpivot', 
                       'Disk parameters'])
        modpar.setPar(['hrpivot', '50.0*au', 
                       ' Reference radius at which Hp/R is taken',
                       'Disk parameters'])
        modpar.setPar(['plh', '2.0/7.0', ' Flaring index', 
                       'Disk parameters'])
        modpar.setPar(['plsig1', '-1.0', 
                       ' Power exponent of the surface density distribution as a function of radius', 
                       'Disk parameters'])
        modpar.setPar(['plsig2', '-40.0', 
                       ' Power law exponent at r > rdisk (abrubt cutoff at rdisk is not realistic)', 
                       'Disk parameters'])

    # Return
    return modpar

def blockPrint():
    '''
    Directs print function output to dev0, preventing messages from appearing 
    on standard output.
    The purpose of the function is to suppress radmc3dPy output when it is not 
    necessary.
    '''
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    '''
    Directs print function output to standard output. Call function always after 
    calling blockPrint() when you want to restore print function output.
    '''
    sys.stdout = sys.__stdout__
