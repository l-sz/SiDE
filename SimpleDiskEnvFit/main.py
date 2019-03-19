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
import numpy as np
from shutil import copyfile, rmtree

import radmc3dPy
import radmc3dPy.natconst as nc

# Internal routines
from . import models
from . import runner

import galario
import uvplot

__all__ = ['radmc3dModel','getParams']

global module_dir

module_dir = os.path.dirname(sys.modules['SimpleDiskEnvFit'].__file__)

class radmc3dModel:
    '''
    Interface for creating, running and analysing RADMC-3D models
    
    The class provides storage, read-in and write-out routines. Plotting 
    routines should be added in the future.
    '''
    
    ID = None
    
    # radmc3dPy objects
    modpar = None
    grid = None
    data = None
    radsource = None
    opac = None
    opac_files = None
    
    # Model specific data
    model_dir = None
    resource_dir = None
    
    # Computed model parameters
    mdisk = 0.0
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
   
    # Fitting parameters
    prediction = None
    
    # runner
    rrun = None
    
    def __init__(self, modpar=None, model_dir=None, resource_dir=None, 
                 idisk=True, ienv=True, icav=False, iext=False,
                 write2folder=False, ID=None, verbose=False):
        '''
        
        If folder parameter is set then it overwrites the value specified in 
        modpar parameter list. If folder parameter is not set in initialization 
        call, then use the value from modpar. If folder is set neither as parameter 
        nor in modpar, then default to '.' current folder (using absolute path).
        
        '''
        # Set model ID
        if ID is None:
            self.ID = np.random.randint(0,99999)
        else:
            self.ID = ID
        
        # Create empty model
        self.grid = radmc3dPy.analyze.radmc3dGrid()
        self.data = radmc3dPy.analyze.radmc3dData(self.grid)
 
        # Set parameters
        if modpar is not None:
            self.modpar = modpar
        else:
            self.modpar = getParams()
        
        if model_dir is not None:
            self.model_dir = model_dir
        elif "model_dir" in self.modpar.ppar.keys():
            self.model_dir = self.modpar.ppar["model_dir"]
        else:
            self.model_dir = os.path.realpath('.')
        
        print (self.model_dir)
        
        if resource_dir is None:
            
            self.resource_dir = '{}/{}'.format(module_dir,'/lnk_files')
            
        # Update model_dir parameter in modpar
        self.modpar.setPar(["model_dir", "'{}'".format(str(self.model_dir)), 
                            " model folder path",
                            "Model carolina"])
        
        # Update control parameters
        self.modpar.setPar(['idisk',str(idisk),' Include disk in model?',
                                'Disk parameters'])
        
        self.modpar.setPar(['ienv',str(ienv),' Include envelope in model?',
                            'Envelope parameters'])
        
        self.modpar.setPar(['icav',str(icav), 
                            ' Include envelope cavity in model?',
                            'Envelope parameters'])
            
        self.modpar.setPar(['iext',str(iext),'Include external radiation?',
                            'Radiation sources'])

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
            infoModelParams(modpar=self.modpar)
        
        # Done

    def readModel(self, model_dir=None):
        '''
        Read model from folder (parameter file, grid, density)
        
        Parameters
        ----------
          model_dir  -  string, path to model to be read
        '''
        print ('INFO [{:06}]: This function is not implemented yet!'.format(ID))
        
        return 0

    def updateModel(self, modpar=None, model_dir=None, verbose=False):
        '''
        Updates model density distribution and location according to
        set parameters.
        
        See also switchComponents() method.
        
        Parameters
        ----------
          modpar   -  radmc3dPy.radmc3dPar class object containing the new 
                      or updated model parameter values
          model_dir-  string, new model folder name
          verbose  -  if True then write parameter summary to screen
        '''
        self.__init__(modpar=modpar, model_dir=model_dir, verbose=verbose)
        
        return 0
        
    def _setRadSources(self):
        '''
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
        Creates wavelength and spatial grids using internal radmc3dPy functions
        '''
        
        # Create the wavelength grid
        self.grid.makeWavelengthGrid(ppar=self.modpar.ppar)
        
        # Create the spatial grid
        self.grid.makeSpatialGrid(ppar=self.modpar.ppar)
        
        return 0

    def _setDustDensity(self):
        '''
        Computes dust density distribution
        '''
        ppar = self.modpar.ppar   # in order to shorten code
   
        # Compute density distributions
        rho_disk_dust = np.zeros([self.grid.nx, self.grid.ny, self.grid.nz, 1], 
                                 dtype=np.float64)
        rho_env_dust = np.zeros([self.grid.nx, self.grid.ny, self.grid.nz, 1], 
                                dtype=np.float64)
        
        if ppar['idisk'] == True:
            
            rho_disk_dust, self.mdisk, self.sig0 = \
                models.flaring_disk(self.grid, ppar=ppar)

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
        if ppar['ngs'] == 2 and ppar['idisk'] and ppar['ienv']:
            rhodust = np.zeros([self.grid.nx, self.grid.ny, self.grid.nz,
                                ppar['ngs']])
            rhodust[:,:,:,0] = rho_disk_dust[:,:,:,0]
            rhodust[:,:,:,1] = rho_env_dust[:,:,:,0]
            self.data.rhodust = rhodust
        else:
            self.data.rhodust = rho_env_dust + rho_disk_dust
        
        return 0
        
    def _writeSourceExternal(self, fname='external_source.inp'):
        '''
        Writes external radiation file to disk
        
        Parameters
        ----------
          fname  -  string, name of output file (default external_source.inp)
        '''
        print('INFO [{:06}]: Writing {}'.format(self.ID, fname))
        
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
          idisk  -  include disk in model
          ienv   -  include envelope in model
          icav   -  include envelope cavity in model (only if ienv=True)
          iext   -  include external radiation
          write  -  write new density distribution to disk (model_dir)
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
        Write RADMC-3D model to given directory given by the self.model_dir 
        variable.
        
        Parameters
        ----------
          write_param  -  if True overwrite parameter file in folder, given
                          that file already exists. Note that parameter file is
                          written regardless if it is not already present in 
                          folder.
        '''
        current_dir = os.path.realpath('.')

        # Create folder if necessary
        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir)
        os.chdir(self.model_dir)
        
        # Frequency grid
        self.grid.writeWavelengthGrid(old=False)
        # Spatial grid
        self.grid.writeSpatialGrid(old=False)
        # Internal radiation field
        self.radsource.writeStarsinp(old=False)
        
        if self.modpar.ppar['iext']:
            self._writeSourceExternal()
        
        # Dust density distribution
        self.data.writeDustDens(binary=False, old=False)
        
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
        Show model parameters (grid, star, disk, envelope) and computed values 
        (disk and envelope mass).
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
            print(" Density at {:.2f} AU = {:.2E} g/cm^3".format(ppar['r0Env'] 
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
                print(" Opening angle = {:.2f} deg".format(ppar['thetDegCav']))
                print(" Density reduction factor = {:.2E}".format(ppar['redFactCav']))
                
            print("\n Menv (<3000 au) = {:.2E} Msol".format(self.menv3000/nc.ms))
            print(" Menv (final) = {:.2E} Msol".format(self.menv/nc.ms))
      
        else:
            print(" *NO* envelope is included!")
    
        # Disk parameters:
        print("\nDisk parameters:\n")
        if ppar["idisk"] == True:
            print(" Disk is included in the model!")
            if ('mdisk' in ppar.keys()): 
                print("Mdisk (dust+gas) = {:.2E} Msol".format(ppar['mdisk'] / nc.ms))
            else:
                print(" Mdisk is not set, using Sig(0)")
                if ('sig0' in ppar.keys()): 
                    print(" Sig0(Rsig) = {:.2E} g/cm^2".format(ppar['sig0']))
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
            print(" Dust-to-gas = {:.2E}".format(ppar['dusttogas']))
        
            # Print the final mass and the sig0
            print(" Mdisk (final) = {:.2E} Msol".format(self.mdisk/nc.ms))
            print(" Sig0(rsig) (final) = {:.2E} g/cm^2".format(self.sig0))
        else:
            print(" *NO* disk is included!")
        
        return 0
    
    def copyOpac(self, path=None):
        '''
        Copy the opacity file from resource_dir folder to model_dir folder.
        
        This function should be replaced by in place opacity computation!
        
        Parameters
        ----------
        path   --  path to the main model folder containing the opacity files.
                   Should be absolute or relative path to the current model dir.
                   If not set then use resource_dir class variable.
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
        
    def writeOpac(self):
        '''
        '''
        current_dir = os.path.realpath('.')
        os.chdir(self.model_dir)
        
        # Start only if opacity is stored in object
        if self.opac.wav:
            
            ngs = self.modpar.ppar['ngs']
            
            for i in range(ngs):
                print ('INFO [{:06}]: Writing dustkappa_{}.inp'.format(self.ID, self.opac_files[i]))
                self.opac.writeOpac(ext=self.opac_files[i], idust=i)
        
        os.chdir(current_dir)
        
        return 0
        
    def runModel(self, bufsize=500000, nthreads=1, radmc3dexec=None,
                 mctherm=True, noscat=None, nphot_therm=None, nphot_scat=None,
                 nphot_mcmono=None, impar=None, verbose=False):
        '''
        Run Monte Carlo dust radiation transport to determine dust temperature 
        and / or compute image according impar dictionary.
        '''
        current_dir = os.path.realpath('.')
        os.chdir(self.model_dir)
        
        # Initialize radmc3dRunner
        self.rrun = runner.radmc3dRunner(model_dir=self.model_dir, bufsize=500000,
                                         nthreads=1, radmc3dexec=None, 
                                         ID=self.ID)
        
        # Compute dust temperature
        if mctherm:
            self.rrun.runMCtherm(noscat=noscat, 
                                 nphot_therm=nphot_therm,
                                 nphot_scat=nphot_scat, 
                                 nphot_mcmono=nphot_mcmono,
                                 verbose=verbose)
        
        # Compute image(s)
        if impar is not None:
            
            if type(impar) == dict:
                impar = [impar]
        
            self.image = []
            
            for ip in impar:
                img = self.rrun.getImage(**ip)
                self.image.append(img)

        # Compute SED(s) if needed
        #
        #

        # Terminate radmc3dRunner
        self.rrun.terminate(verbose=verbose)
        
        # Read dust temperature
        binary = False
        if 'rto_style' in self.modpar.ppar.keys():
            if self.modpar.ppar['rto_style'] > 1:
                binary = True
        self.data.readDustTemp(binary=binary)
        
        os.chdir(current_dir)
        
        return 0
        
    def getVis(self, uvdata, dpc=1.0, PA=0., dRA=0.0, dDec=0.0):
        '''
        Compute visibility of previously computed images and their chi2 
        compared to observations, using the Galario library.
        
        Parameters
        ----------
          uvdata  -  dictionary or list of dictionaries containing observed 
                     visibility data. The 'u', 'v', 'Re', 'Im', 'w' and 'wav'
                     keywords need to be defined.
          dpc     -  float, distance to object in unit of parsec
          PA      -  position angle (in radian)
          dRA     -  offset in RA (in radian)
          dDec    -  offset in Dec (in radian)
        '''
        
        if self.image is None:
            
            print ("WARN [{:06}]: No images stored in object!".format(self.ID)) 
            pass
            
        else:
            
            if type(uvdata) == dict:
                uvdata = [uvdata]
            
            wav_arr = []
            for im in self.image:
                wav_arr.append(im.wav[0])

            for uv in uvdata:
                
                # extract uv data from dictionary
                u = np.ascontiguousarray(uv['u'])
                v = np.ascontiguousarray(uv['v'])
                Re = np.ascontiguousarray(uv['Re'])
                Im = np.ascontiguousarray(uv['Im'])
                w = np.ascontiguousarray(uv['w'])
                wav = uv['wav']
                
                # Find image for uv wavelength
                try:
                    iim = wav_arr.index(wav)
                except:
                    print ('WARN [{:06}]: micron image not found,\
                           continue...'.format(self.ID, wav))
                    continue
                
                if self.vis_mod is None:
                    self.vis_mod = []
                if self.vis_inp is None:
                    self.vis_inp = []
                if self.chi2 is None:
                    self.chi2 = []

                wle = wav * 1.0e-6
            
                imJyppix = self.image[iim].imageJyppix[:,:,0] / dpc**2
            
                dxy = self.image[iim].sizepix_x / nc.au / dpc / 3600. * galario.deg
        
                vis = galario.double.sampleImage( imJyppix, dxy, u/wle, v/wle,
                                                  check=True )
    
                self.vis_mod.append( uvplot.UVTable(uvtable=[u, v, vis.real,
                                        vis.imag, w], wle=wle,
                                        columns=uvplot.COLUMNS_V0) )
                self.vis_inp.append( uvplot.UVTable(uvtable=[u, v, Re,
                                        Im, w], wle=wle,
                                        columns=uvplot.COLUMNS_V0) )

                chi2 = galario.double.chi2Image( imJyppix, dxy, u/wle, v/wle, 
                                                  Re, Im, w, dRA=dRA, dDec=dDec,
                                                  PA=PA )
                self.chi2.append(chi2)

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
                
            if 'agraincm' in par.keys():
                agraincm = par['agraincm']
                if type(agraincm) is not list:
                    agraincm = [agraincm]
            else:
                agraincm = [0.1 * 1.0e-4] # 0.1 micron converted to cm
                print ("WARN [{:06}]: agraincm not defined in parameter file, \
                           using {:.2} cm".format(self.ID, agraincm[0]))
            
            n_lnk = len(lnk_fname)
            n_mat = len(matdens)
            n_gra = len(agraincm)
            
            ngs = par['ngs']
            
            if ngs > 1:
                if n_lnk == 1: lnk_fname = lnk_fname * ngs
                if n_mat == 1: matdens = matdens * ngs
                if n_gra == 1: agraincm = agraincm * ngs
            
            for i in range(ngs):
                
                lnk = lnk_fname[i]
                agr = agraincm[i]
                mtd = matdens[i]
                
                # Find file in local or main directory
                if os.path.isfile(lnk):
                    fname = lnk
                else:
                    fname = "{}/{}".format(self.resource_dir,lnk)

                opac_tmp = radmc3dPy.miescat.compute_opac_mie(fname=fname,
                                                             matdens=mtd,
                                                             agraincm=agr,
                                                             lamcm=lamcm,
                                                             extrapolate=True)
                
                self.opac.kabs.append( opac_tmp['kabs'] )
                self.opac.ksca.append( opac_tmp['kscat'] )
                self.opac.wav.append( opac_tmp['lamcm'] * 1.0e4 )
                self.opac.nwav.append( len(opac_tmp['lamcm']) )
                self.opac.freq.append( nc.cc / opac_tmp['lamcm'] )
                self.opac.nfreq.append( len(opac_tmp['lamcm']) )
                self.opac.scatmat.append( False )
                self.opac.phase_g.append( opac_tmp['gscat'] )
                self.opac.idust.append( i )
                
                self.opac.ext.append( "ag_{:06.2}".format( agr * 1.0e4 ) )
                
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
      paramfile - name of parameter file (usually problem_params.inp, default None).
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
        modpar.setPar(['agraincm', '1.0e-5', ' Grain size [cm]', 'Dust opacity'])
        modpar.setPar(['ngs', '1', ' Number of grain populations', 'Dust opacity'])

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
        
