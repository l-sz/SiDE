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

import subprocess
import numpy as np
import radmc3dPy
import os

__all__ = ['radmc3dRunner']

class radmc3dRunner:
    '''
    The radmc3dRunner class provides a subprocess to compute dust temperature and 
    dust continuum emission maps in the current folder.
    '''
    radmc3dexec = 'radmc3d'   # use the system resolver
    folder = None
    proc = None
    cpu_nr = 0
    cpu_us = 0
    pid = 0
    ID = None
    
    def __init__(self, folder='.', bufsize=500000, nthreads=1, 
                 radmc3dexec=None, ID=None):
        '''
        Initializes RADMC3D child process and subprocess handlers.
        '''
        if ID is None:
            self.ID = np.random.randint(0,99999)
        else:
            self.ID = ID
        
        if radmc3dexec:
            self.radmc3dexec = radmc3dexec
        
        self.folder = folder
        
        self.proc = subprocess.Popen([self.radmc3dexec, 'child', 'setthreads',
                                      str(nthreads)], shell=False,
                                      stdin=subprocess.PIPE, 
                                      stdout=subprocess.PIPE, 
                                      bufsize=bufsize)
        
        # Store CPU uses
        dum = self.proc.stdout.readline()
        self.cpu_nr = ''.join(filter(lambda x: x.isdigit(), str(dum)))
        dum = self.proc.stdout.readline()
        self.cpu_us = ''.join(filter(lambda x: x.isdigit(), str(dum)))
        
        # Store PID
        self.pid = self.proc.pid
        
        return None

    def terminate(self, verbose=False):
        '''
        Terminate child process and close pipes.
        '''

        self.proc.stdin.write(b"exit\n")
        self.proc.stdin.flush()
        
        self.proc.terminate()
        self.proc.wait()
        
        if self.proc.poll() < 0:
            done = True
        else:
            done = False
            
        if (done and verbose):
            print('INFO [{:06}]: process PID {} terminated'.format(self.ID,
                                                                   self.pid))
            
        return 0
    
    def runMCtherm(self, noscat=None, nphot_therm=None, 
                    nphot_scat=None, nphot_mcmono=None, 
                    verbose=False):
        '''
        Send instruction to RADCM3D child process to compute dust temperature.
        
        nthreads number of CPUs are used in the thermal MC computation. The 
        computed dust temperatures are stored in memory (of child process) and 
        are written to disk (dust_temperature.dat).
        
        Note that it is recommended to set the photon package number in the 
        radmc3d.inp file and not at runtime.
        
        Parameters
        ----------
          noscat       - set of scattering process (opacity)
          nphot_therm  - set number of thermal photon packages
          nphot_scat   - set number of photon packages used for scattering
          nphot_mcmono - set number of photon packages in monochromatic MC 
                         simulation.
        '''
        current_dir = os.path.realpath('.')
        if verbose:
            print('INFO [{:06}]: process PID {} started at: \n     {}'.format(self.ID, self.pid,current_dir))
        
        self.proc.stdin.write(b"mctherm\n")
        
        if noscat:
            self.proc.stdin.write(b"noscat\n")
        if nphot_therm:
            self.proc.stdin.write(b"nphot_therm\n")
            self.proc.stdin.write(str.encode("{}\n".format(str(int(nphot_therm)))))
        if nphot_scat:
            self.proc.stdin.write(b"nphot_scat\n")
            self.proc.stdin.write(str.encode("{}\n".format(str(int(nphot_scat)))))
        if nphot_mcmono:
            self.proc.stdin.write(b"nphot_mcmono\n")
            self.proc.stdin.write(str.encode("{}\n".format(str(int(nphot_mcmono)))))
                                  
        self.proc.stdin.write(b"respondwhenready\n")
                                  
        self.proc.stdin.write(b"enter\n")
        self.proc.stdin.flush()
        
        # should wait until it runs
        stat = self.proc.stdout.readline()
        if verbose:
            print ('INFO [{:06}]: Thermal MC run done!'.format(self.ID))
        
        return 0
        
    def runImage(self, npix=None, incl=None, wav=None, sizeau=None, phi=None,      
                posang=None, pointau=None, fluxcons=True, nostar=False,
                noscat=False, lambdarange=None, nlam=None, stokes=False, **arg):
        '''
        Send instructions to RADMC3D child process to compute image.
        
        Based on the radmc3dPy.image.makeImage() function and the parameters 
        have the same meaning.
        '''

        self.proc.stdin.write(b"image\n")
        
        if incl:
            self.proc.stdin.write(b"incl\n")
            self.proc.stdin.write(str.encode("{}\n".format(str(incl))))
        if npix:
            self.proc.stdin.write(b"npix\n")
            self.proc.stdin.write(str.encode("{}\n".format(str(int(npix)))))
        if lambdarange and nlam and len(lambdarange) == 2:
            self.proc.stdin.write(b"lambdarange\n")
            self.proc.stdin.write(str.encode("{}\n".format(str(lambdarange[0]))))
            self.proc.stdin.write(str.encode("{}\n".format(str(lambdarange[1]))))
            self.proc.stdin.write(b"nlam\n")
            self.proc.stdin.write(str.encode("{}\n".format(str(int(nlam)))))
        elif wav:
            self.proc.stdin.write(b"lambda\n")
            self.proc.stdin.write(str.encode("{}\n".format(str(wav))))
        else:
            print("ERROR [{:06}]: no wavelength parameter \
                  set in run_image()!".format(self.ID))
            return -1
        if sizeau:
            self.proc.stdin.write(b"sizeau\n")
            self.proc.stdin.write(str.encode("{}\n".format(str(sizeau))))
        if phi:
            self.proc.stdin.write(b"phi\n")
            self.proc.stdin.write(str.encode("{}\n".format(str(phi))))
        if posang:
            self.proc.stdin.write(b"posang\n")
            self.proc.stdin.write(str.encode("{}\n".format(str(posang))))
        if pointau and len(pointau) == 3:
            self.proc.stdin.write(b"pointau\n")
            self.proc.stdin.write(str.encode("{}\n".format(str(pointau[0]))))
            self.proc.stdin.write(str.encode("{}\n".format(str(pointau[1]))))
            self.proc.stdin.write(str.encode("{}\n".format(str(pointau[2]))))
        if fluxcons:
            self.proc.stdin.write(b"fluxcons\n")
        else:
            self.proc.stdin.write(b"nofluxcons\n")
        if nostar:
            self.proc.stdin.write(b"nostar\n")
        if noscat:
            self.proc.stdin.write(b"noscat\n")
        if stokes:
            self.proc.stdin.write(b"stokes\n")        
        
        self.proc.stdin.write(b"enter\n")
        self.proc.stdin.flush()
        
        self.proc.stdin.write(b"writeimage\n")
        self.proc.stdin.flush()
        
        # should wait until it runs
        
        return 0
        
    def readImage(self):
        '''
        Reads image data from pipe and returns a radmc3dPy image object.
        
        Based on the radmc3dImage.image.radmc3dImage.readImage() method.
        '''
        pc = 3.08572e18         # parsec [cm]
        cc = 2.99792458e10      # speed of light [cm/s]

        img = radmc3dPy.image.radmc3dImage()
        
        rfile = self.proc.stdout

        iformat = int(rfile.readline())

        # Nr of pixels
        dum = rfile.readline()
        dum = dum.split()
        img.nx  = int(dum[0])
        img.ny  = int(dum[1])
        
        # Nr of frequencies
        img.nfreq = int(rfile.readline())
        img.nwav  = img.nfreq
        
        # Pixel sizes
        dum = rfile.readline()
        dum = dum.split()
        img.sizepix_x = float(dum[0])
        img.sizepix_y = float(dum[1])
        
        # Wavelength of the image
        img.wav = []
        for iwav in range(img.nwav):
            img.wav.append(float(rfile.readline()))
        img.wav = np.array(img.wav)
        img.freq = cc / img.wav * 1e4
                
        # If we have a normal total intensity image
        if iformat==1:
            img.stokes = False
                        
            img.image = np.zeros([img.nx,img.ny,img.nwav], dtype=np.float64)
            for iwav in range(img.nwav):
                # Blank line
                dum = rfile.readline()
                for iy in range(img.nx):
                    for ix in range(img.ny):
                        img.image[ix,iy,iwav] = float(rfile.readline())

        # If we have the full stokes image
        elif iformat==3:
            img.stokes = True
            img.image = np.zeros([img.nx,img.ny,4,img.nwav], dtype=np.float64)
            for iwav in range(nwav):
                # Blank line
                dum = rfile.readline()
                for iy in range(img.nx):
                    for ix in range(img.ny):
                        dum = rfile.readline().split()
                        imstokes = [float(i) for i in dum]
                        img.image[ix,iy,0,iwav] = float(dum[0])
                        img.image[ix,iy,1,iwav] = float(dum[1])
                        img.image[ix,iy,2,iwav] = float(dum[2])
                        img.image[ix,iy,3,iwav] = float(dum[3])

        dum = rfile.readline()
        
        # Conversion from erg/s/cm/cm/Hz/ster to Jy/pixel
        img.dpc = 1.0
        conv  = img.sizepix_x * img.sizepix_y / (img.dpc * pc)**2. * 1e23
        img.imageJyppix = img.image * conv

        img.x = ((np.arange(img.nx, dtype=np.float64) + 0.5) - img.nx/2) * img.sizepix_x
        img.y = ((np.arange(img.ny, dtype=np.float64) + 0.5) - img.ny/2) * img.sizepix_y
        
        return img
        
        
    def getImage(self, **args):
        '''
        Compute, read and return RADMC3D image.
        
        This method instructs subprocess to compute the image (using runImage) 
        and reads the resulting image from buffer. The return value is a 
        radmc3dImage object.
        
        getImage() takes the same arguments as runImage() and radmc3dPy.image.makeImage().
        '''
        # Compute image
        self.runImage(**args)
        
        # Read image
        img = self.readImage()
        
        return img
