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

import subprocess
import numpy as np
import radmc3dPy
import os
from timeit import default_timer as timer

__all__ = ['radmc3dRunner']

class radmc3dRunner:
    '''
    The radmc3dRunner class provides a subprocess to compute dust temperature and 
    dust continuum emission maps in the current folder.
    '''
    radmc3dexec = 'radmc3d'   # use the system resolver
    model_dir = None
    proc = None
    cpu_nr = 0
    cpu_us = 0
    pid = 0
    ID = None
    verbose = None
    
    def __init__(self, model_dir='.', bufsize=500000, nthreads=1, 
                 radmc3dexec=None, ID=None, verbose=False):
        '''
        Initializes RADMC3D child process and subprocess handlers.
        '''
        if ID is None:
            self.ID = np.random.randint(0,99999)
        else:
            self.ID = ID
        self.verbose = verbose
        
        if radmc3dexec:
            self.radmc3dexec = radmc3dexec
        
        self.model_dir = model_dir
        
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

    def terminate(self, verbose=None):
        '''
        Terminate child process and close pipes.
        '''
        # If verbosity not set then use global class variable
        if verbose is None:
            verbose = self.verbose

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
                    verbose=None, time=False):
        '''
        Send instruction to RADCM3D child process to compute dust temperature.
        
        nthreads number of CPUs are used in the thermal MC computation. The 
        computed dust temperatures are stored in memory (of child process) and 
        are written to disk (dust_temperature.dat).
        
        Note that it is recommended to set the photon package number in the 
        radmc3d.inp file and not at runtime.
        
        Parameters
        ----------
        noscat : int, optional
                Switch off scattering process even if dust opacity file contains 
                scattering cross sections. If None, then option from radmc3d.inp 
                is used.
        nphot_therm : int, optional
                Set number of thermal photon packages. If None, then option from 
                radmc3d.inp is used.
                Note: as of RADMC-3D version 0.41 setting nphot_therm in child 
                mode is not supported. If code crashes, set this to None and 
                adjust nphot_therm in the radmc3dPy parameter file.
        nphot_scat : int, optional
                Set number of photon packages used for scattering. If None, then 
                option from radmc3d.inp is used.
        nphot_mcmono : int, optional
                Set number of photon packages in monochromatic MC simulation. 
                If None, then option from radmc3d.inp is used.
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

        if time:
            end = timer()
            dt = end-start

        if verbose and time:
            print ('INFO [{:06}]: Thermal MC run done in {:.2f} s!'.format(self.ID, dt))
        elif verbose:
            print ('INFO [{:06}]: Thermal MC run done!'.format(self.ID))

        return 0

    def runImage(self, npix=None, incl=None, phi=None, posang=None, wav=None, 
                 lambdarange=None, nlam=None, sizeau=None,  pointau=None, 
                 fluxcons=True, nostar=False, noscat=False, stokes=False, 
                 sloppy=False, **arg):
        '''
        Send instructions to RADMC3D child process to compute image.
        
        Based on the radmc3dPy.image.makeImage() function and the parameters 
        have the same meaning.
        
        Note that when used together with getVis() function, then a non-zero 
        position angle should be set either in runImage() or in getVis(), the 
        preferably set it in getVis().
        
        Parameters
        ----------
        npix    : int, optional
                  Pixel number (same along x and y).
        incl    : float, optional
                  Inclination of the object on the sky of the observer. incl=0
                  means a view from the north pole downward, incl=90 means an 
                  edge-on view.
        phi     : float, optional
                  The rotation of the object along its z-axis. Positive phi means
                  that the object rotates counter-clockwise. This is useful for 
                  asymmetric (3D) models.
        posang  : float, optional
                  The position angle of the camera in degrees. The camera is 
                  rotated around the (0,0) point by this angle. If the image is 
                  post-processed with galario (e.g. to compute chi^2), then the 
                  projection should be done only once: set posang = 0.0 in 
                  runImage() and set PA to the desired value in getVis() function.
        wav     : float
                  Frequency where image is computed. Minimally wav must be set.
        lambdarange : ndarray or None
                  If set then compute multiple images at wavelengths given by 
                  lambdarange.
        nlam    : int or None
                  Number of wavelength points within lambdarange that will be 
                  computed at once.
        sizeau  : float or None(optional)
                  Image size in physical scale, in unit of AU
        pointau : array-like float, optional
                  Defines the position and direction of camera in Cartesian space.
                  The coordinates are given in au unit.
        fluxcons: bool (default True)
                  If True, then ensure flux conversion (see RADMC3D manual).
        nostar  : bool (default False)
                  If True, then do not include stellar radiation in image.
        noscat  : bool (default False)
                  If True, then omit scattering, even if scattering data is given 
                  in opacity input file.
        stokes  : bool (default False)
                  If polarization information is given in the dust opacity file, 
                  then setting this True will include the Stokes parameters to
                  the computed images. 
        sloppy  : bool (default False)
                  If True then subpixel refinement criterion are relaxed when 
                  raytracing. This is equivivalent with the sloppy command line 
                  parameter in RADMC-3D and sets camera_min_dangle, camera_min_drr 
                  and camera_spher_cavity_relres to 0.1. This may speed up image 
                  raytracing if the image resolution is relatively small. If 
                  the resolution is high (e.g. ALMA resolution), then subpixel 
                  refinement is not necessary and the option will not lead to 
                  speed up.
                
        **arg : further arguments are not used
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
        elif type(wav) not in [list, np.ndarray]:
            self.proc.stdin.write(b"lambda\n")
            self.proc.stdin.write(str.encode("{}\n".format(str(wav))))
        elif type(wav) in [list, np.ndarray]:
            nwav = len(wav)
            f = open('camera_wavelength_micron.inp','w')
            f.write("{:6}\n".format(nwav))
            for i in range(nwav):
                f.write("{:12.9E}\n".format(wav[i]))
            f.close()
            self.proc.stdin.write(b"loadlambda\n")
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
        if sloppy:
            self.proc.stdin.write(b"sloppy\n")
        
        self.proc.stdin.write(b"enter\n")
        self.proc.stdin.flush()
        
        self.proc.stdin.write(b"writeimage\n")
        self.proc.stdin.flush()

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
        
        
    def getImage(self, verbose=None, time=False, **args):
        '''
        Compute, read and return RADMC3D image.
        
        This method instructs subprocess to compute the image (using runImage) 
        and reads the resulting image from buffer. The return value is a 
        radmc3dImage object.
        
        getImage() takes the same arguments as runImage() and radmc3dPy.image.makeImage().
        
        Parameters
        ----------
        verbose : bool, optional
                Print INFO messages to standard output. Default is False.
        time :  bool, optional
                Prints function runtime information. Useful for profiling.
                Default is False.
        **arg : dict
                arguments passed to runImage() class method.
        '''
        # If verbosity not set then use global class variable
        if verbose is None:
            verbose = self.verbose
        
        if time:
            start = timer()
        
        # Compute image
        self.runImage(**args)
        
        # Read image
        img = self.readImage()

        if time:
            end = timer()
            dt = end-start

        if verbose and time:
            print ('INFO [{:06}]: Image at {} micron computed in {:.2f} s!'.format(
                            self.ID, args['wav'], dt))
        elif verbose:
            print ('INFO [{:06}]: Image at {} micron computed!'.format(self.ID,
                                                                  args['wav']))

        return img
