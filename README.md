SimpleDiskEnvFit
================

Self-contained radmc3dModel with child-mode RADMC3D runner and with 
image, visibility and chi^2 storage.
The user provides a radmc3dPar() object and optional observed     
constrains in the uv space. The code computes density structure, 
dust opacity and writes to disk. Then the runner computes dust 
temperature and continuum images. The images are transformed to the 
uv space and the chi2 is computed compared to the input observations.

Based on SimpleDiskEnv. The code is intended to be used with the 
emcee MCMC sampler tool.

Features and options:
--------------------

- Envelope models: Ulrich+ (1976), Tafalla+ (2004) 
- Disk model: parametric disk with hydrostatic-like vertical distribution
- Compute dust opacity using Mie theory on the fly
- Compute dust continuum emission maps and complex visibilities
- Compute chi^2 when observed complex visibilities are provided

 
Requirements:
------------

- python (2.7 or 3.6+)
- matplotlib
- numpy
- [radmc3dPy](https://www.ast.cam.ac.uk/~juhasz/radmc3dPyDoc)
- [galario](https://github.com/mtazzari/galario) 
- [uvplot](https://github.com/mtazzari/uvplot)
- emcee (optional)
- jupyter (optional)

- [RADMC3D](http://www.ita.uni-heidelberg.de/~dullemond/software/radmc-3d)


Installation:
------------

Download the repository in your browser or using the git clone utility.

Add the repository location to your PYTHONPATH:

```
$ export PYTHONPATH=$PYTHONPATH:/path/to/your/SimpleDiskEnvFit/directory
```
You can make this addition permanent by saving the export command to the 
~/.bashrc or ~/.profile files.

After completing this step, the module should be available in python:

```
>>> import SimpleDiskEnvFit
```

Basic usage:
-----------

```
>>> par = SimpleDiskEnvFit.getParam()

>>> par.setPar(['mdisk', '0.01*ms', ' Disk mass', 'Disk parameters'])
>>> par.setPar(['rdisk', '100.*au', ' Disk radius', 'Disk parameters'])
>>> par.setPar(['rho0Env', '1e-20', ' Envelope reference density [g/cm^3]', 'Envelope parameters'])

>>> mod = SimpleDiskEnvFit.radmc3dModel(modpar=par, main_dir='../opacities/')
>>> mod.write2folder()

>>> impar = [{'npix':512,'wav':1100.,'sizeau':6000,'incl':60},
             {'npix':512,'wav':3000.,'sizeau':6000,'incl':60}]
             
>>> mod.runModel(mctherm=True,impar=impar)

>>> u1, v1, Re1, Im1, w1 = np.loadtxt('Elias29uvt_270.txt', unpack=True)
>>> u2, v2, Re2, Im2, w2 = np.loadtxt('Elias29uvt_94.txt', unpack=True)

>>> vis = [{'u':u1, 'v':v1, 'Re':Re1, 'Im':Im1, 'w':w1, 'wav':1100.},
           {'u':u2, 'v':v2, 'Re':Re2, 'Im':Im2, 'w':w2, 'wav':3000.}]
        
>>> mod.getVis(uvdata=vis, dpc=125.)

>>> uvbin = 10000
>>> ax0 = mod.vis_inp[0].plot(uvbin_size=uvbin)
>>> mod.vis_mod[0].plot(uvbin_size=uvbin, axis=ax0, linestyle='r-')

>>> ax1 = mod.vis_inp[1].plot(uvbin_size=uvbin)
>>> mod.vis_mod[1].plot(uvbin_size=uvbin, axis=ax1, linestyle='r-')
```

Acknowledgement:
---------------

The module relies on functiality provided by the radmc3dPy library (developed by 
Attila Juhász). The code for the Ulrich (1976) envelope model is adapted from 
the [HYPERION radiative transfer code](http://www.hyperion-rt.org/).
The RADMC3D images are transformed to complex visibilities using the GALARIO 
package and the visibilities are stored and plotted using UVplot package.
