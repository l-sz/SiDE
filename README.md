SimpleDiskEnvFit
================

Self-contained `radmc3dModel` with child-mode RADMC3D runner and with 
image, visibility and $`\chi^2`$ storage.
The user provides a `radmc3dPar` object and optional observed     
constrains in the uv space. The code computes density structure, 
dust opacity and writes to disk. Then the runner computes dust 
temperature and continuum images. The images are transformed to the 
uv space and the $`\chi^2`$ is computed compared to the input observations.

Based on [SimpleDiskEnv](https://gitlab.mpcdf.mpg.de/szucs/SimpleDiskEnv). 
The code is intended to be used with the `emcee` MCMC sampler tool.

Features and options:
--------------------

- Envelope models: Ulrich+ (1976), Tafalla+ (2004) 
- Disk model: parametric disk with hydrostatic-like vertical distribution
- Compute dust opacity using Mie theory on the fly
- Compute dust continuum emission maps and complex visibilities
- Compute $`\chi^2`$ when observed complex visibilities are provided

 
Requirements:
------------

- python (2.7 or 3.5+)
- matplotlib
- numpy
- [radmc3dPy](https://www.ast.cam.ac.uk/~juhasz/radmc3dPyDoc)
- [galario](https://github.com/mtazzari/galario) 
- [uvplot](https://github.com/mtazzari/uvplot)
- [emcee](http://dfm.io/emcee) (used for MCMC fitting)
- [corner](https://corner.readthedocs.io)
- mpi4py (optional, used for emcee parallelism)

- [RADMC3D](http://www.ita.uni-heidelberg.de/~dullemond/software/radmc-3d)

**Importnat:** 

The current release of RADMC-3D (version 0.41) needs to be patched before model 
fitting. Replace `main.f90` with `radmc3d_path_main.f90` in  
```bash
$RADMC3D_PATH/version_0.41/src
```
folder and recompile code. The patch is already merged to the developement 
branch of RADMC-3D, but have not been released yet.

Installation:
------------

Download the repository in your browser or using the git clone utility.

Use Python's distutil utility and the provided setup.py script. 

```bash
$ python setup.py install --user
```

On linux this 
installes the module to ~/.local/lib/python{2.7/3.6}/site-packages directory, 
which is usually included in the python search path.

Alternatively, you may directly add the repository location to your PYTHONPATH:

```bash
$ export PYTHONPATH=$PYTHONPATH:/path/to/your/SimpleDiskEnvFit/directory
```
You can make this addition permanent by saving the export command to the 
~/.bashrc or ~/.profile files.

After completing the installation step, the module should be available in python:

```python
import SimpleDiskEnvFit
```

Basic usage:
-----------

Load default parameter configuration (disk and envelope included, no cavity) and 
modify the disk mass and radius and the envelope reference density.
```python
import SimpleDiskEnvFit
import numpy as np

par = SimpleDiskEnvFit.getParam()

par.setPar(['mdisk', '0.01*ms', ' Disk mass', 'Disk parameters'])
par.setPar(['rdisk', '100.*au', ' Disk radius', 'Disk parameters'])
par.setPar(['rho0Env', '1e-20', ' Envelope reference density [g/cm^3]', 'Envelope parameters'])
```

Create the `radmc3dModel` object and print model information (included components, 
component masses, densities, etc.). Then write the model to current folder (note 
that files need to be written to hard drive before the dust temperature and 
images are computed).
```
mod = SimpleDiskEnvFit.radmc3dModel(modpar=par, main_dir='../opacities/')
mod.infoModelParams()
mod.write2folder()
```

Read observed visibility data (obtained at 1.1 and 3 mm wavelength) and set 
image parameters.
```python
u1, v1, Re1, Im1, w1 = np.loadtxt('Elias29uvt_270.txt', unpack=True)
u2, v2, Re2, Im2, w2 = np.loadtxt('Elias29uvt_94.txt', unpack=True)

vis = [{'u':u1, 'v':v1, 'Re':Re1, 'Im':Im1, 'w':w1, 'wav':1100.},
       {'u':u2, 'v':v2, 'Re':Re2, 'Im':Im2, 'w':w2, 'wav':3000.}]
           
impar = [{'npix':512,'wav':1100.,'sizeau':6000,'incl':60},
         {'npix':512,'wav':3000.,'sizeau':6000,'incl':60}]
```

Compute the dust temperature and the dust continuum emission. Then decompose the 
images to complex visibility space. The computed images are stored in the mod.image 
class variable (list type). The individual images are `radmc3dImage` objects. It is 
possible to use the standard `radmc3dPy` methods on the images (e.g. to write to fits 
file format).
```python
mod.runModel(mctherm=True,impar=impar)
mod.getVis(uvdata=vis, dpc=125.)
```

Finally, compare the observed and modelled visibilities.
```python
uvbin = 10000
ax0 = mod.vis_inp[0].plot(uvbin_size=uvbin, label='1.1 mm')
mod.vis_mod[0].plot(uvbin_size=uvbin, axis=ax0, linestyle='r-')

ax1 = mod.vis_inp[1].plot(uvbin_size=uvbin), label='3 mm'
mod.vis_mod[1].plot(uvbin_size=uvbin, axis=ax1, linestyle='r-')
```

Example: Fitting Elias 29
-------------------------

The examples folder contains the parameter file, fitting scripts, opacity data 
and observed complex visibilities for modelling the Class I protostar Elias 29.

Contents of the `examples/elias29` folder:

    elias29_params.inp              Default model parameters 
    porous_natta2004_rhod_1.36.lnk  Complex refractive index data of dust grains
    Elias29uvt_270.txt              Observed complex visibility at 1.1 mm wavelength 
    Elias29uvt_94.txt               Observed complex visibility at 3.0 mm wavelength


The `fit_elias29.py` file contains the `run_mcmc()` function, which is tailord 
for fitting the Elias 29 data and the `plot_corner()` function for plotting the 
result posterior probability density distribution. The routines may be imported 
in interactive Python shell or run as a script. In the later case, the fitting 
parameters (nwalkers, nthreads, nsteps, use_mpi, etc.) should be set in the .

The `elias29_slurm.sh` provides an example for configuring and running the script 
on a cluster (CCAS at MPCDF) with SLURM scheduling system.

To run the script in MPI mode use one of the following commands:

```bash
# Without scheduling system, using 8 threads
mpirun -n 8 python fit_elias29.py 

# With SLURM system
sbatch elias29_slurm.sh
```

Make sure that the call to `run_mcmc()` has use_mpi=True and that in the elias29_slurm.sh 
script the partition, ntasks-per-node, nodes parameters are set correctly and 
that the srun -n argument reflects the choice of the above parameters.

The result chain is written to `examples/elias29/elias29_mcmc_save.p` python 
pickle format fime. A plotting routine (`plot_corner`) is provided to visualise 
the results in a corner plot.

### Resuming MCMC computation

The example script saves the parameter and posterior probability after each 
step to `examples/elias29/chain_0.dat` file. This file can be used to restart unfinished or 
interupted computations. 
*Important*: the same number of walkers must be used in subsequent runs.

To resume a previous run, edit the `fit_elias29.py` script and update the call to 
`run_mcmc()` in the last lines:

```python
# Resume example
current_dir = os.path.realpath('.')
run_mcmc(current_dir+'/elias29', nwalkers=40, nsteps=1000, nburnin=0, use_mpi=True, 
         resume=True, restart_file=current_dir+'/elias29/chain.dat')
```

With these parameters `SimpleDiskEnvFit` will read in the chain file and start 
the new MCMC run from the last saved state of the `emcee` sampler. The chain 
will continue with 1000 steps (i.e. 1000 times 40 models) and write the results 
from the current run to `examples/elias29/chain_0.dat`. The results in the 
`elias29_mcmc_save.p` pickle file will contain the chain from both the 
first and the rerun.

**General notes**

The minimally required files for the fitting are the parameter file (\*param.inp) 
and the observational constraints (\*.txt). If the grain size distribution is 
fitted, then a file containing the complex index of refraction need to be provided. 
SimpleDiskEnvFit is distributed with the astronomical silicate dust model of 
Draine & Lee (2003) and the porous silicate:carbon:ice:vacuum (1:2:3:6) dust 
model used in Natta & Testi (2004). 



Acknowledgement:
---------------

The module relies on functiality provided by the radmc3dPy library (developed by 
Attila Juhász). The code for the Ulrich (1976) envelope model is adapted from 
the [HYPERION radiative transfer code](http://www.hyperion-rt.org/).
The RADMC3D images are transformed to complex visibilities using the GALARIO 
package and the visibilities are stored and plotted using UVplot package.
