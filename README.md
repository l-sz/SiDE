SimpleDiskEnv (SiDE)
====================

Self-contained `radmc3dModel` with child-mode RADMC3D runner and with 
image, visibility and $`\chi^2`$ storage.
The user provides a `radmc3dPar` object and complex visibility observational 
constrains. The code computes density structure, 
dust opacity and writes to disk. Then the runner computes dust 
temperature and continuum images. The images are transformed to the 
(u,v) space and the $`\chi^2`$ is computed compared to the input observations.

The model parameters are optimised using the `emcee` MCMC sampler tool. The final 
results include the posterior probability distribution of the fitted parameters 
and the walked chains in terms of $`\chi^2`$ and parameter combination.
Best fit model visibilities may be plotted with the observation data.

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

The current release of RADMC-3D (version 0.41) needs to be patched before running 
this code. Replace `main.f90` with `radmc3d_patch/main_patch.f90` in  
```bash
$RADMC3D_PATH/version_0.41/src
```
folder and recompile code. The patch is already merged to the development 
branch of RADMC-3D, but have not been released yet.

Installation:
------------

Download the repository in your browser or using the git clone utility.

Use Python's distutil utility and the provided setup.py script. 

```bash
$ python setup.py install --user
```

On Linux this installs the module to ~/.local/lib/python{2.7/3.6}/site-packages directory, 
which is usually included in the python search path.

Alternatively, you may directly add the repository location to your `PYTHONPATH`:

```bash
$ export PYTHONPATH=$PYTHONPATH:/path/to/your/SiDE/directory
```
You can make this addition permanent by saving the export command to the 
~/.bashrc or ~/.profile files.

After completing the installation step, the module should be available in python:

```python
import side
```

Basic usage:
-----------

Load default parameter configuration (disk and envelope included, no cavity) and 
modify the disk mass and radius and the envelope reference density.
```python
import side
import numpy as np

par = side.getParam()

par.setPar(['mdisk', '0.01*ms', ' Disk dust mass', 'Disk parameters'])
par.setPar(['rdisk', '100.*au', ' Disk radius', 'Disk parameters'])
par.setPar(['rho0Env', '1e-20', ' Envelope reference dust density [g/cm^3]', 'Envelope parameters'])
```

Create the `radmc3dModel` object and print model information (included components, 
component masses, densities, etc.). Then write the model to current folder (note 
that files need to be written to hard drive before the dust temperature and 
images are computed).
```
mod = side.radmc3dModel(modpar=par, main_dir='../opacities/')
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


The `examples/fit_elias29.py` script is tailored for fitting the complex visibility 
data of Elias29. The script prepares the complex visibility data (`visdata`) and 
sets the control parameters for the fitting (`kwargs`). The parameters are passed 
to the `side.run_mcmc()` function. At least the `main_dir`, `uvdata`, 
`paramfile`, `parname`, `p_ranges` and `p0` arguments need to be set. Note that 
the `kwargs` dictionary contains further important parameters (e.g. `dpc`, `incl`).
If `kwargs` is not set, then default values are used. Please consult the function 
documentation for the detailed description of the arguments.

The chain itself is controlled by setting the `nwalkers` and  `nsteps` arguments 
to the `run_mcmc()` function call. The number of threads used may be set by the 
`nthreads` arguments. In MPI mode, the `use_mpi` must be True. In this case the 
`nthreads` is ignored and as many cores will be used as the MPI provides to the 
code.

The `elias29_slurm.sh` provides an example for configuring and running the script 
on a cluster (CCAS at MPCDF) with SLURM scheduling system.

To run the script in MPI mode use one of the following commands:

```bash
# Without MPI, running locally
python fit_elias29.py

# Without scheduling system, using 8 threads
mpirun -n 8 python fit_elias29.py 

# With SLURM system
sbatch elias29_slurm.sh
```

Make sure that the call to `run_mcmc()` has `use_mpi = True` and that in the elias29_slurm.sh 
script the partition, ntasks-per-node and node parameters are set correctly.

The result chain is written to `examples/elias29/elias29_mcmc_save.p` python 
pickle format and `examples/elias29/chain.dat` ASCII files. The `side.tools` 
module provides functions for reading and visualizing the output (see Wiki).

### Resuming MCMC computation

The example script saves the parameter and posterior probability after each 
step to `examples/elias29/chain.dat` file. This file can be used to restart unfinished or 
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

With these parameters SiDE will read in the chain file and start 
the new MCMC run from the last saved state of the `emcee` sampler. The chain 
will continue with 1000 steps (i.e. 1000 times 40 models) and write the results 
from the current run to `examples/elias29/chain_0.dat`. The results in the 
`elias29_mcmc_save.p` pickle file will contain the chain from both the 
first and the rerun. It is possible to read and merge multiple ASCII chain files 
using the `side.tools.emcee_chain` class.

**General notes**

The minimally required files for the fitting are the parameter file (\*param.inp) 
and the observational constraints (\*.txt). If the grain size distribution is 
fitted, then a file containing the complex index of refraction need to be provided. 
SiDE is distributed with the astronomical silicate dust model of 
Draine & Lee (2003) and the porous silicate:carbon:ice:vacuum (1:2:3:6) dust 
model used in Natta & Testi (2004). 

Note that mass and density related parameters always refer to the dust component!


Acknowledgement:
---------------

The module relies on functiality provided by the radmc3dPy library (developed by 
Attila Juhász). The code for the Ulrich (1976) envelope model is adapted from 
the [HYPERION radiative transfer code](http://www.hyperion-rt.org/).
The RADMC3D images are transformed to complex visibilities using the GALARIO 
package and the visibilities are stored and plotted using UVplot package.
