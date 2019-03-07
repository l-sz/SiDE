SimpleDiskEnvFit

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

 * Envelope models: Ulrich+ (1976), Tafalla+ (2004) 
 * Disk model: parametric disk with hydrostatic-like vertical distribution
 * Compute dust opacity using Mie theory on the fly
 * Compute dust continuum emission maps and complex visibilities
 * Compute chi^2 when observed complex visibilities are provided
 
The package uses the following libraries:

 * UVplot
 * galario
 * radmc3dPy
