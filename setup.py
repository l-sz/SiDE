#!/usr/bin/env python

from distutils.core import setup

setup(name='SimpleDiskEnvFit',
      version='0.1',
      requires=['radmc3dPy', 'corner', 'galario', 'emcee'],
      description='RADMC3D based Class 0/I/II protostar fitting tool.',
      author='Laszlo Szucs',
      author_email='laszlo.szucs@mpe.mpg.de',
      url='https://gitlab.mpcdf.mpg.de/szucs/SimpleDiskEnvFit',
      packages=['SimpleDiskEnvFit'],
      package_data={'SimpleDiskEnvFit':['lnk_files/*.lnk']},
     )
