#!/usr/bin/env python

#from distutils.core import setup
from setuptools import setup

try:
   import galario
   print ('galario found in path')
except ImportError:
   print ('WARN: galario must be installed manually: https://github.com/mtazzari/galario')

setup(name='SimpleDiskEnvFit',
      version='0.1',
      install_requires=['radmc3dPy', 'corner', 'emcee', 'uvplot'],
      extras_require={'galario':['galario']},
      provides=['SimpleDiskEnvFit'],
      description='RADMC3D based Class 0/I/II protostar fitting tool.',
      author='Laszlo Szucs',
      author_email='laszlo.szucs@mpe.mpg.de',
      license='GPLv2',
      url='https://gitlab.mpcdf.mpg.de/szucs/SimpleDiskEnvFit',
      packages=['SimpleDiskEnvFit'],
      package_data={'SimpleDiskEnvFit':['lnk_files/*.lnk']},
     )
