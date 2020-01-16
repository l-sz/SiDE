#!/usr/bin/env python

#from distutils.core import setup
from setuptools import setup

try:
   import galario
   print ('galario found in path')
except ImportError:
   print ('WARN: galario must be installed manually: https://github.com/mtazzari/galario')

setup(name='side',
      version='0.1.2',
      install_requires=['radmc3dPy >= 0.30.2', 'corner >= 2.0.0', 'emcee >= 2.2.1',
			'uvplot >= 0.2.8', 'mpi4py >= 3.0.3'],
      extras_require={'galario':['galario']},
      dependency_links=['http://www.ita.uni-heidelberg.de/~dullemond/software/radmc-3d/radmc3dPyDoc/_downloads/radmc3dPy-0.30.2.tar.gz'],
      provides=['side'],
      description='RADMC3D based Class 0/I/II protostar fitting tool.',
      author='Laszlo Szucs',
      author_email='laszlo.szucs@mpe.mpg.de',
      license='GPLv2',
      url='https://github.com/l-sz/SiDE',
      packages=['side'],
      package_data={'side':['lnk_files/*.lnk']},
     )
