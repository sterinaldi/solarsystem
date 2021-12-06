import numpy
from setuptools import setup, find_packages
from setuptools import Extension
from setuptools.command.build_ext import build_ext as _build_ext
from codecs import open
from os import path
from distutils.extension import Extension
from Cython.Build import cythonize
import os

# see https://stackoverflow.com/a/21621689/1862861 for why this is here
class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        self.include_dirs.append(numpy.get_include())

#ext_modules=[
#             Extension("solarsystem.constants",
#                       sources=[os.path.join("solarsystem","constants.py")],
#                       libraries=["m"], # Unix-like specific
#                       extra_compile_args=["-O3","-ffast-math"],
#                       include_dirs=['solarsystem', numpy.get_include()]
#                       ),
#             Extension("solarsystem.dynamics",
#                       sources=[os.path.join("solarsystem","dynamics.py")],
#                       libraries=["m"], # Unix-like specific
#                       extra_compile_args=["-O3","-ffast-math"],
#                       include_dirs=['solarsystem', numpy.get_include()]
#                       ),
#             Extension("solarsystem.kepler",
#                       sources=[os.path.join("solarsystem","kepler.py")],
#                       libraries=["m"], # Unix-like specific
#                       extra_compile_args=["-O3","-ffast-math"],
#                       include_dirs=['solarsystem', numpy.get_include()]
#                       )
#             ]

#
#setup(
#      name = 'solarsystem/dynamics',
#      ext_modules = ext_modules,
#      include_dirs=[numpy.get_include()]
#      )
#setup(
#      name = 'solarsystem/constants',
#      ext_modules = ext_modules,
#      include_dirs=[numpy.get_include()]
#      )
#setup(
#      name = 'solarsystem/kepler',
#      ext_modules = ext_modules,
#      include_dirs=[numpy.get_include()]
#      )

setup(
    name = 'solarsystem',
    use_scm_version=True,
    description = 'Solar system dynamics',
    author = 'Walter Del Pozzo, Stefano Rinaldi',
    author_email = 'walter.delpozzo@unipi.it, stefano.rinaldi@phd.unipi.it',
    url = 'https://github.com/sterinaldi/solarsystem',
    python_requires = '>=3.6',
    packages = ['solarsystem'],
    include_dirs = [numpy.get_include()],
    entry_points = {},
#    ext_modules=ext_modules,
    )

