"""Setup file."""

from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(name="Shared objects",
      ext_modules=cythonize("*.pyx"),
      include_dirs=[np.get_include()]
      )
