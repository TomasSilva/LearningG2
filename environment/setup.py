# setup.py
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="geometry.wedge_product",                   # <-- Explicit module name!
        sources=["geometry/wedge_product.pyx"],          # <-- File path
        include_dirs=[np.get_include()],
    )
]

setup(
    ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}),
)