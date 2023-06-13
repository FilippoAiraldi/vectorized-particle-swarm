from setuptools import setup, Extension
from Cython.Build import cythonize


extensions = [
    Extension("vpso.math2", ["src/vpso/math2.pyx"]),
    # Extension("vpso.module2", ["src/vpso/module2.pyx"]),
]

ext_modules = cythonize(extensions, language_level="3")

setup(
    ext_modules=ext_modules,
)
