from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy

extensions = [
    Extension(
        "clustering",
        sources=["clustering.pyx"],
        language="c++",
        extra_compile_args=["-std=c++11"],
    ),
    Extension(
        "feature_extractor",
        sources=["feature_extractor.pyx"],
        language="c++",
        extra_compile_args=["-std=c++11"],
    ),
]

setup(
    name='clustering',
    ext_modules=cythonize(extensions),
    install_requires=['numpy', 'hnswlib'],
    include_dirs=[numpy.get_include()]
)