# -*- encoding: utf-8 -*-
from setuptools import setup, find_packages, Extension
from os import path
import ast
import io
import re


try:
    from Cython.Distutils import build_ext as _build_ext
    from Cython.Build import cythonize
    use_cython = True
except ImportError:
    from setuptools import build_ext as _build_ext
    use_cython = False


def read(*names, **kwargs):
    return io.open(
        path.join(path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8")
    ).read()


def get_version():
    # Version parsing from __init__ pulled from Flask's setup.py
    # https://github.com/mitsuhiko/flask/blob/master/setup.py
    _version_re = re.compile(r'__version__\s+=\s+(.*)')
    hit = _version_re.search(read('lavaburst', '__init__.py')).group(1)
    return str(ast.literal_eval(hit))


def get_long_description():
    return read('README.rst')


classifiers = """
    Development Status :: 1 - Planning
    License :: OSI Approved :: MIT License
    Topic :: Scientific/Engineering
    Topic :: Scientific/Engineering :: Bio-Informatics
    Programming Language :: Python
    Programming Language :: Python :: 2
    Programming Language :: Python :: 2.7
    Operating System :: Unix
    Operating System :: POSIX
    Operating System :: MacOS :: MacOS X
"""


# Extension module build configuration
class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Fix to work with bootstrapped numpy installation
        # http://stackoverflow.com/a/21621689/579416
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())


def get_ext_modules():
    ext = '.pyx' if use_cython else '.c'
    ext_modules = [
        Extension("*", [path.join("lavaburst", "core", "*"+ext)])
    ]
    if use_cython:
        ext_modules = cythonize(ext_modules) #, annotate=True
    return ext_modules


setup(
    name='lavaburst',
    version=get_version(),
    license='MIT',
    description='Probabilistic segmentation modeling for Hi-C data',
    long_description=get_long_description(),
    classifiers=[s.strip() for s in classifiers.split('\n') if s],
    keywords=['bioinformatics', 'genomics', 'Hi-C', 'topological domains'],
    author='Nezar Abdennur',
    author_email='nezar@mit.edu',
    url='http://nezar-compbio.github.io/lavaburst/',
    packages=find_packages(),
    ext_modules = get_ext_modules(),
    cmdclass = {'build_ext': build_ext},
    setup_requires=['numpy'],
    install_requires=['numpy'],
    zip_safe=False,
    include_package_data=True,
    # package_data={
    #     'sample': ['package_data.dat'],
    # },
    # entry_points={
    #     'console_scripts': [
    #         'sample=sample:main',
    #     ],
    # },
)
