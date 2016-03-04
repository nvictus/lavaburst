# -*- encoding: utf-8 -*-
from setuptools import setup, find_packages, Extension
import glob
import ast
import os
import io
import re

try:
    from Cython.Distutils import build_ext as _build_ext
    from Cython.Build import cythonize
    HAVE_CYTHON = True
except ImportError:
    from setuptools.command.build_ext import build_ext as _build_ext
    HAVE_CYTHON = False


classifiers = """
    Development Status :: 4 - Beta
    Operating System :: OS Independent
    Programming Language :: Python
    Programming Language :: Python :: 2
    Programming Language :: Python :: 2.7
    Programming Language :: Python :: 3
    Programming Language :: Python :: 3.3
    Programming Language :: Python :: 3.4
    Programming Language :: Python :: 3.5
"""


def _read(*parts, **kwargs):
    filepath = os.path.join(os.path.dirname(__file__), *parts)
    encoding = kwargs.pop('encoding', 'utf-8')
    with io.open(filepath, encoding=encoding) as fh:
        text = fh.read()
    return text


def get_version():
    version = re.search(
        r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
        _read('lavaburst', '__init__.py'),
        re.MULTILINE).group(1)
    return version


def get_long_description():
    return _read('README.rst')


def get_ext_modules():
    ext = '.pyx' if HAVE_CYTHON else '.c'
    src_files = glob.glob(os.path.join("lavaburst", "core", "*" + ext))

    ext_modules = []
    for src_file in src_files:
        name = "lavaburst.core." + os.path.splitext(os.path.basename(src_file))[0]
        ext_modules.append(Extension(name, [src_file]))

    if HAVE_CYTHON:
        # .pyx to .c
        ext_modules = cythonize(ext_modules)  #, annotate=True

    return ext_modules


class build_ext(_build_ext):
    # Extension module build configuration
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Fix to work with bootstrapped numpy installation
        # http://stackoverflow.com/a/21621689/579416
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())


install_requires = ['six', 'numpy', 'scipy', 'cython']
tests_require = ['nose']
extras_require = {
    'docs': ['Sphinx>=1.1', 'numpydoc>=0.5'],
}


setup(
    name='lavaburst',
    version=get_version(),
    license='MIT',
    author='Nezar Abdennur',
    author_email='nezar@mit.edu',
    url='http://nezar-compbio.github.io/lavaburst/',
    description='Probabilistic segmentation modeling for Hi-C data',
    keywords=['bioinformatics', 'genomics', 'Hi-C', 'topological domains'],
    long_description=get_long_description(),
    classifiers=[s.strip() for s in classifiers.split('\n') if s],
    packages=find_packages(),
    ext_modules = get_ext_modules(),
    cmdclass = {'build_ext': build_ext},
    install_requires=install_requires,
    tests_require=tests_require,
    extras_require=extras_require,
    zip_safe=False,
    # include_package_data=True,
    # package_data={
    #     'sample': ['package_data.dat'],
    # },
    # entry_points={
    #     'console_scripts': [
    #         'sample=sample:main',
    #     ],
    # },
)
