from setuptools import setup, Extension

try:
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext
    import numpy
except ImportError:
    use_cython = False
else:
    use_cython = True

cmdclass = {}
ext_modules = []

if use_cython:
    ext_modules += [
        Extension("*", ["lavaburst/core/*.pyx"], include_dirs=[numpy.get_include()])
    ]
    cmdclass.update({'build_ext': build_ext})
    ext_modules = cythonize(ext_modules) #, annotate=True

setup(
    name='lavaburst',
    version='0.1.0',
    author='Nezar Abdennur',
    author_email='nezar@mit.edu',
    license='MIT',
    description='Probabilistic segmentation modeling for Hi-C data',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules,
    packages=['lavaburst'],
    # long_description='',
    # classifiers=[
    #     'Development Status :: 3 - Alpha',
    # ],
    # keywords='sample setuptools development',
    # install_requires=[],
    # package_data={
    #     'sample': ['package_data.dat'],
    # },
    #In the following case, 'data_file' will be installed into '<sys.prefix>/my_data'
    # data_files=[('my_data', ['data/data_file'])],
    # entry_points={
    #     'console_scripts': [
    #         'sample=sample:main',
    #     ],
    # },
)