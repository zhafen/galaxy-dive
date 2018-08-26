import setuptools

setuptools.setup(
    name='galaxy_dive',
    version='0.8.1.2',
    description='A general analysis suite for hydrodynamic galaxy simulations.',
    url='https://github.com/zhafen/galaxy-dive',
    author='Zach Hafen',
    author_email='zachary.h.hafen@gmail.com',
    packages=setuptools.find_packages(),
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.7',
    ],
    install_requires=[
        'numpy>=1.14.5',
        'pandas>=0.20.3',
        'mock>=2.0.0',
        'pytest>=3.4.0',
        'unyt>=1.0.4',
        'colossus>=1.2.1',
        'setuptools>=28.8.0',
        'matplotlib>=2.0.2',
        'h5py>=2.7.0',
        'scipy>=1.1.0',
    ],
    include_package_data=True,
)
