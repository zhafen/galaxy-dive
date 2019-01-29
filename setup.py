import setuptools

setuptools.setup(
    name='galaxy_dive',
    version='0.9.0',
    description='A general analysis suite for hydrodynamic galaxy simulations.',
    url='https://github.com/zhafen/galaxy-dive',
    author='Zach Hafen',
    author_email='zachary.h.hafen@gmail.com',
    packages=setuptools.find_packages(),
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
    ],
    install_requires=[
        'pandas>=0.20.3',
        'mock>=2.0.0',
        'numpy>=1.14.5',
        'pytest>=3.4.0',
        'unyt>=1.0.4',
        'six>=1.10.0',
        'setuptools>=28.8.0',
        'colossus>=1.2.2',
        'matplotlib>=2.0.2',
        'h5py>=2.7.0',
        'scipy>=1.2.0',
        'verdict>=1.0',
    ],
    include_package_data=True,
)
