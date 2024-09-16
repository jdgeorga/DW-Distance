from setuptools import setup, find_packages

setup(
    name='dw_distance',
    version='0.1.0',
    author='Johnathan Dimitrios Georgaras',
    author_email='your.email@example.com',
    description='A tool for comparing atomic structures using the Diffusion Wasserstein Distance.',
    packages=find_packages(),
    install_requires=[
        'ase',
        'numpy',
        'scipy',
        'networkx',
        'dscribe',
        'pot',
        'matplotlib',
    ],
)