from setuptools import setup


setup(
    name='tsdb',
    packages=['tsdb'],
    install_requires=[
             'numba>=0.32', 
			'numpy>=1.10',
             'pandas>=0.19',
             'bcolz>=1.1.2',
             ]
    )