from setuptools import setup, find_packages

setup(
    name='echoedge_lib',
    version='0.1',
    packages=find_packages(where='lib'),
    package_dir={'': 'lib'},
)