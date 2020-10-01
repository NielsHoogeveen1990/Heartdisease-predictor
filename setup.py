from setuptools import setup, find_packages
import os

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='heart-disease',
    keywords='',
    version='1.0',
    author='Niels Hoogeveen',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    description='ML model to predict whether a person has a heart disease.',
    entry_points={'console_scripts': ['heartdisease = heartdisease.cli:main']},
    long_description=read('README.md'),
    long_description_content_type='text/markdown'
)
