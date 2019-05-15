from setuptools import setup, find_packages

setup(
        name='mlutils',
        version='0.1',
        description='Utilities for training and using Machine Learning models.',
        author='Matthew A. Clapp',
        author_email='itsayellow+dev@gmail.com',
        packages=[
            'mlutils'
            ],
        install_requires=[
            'tensorflow',
            'keras',
            'numpy',
            'matplotlib',
            ],
        )
