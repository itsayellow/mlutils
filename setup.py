from setuptools import setup, find_packages

setup(
    name="mlutils",
    version="0.3",
    description="Utilities for training and using Machine Learning models.",
    author="Matthew A. Clapp",
    author_email="itsayellow+dev@gmail.com",
    url="https://github.com/itsayellow/mlutils",
    packages=["mlutils"],
    install_requires=[
        #'tensorflow',
        "keras",
        "numpy",
        "matplotlib",
        "tictoc @ git+https://github.com/itsayellow/tictoc@master",
    ],
    python_requires=">=3.6",
)
