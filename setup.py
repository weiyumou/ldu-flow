#!/usr/bin/env python

from setuptools import setup, find_packages

# Package meta-data.
NAME = "project"
DESCRIPTION = "Sinusoidal Flow"
URL = "https://github.com/weiyumou/ldu-flow.git"
EMAIL = ""
AUTHOR = ""
REQUIRES_PYTHON = ">=3.8.0"
VERSION = "0.1.0"

# What packages are required for this module to be executed?
REQUIRED = [
    "numpy==1.19.2", "torch==1.7.1", "pytorch-lightning==1.1.1",
    "matplotlib==3.3.2", "scikit-learn==0.22.1", "pandas==1.2.0",
    "h5py==2.10.0"
]
# What packages are optional?
EXTRAS = {

}

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    # If your package is a single module, use this instead of "packages":
    # py_modules=["mypackage"],
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True
)
