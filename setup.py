#!/usr/bin/env python
# -*- coding: utf-8 -*-

# From https://github.com/navdeep-G/setup.py

import io
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command

NAME = 'Network Distributed Algorithms'
DESCRIPTION = 'Network distributed algorithms and experiments'
URL = 'https://github.com/liboyue/Network-Distributed-Algorithm'
EMAIL = 'boyuel@andrew.cmu.edu'
AUTHOR = 'Boyue Li'
VERSION = '0.1'

# What packages are required for this module to be executed?
REQUIRED = ['matplotlib', 'numpy', 'networkx', 'cvxpy', 'scipy', 'sklearn', 'colorlog']

# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(here, project_slug, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION


# Where the magic happens:
setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    # python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*", "exps"]),
    # If your package is a single module, use this instead of 'packages':
    # py_modules=['nda'],

    # entry_points={
    #     'console_scripts': ['mycli=mymodule:cli'],
    # },
    install_requires=REQUIRED,
    # extras_require=EXTRAS,
    include_package_data=True,
    license='MIT',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        # 'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
)
