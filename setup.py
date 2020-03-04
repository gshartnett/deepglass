#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

readme = open('README.md').read()

requirements = [
    'numpy',
    'torch',
    'scikit-learn',
    'scipy',
    'matplotlib',
    'seaborn',
    'pathlib',
    'jupyter',
    'ipykernel',
    'tqdm',
    'ipywidgets'
]

setup(
    name='deepglass',
    version='1.0.0',
    description='Deep generative spin-glass models with normalizing flows https://arxiv.org/abs/2001.00585',
    author='Gavin Hartnett',
    author_email='gshartnett@gmail.com',
    url='https://github.com/gshartnett/deepglass',
    include_package_data=True,
    install_requires=requirements,
    license="MIT",
)
