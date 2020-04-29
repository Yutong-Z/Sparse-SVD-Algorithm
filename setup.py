#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 16:28:10 2020

@author: yutongzhang
"""

from setuptools import setup,find_packages

with open("README.md", "r") as fh:
  long_description = fh.read()
  
setup(name='SSVD663',
      version='1.0',
      description='A python relization of sparse singular value decomposition algorithm. It is based on paper "Biclustering via Sparse Singular Value Decomposition" written by Mihee Lee, Haipeng Shen, Jianhua Z. Huang, and J. S. Marron.',
      url='https://github.com/Yutong-Z/Sparse-SVD-Algorithm.git',
      author='Jiaxi Yin, Yutong Zhang',
      author_email='jy280@duke.edu,yz566@duke.edu',
      license='MIT',
      packages=find_packages(),
      zip_safe=False,
      install_requires=['numpy','pandas',],
      )