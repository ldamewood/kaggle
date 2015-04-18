#!/usr/bin/env python
from setuptools import setup

macros = []
cmdclass={}
setup(name='kaggle',
      version='0.0.1',
      author="Liam Damewood",
      author_email="liam.physics82@gmail.com",
      description="Helper routines for Kaggle",
      url="http://github.org/ldamewood/kaggle",
      license='MIT',
      cmdclass = cmdclass,
      ext_modules = ext_modules,
      packages=['kaggle'],
      )
