#!/usr/bin/env python
import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

macros = []
cmdclass={}
setup(
    name = "kaggle",
    version = "0.0.1",
    author = "Liam Damewood",
    author_email = "liam.physics82@gmail.com",
    description = ("Kaggle helper scripts"),
    license = "MIT",
    packages=['kaggle'],
    long_description=read('README.md'),
)
