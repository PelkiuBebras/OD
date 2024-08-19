#!/usr/bin/env python3

from pathlib import Path
from setuptools import setup

directory = Path(__file__).resolve().parent
with open(directory / 'README.md', encoding='utf-8') as f:
  long_description = f.read()

setup(name='od',
      version='0.0.1',
      license='MIT',
      long_description=long_description,
      long_description_content_type='text/markdown',
      install_requires=[
        "Pillow",
        "tinygrad @ git+https://github.com/tinygrad/tinygrad.git"
      ],
      python_requires='>=3.8')