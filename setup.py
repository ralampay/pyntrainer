from setuptools import find_packages
from setuptools import setup

setup(
  name='pyntrainer',
  py_modules=['pyntrainer'],
  version='0.1',
  description='python autoencoder trainer tool',
  author='Raphael Alampay',
  author_email='raphael.alampay@gmail.com',
  url='https://github.com/ralampay/pyntrainer',
  packages=find_packages(),
  entry_points={
    'console_scripts': [
      'pyntrainer-cli = pyntrainer.__main__:main'
    ]
  }
)
