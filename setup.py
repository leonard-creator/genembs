# setup file for the generate new embeddings Module 
# to install via pip install -e . (source files will be immediately available to other users of the package on our system)
# March 2022

# use like this in Python:
# import genembs 
# my.function()

#TODO: add gitHub

from setuptools import setup

setup(name='genembs',
      version='0.1.0',
      description='generate embeddings',
      long_description='new way to train embeddings with hyperparameter optimisation, immediatly fetch learned embeddings and save them as Emb_dict.'
      url='http://github.com/',
      author='Leonard Tiling',
      author_email='leonard.tiling@charite.de',
      license='MIT',
      packages=['genembs'],
      zip_safe=False)