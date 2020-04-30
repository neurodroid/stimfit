try:
    from setuptools import core
except Exception:
    from distutils.core import setup

setup(name='stfio',
      version='0.16.0',
      description='stfio module',
      package_dir={'stfio': '.'},
      packages=['stfio'],
      package_data={'stfio': ['_stfio.so']},
      )
