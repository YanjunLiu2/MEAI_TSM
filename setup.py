from setuptools import find_namespace_packages, setup

requirements = [
    'gpytorch==1.6.0',
    'matplotlib==3.4.3',
    'xlrd==2.0.1',
    'botorch==0.4.0',
    'xlwt==1.3.0',
]

setup(name='MEAI_TSM',
      version='1.0',
      description='MEAI for TSM',
      author='Yanjun Liu',
      install_requires=requirements,
      packages=find_namespace_packages(include=['MEAI_TSM*']))