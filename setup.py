from setuptools import find_packages, setup
from typing import List

e_dot = '-e .'

def get_requirements(filepath:str)->List[str]:
    R = []
    with open(filepath) as f:
        R = f.readlines()
        R = [r.replace('\n','') for r in R]
        if e_dot in R:
            R.remove(e_dot)
        return R
    
setup(name='Diamond Price Prediction', version='0.1', author='Piyush', packages=find_packages(),
      install_requires = get_requirements('requirements.txt') )