from setuptools import find_packages,setup
from typing import List
def get_requirements(file_path:str)-> List[str]:
    '''
    this function will return a list of requirements
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

setup(
    name='Crop-Yield-Estimation-Model',
    version='0.0.1',
    author='Parmita Ghosh',
    author_email='paramita.january841@gmail.com',
    packages=find_packages(),
    install_require=get_requirements('requirement.txt')
)