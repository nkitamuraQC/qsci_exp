from setuptools import setup, find_packages

# requirements.txt を読み込む関数
def load_requirements(file_name):
    with open(file_name, 'r') as f:
        return f.read().splitlines()

setup(
    name='QSCI',
    version='0.1', 
    packages=find_packages(),  
    install_requires=load_requirements('requirements.txt'), 
    author='Naoki Kitamura',   
    url='https://github.com/nkitamuraQC/QSCI.git',  
)
