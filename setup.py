from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content]

setup(name='attention',
      version="0.0.10",
      description="Attention tracker for audience in presentation",
      author="Le Wagon 1137",
      install_requires=requirements,
      packages=find_packages()
      )
