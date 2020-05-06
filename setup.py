"""
Array combining tool using numba
"""

from setuptools import setup, find_packages

setup_requires = []
install_requires = ['numpy',
                    'numba']

classifiers = ["Intended Audience :: Science/Research",
               "Operating System :: OS Independent",
               "Programming Language :: Python :: 3"]

setup(
    name="nbcombine",
    version="0.1",
    author="Yoonsoo P. Bach",
    author_email="dbstn95@gmail.com",
    description="",
    license="",
    keywords="",
    url="",
    classifiers=classifiers,
    packages=find_packages(),
    python_requires='>=3.5',
    install_requires=install_requires)
