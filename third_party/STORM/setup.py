import pathlib

import setuptools
from setuptools import find_namespace_packages

setuptools.setup(
    name="storm_wm",
    version="1.0.0",
    description="STORM: Efficient Stochastic Transformer based World Models for Reinforcement Learning",
    author="Weipu Zhang et al.",
    long_description=pathlib.Path("readme.md").read_text(),
    long_description_content_type="text/markdown",
    packages=find_namespace_packages(
        include=["storm_wm"]
    ),
    include_package_data=True,
    install_requires=pathlib.Path("requirements.txt").read_text().splitlines(),
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
