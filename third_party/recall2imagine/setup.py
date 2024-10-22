import pathlib

import setuptools
from setuptools import find_namespace_packages

setuptools.setup(
    name="recall2imagine",
    version="1.0.0",
    description="Mastering Memory Tasks with World Models",
    author="Chandar Lab",
    url="https://github.com/chandar-lab/Recall2Imagine",
    long_description=pathlib.Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    packages=find_namespace_packages(
        include=["recall2imagine"], exclude=["example.py"]
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
