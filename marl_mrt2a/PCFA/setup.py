#!/usr/bin/env python3
"""
Setup script for the PCFA package.

This package implements the market-based task allocation algorithm described in:
Oh, G., Kim, Y., Ahn, J., & Choi, H. L. (2017). Market-Based Task Assignment 
for Cooperative Timing Missions in Dynamic Environments. Journal of Intelligent 
and Robotic Systems, 87(1), 97-123. https://doi.org/10.1007/s10846-017-0493-x
"""

from setuptools import setup, find_packages

# Read the README file for long description
try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = """
    Project Manager-oriented Coalition Formation Algorithm (PCFA) for multi-agent task allocation.
    
    This implementation is based on the market-based task allocation algorithm described in:
    Oh, G., Kim, Y., Ahn, J., & Choi, H. L. (2017). Market-Based Task Assignment 
    for Cooperative Timing Missions in Dynamic Environments. Journal of Intelligent 
    and Robotic Systems, 87(1), 97-123. https://doi.org/10.1007/s10846-017-0493-x
    """

setup(
    name="PCFA",
    version="0.1.0",
    author="Lucas C. D. Bezerra",
    author_email="lcdbezerra@gmail.com",
    description="Project Manager-oriented Coalition Formation Algorithm for multi-agent task allocation (based on Oh et al., 2017)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "gymrt2a",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
        ],
    },
    entry_points={
        "console_scripts": [
            "pcfa-example=example:main",
        ],
    },
    include_package_data=True,
    package_data={
        "PCFA": ["README.md"],
        "": ["LICENSE"],
    },
)
