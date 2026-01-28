"""Setup configuration for enzykit package."""

from setuptools import setup, find_packages

setup(
    name="enzykit",
    version="0.1.0",
    description="Tools for enzyme assay analysis using photodiode array spectrophotometry",
    author="Dan Olson",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "plotly>=5.0.0",
        "statsmodels>=0.13.0",
        "scipy>=1.7.0",
    ],
    python_requires=">=3.8",
)
