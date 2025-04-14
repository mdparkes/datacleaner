from setuptools import setup, find_packages

setup(
    name="datacleaner",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
    ],
    python_requires=">=3.7",
)
