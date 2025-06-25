from setuptools import setup, find_packages

setup(
    name="mindcv",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "mindspore>=2.0.0",
        "mindcv>=0.1.0",
        "numpy>=1.20.0",
        "Pillow>=6.2.0",
        "matplotlib>=3.2.0",
        "tqdm>=4.50.0",
        "scikit-learn>=0.24.0"
    ],
) 