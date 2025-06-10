from setuptools import setup, find_packages

setup(
    name="eye_disease_classifier",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "mindspore==2.6.0",
        "numpy>=1.20.0",
        "Pillow>=6.2.0",
        "matplotlib>=3.2.0",
        "tqdm>=4.50.0",
        "scikit-learn>=0.24.0",
    ],
) 