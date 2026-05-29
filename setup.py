from setuptools import setup, find_packages

setup(
    name="EpiTADformer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
    "numpy>=1.24",
    "pandas>=2.0",
    "tensorflow>=2.15",
    "scikit-learn>=1.3"
    ],
    entry_points={
        "console_scripts": [
            "epitadformer-train=EpiTADformer.train:main",
            "epitadformer-predict=EpiTADformer.predict:main",
            "epitadformer-data=EpiTADformer.data_train:main",
        ],
    },
)