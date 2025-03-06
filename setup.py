from setuptools import setup, find_packages

setup(
    name="synthetic_data_generator",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "Pillow>=8.2.0",
        "scikit-learn>=0.24.0",
        "faker>=8.0.0",
        "tqdm>=4.60.0",
        "torch>=1.9.0",
        "transformers>=4.5.0",
        "librosa>=0.8.0",
        "soundfile>=0.10.0",
    ],
    author="Synthetic Data Generator Team",
    author_email="aumparmar@gmail.com",
    description="A comprehensive library for generating synthetic data",
    keywords="synthetic, data, generation, tabular, image, text, time-series, audio",
    python_requires=">=3.8",
)
