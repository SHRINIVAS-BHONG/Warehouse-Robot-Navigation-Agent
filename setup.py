from setuptools import setup, find_packages

setup(
    name="warehouse_rl",
    version="1.0.0",
    author="Machine Learning Engineer",
    description="Autonomous Warehouse Robot Navigation via Deep Reinforcement Learning",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "tensorflow",
        "matplotlib",
        "seaborn",
        "streamlit"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.8',
)
