from setuptools import setup, find_packages

setup(
    name="sdcdp",
    version="0.1.0",
    description=(
        "Provides the Social Distance Configuration model with Degree Preservation (SDC-DP) for sampling networks. "
        "Includes utilities for generating correlated degree sequences and computing connection probabilities using "
        "the Social Distance Attachment (SDA) method generalized to undirected multiplex networks."
    ),
    author="Alec Fluer",
    url="https://github.com/alecfluer/sdcdp",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "networkx",
        "numpy",
        "pandas",
        "scipy"
    ]
)