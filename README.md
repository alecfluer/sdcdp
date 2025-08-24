# **SDCDP**

## **Overview**

SDCDP is a Python package for sampling networks using the **Social Distance Configuration model with Degree Preservation (SDC-DP)**. It includes tools for generating correlated degree sequences and computing connection probabilities using the **Social Distance Attachment (SDA)** method generalized to undirected multiplex networks.

These methods were developed to accompany [Fluer et al. (2025)](#references). However, the package is fully standalone and suitable for independent use.

## **Modules**

The `src/sdcdp/` folder contains the following modules:

- `sdcdp.py`: Provides the `sdcdp_model` function for sampling networks using SDC-DP.
- `sda.py`: Provides the `MultiplexSDA` class for computing connection probabilities using the SDA method.
- `seq.py`: Provides a collection of functions that work together for generating correlated degree sequences.

## **Documentation**

The `docs/` folder contains the following Markdown files:

- `sdcdp.md`: Contains documentation and full mathematical details for the `sdcdp.py` module.
- `sda.md`: Contains documentation and full mathematical details for the `sda.py` module.
- `seq.md`: Contains documentation and full mathematical details for the `seq.py` module.

## **Dependencies**

Requires Python ≥ 3.8 and the following packages:

- `networkx`
- `numpy`
- `pandas`
- `scipy`

## **Installation**

Clone the repository and install locally as follows:

```bash
git clone https://github.com/alecfluer/sdcdp
cd sdcdp
pip install .
```

## **License**

Copyright © 2025 Alec Fluer  
This software is licensed under the MIT License. See LICENSE for details.

## **References**

Alec Fluer, Ian Laga, Logan Graham, Ellen Almirol, Makenna Meyer, and Breschine Cummins.  
From survey data to social multiplex models: Incorporating interlayer correlation from multiple data sources. In preparation.