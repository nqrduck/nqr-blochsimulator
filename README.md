# NQR Bloch Simulator for Python

This is a Python implementation of an NQR Bloch Simulator. It can be used for simulating NQR spectroscopy signals in the time domain. The simulator is based on the paper [1].

Right now the implementation is in a early stage and has not yet been tested and verified.

## Installation

Create a virtual environment and activate it:

```
python -m venv venv
source venv/bin/activate
```

To install the package, run the following command in the root directory of the project:

```
pip install .
```

Alternatively you can install the package via PyPI:

```
pip install nqr-blochsimulator
```

The package can then be tested by running

```
python -m unittest tests/simulation.py
```

This will run a simulation of a simple FID for BiPh3 and plot the result in time domain.


## References

The simulator is based on the paper:

[1] [C. Graf, A. Rund, C.S. Aigner, R. Stollberger, Accuracy and Performance Analysis for Bloch and Bloch-McConnell simulation methods Journal of Magnetic Resonance 329(3):107011 doi: 10.1016/j.jmr.2021.107011](https://doi.org/10.1016/j.jmr.2021.107011)