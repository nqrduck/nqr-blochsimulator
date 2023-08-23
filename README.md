# NQR Bloch Simulator for Python

This is a Python implementation of an NQR Bloch Simulator.

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

The package can then be tested by running

```
python -m unittest tests/simulation.py
```

This will run a simulation of a simple FID for BiPh3 and plot the result in time domain.


## References