# TauFactor

TauFactor is an application for calculating tortuosity factors from tomographic data.

-   Free software: MIT license
-   Documentation: [https://taufactor.readthedocs.io](https://taufactor.readthedocs.io).

<p align="center">
<img src="https://tldr-group.github.io/static/media/tau_example.2c29eaf9.png" alt="TauFactor" width="324" height="324">
</p>
<p align="center">
<a href="https://pypi.python.org/pypi/taufactor">
        <img src="https://img.shields.io/pypi/v/taufactor.svg"
            alt="PyPI"></a>
<a href="https://taufactor.readthedocs.io/en/latest/?badge=latest">
        <img src="https://readthedocs.org/projects/taufactor/badge/?version=latest"
            alt="ReadTheDocs"></a>
<a href="https://opensource.org/licenses/MIT">
        <img src="https://img.shields.io/badge/License-MIT-yellow.svg"
            alt="MIT LICENSE"></a>
<img src="https://github.com/tldr-group/taufactor/actions/workflows/taufactor.yml/badge.svg"
        alt="github actions">

</p>

## Requirements

Before installing taufactor, [download the most recent version of PyTorch](https://pytorch.org/get-started/locally/). Ensure you have `pytorch>=1.10` installed in your Python environment.

For example, for a Linux machine with CUDA GPU

```
conda install pytorch pytorch-cuda=11.7 -c pytorch -c nvidia
```

## Quickstart

To install TauFactor via PyPI

```
pip install taufactor
```

To extract effective diffusivity and tortuosity factor from your data:

```python
import taufactor as tau
import tifffile

# load image
img = tifffile.imread('path/filename')
# ensure 1s for conductive phase and 0s otherwise.

# create a solver object with loaded image
s = tau.Solver(img)

# call solve function
s.solve()

# view effective diffusivity and tau
print(s.D_eff, s.tau)

```

## Tests

To run unit tests navigate to the root directory and run

```
pytest
```

## Credits

This package was created by the [tldr group](https://tldr-group.github.io/) at the Dyson School of Design Engineering, Imperial College London.

## TauFactor MATLAB

The package in this repository refers to a Python implementation of the TauFactor solver. There is a deprecated [MATLAB implementation](https://www.mathworks.com/matlabcentral/fileexchange/57956-taufactor), which is no longer maintained.
