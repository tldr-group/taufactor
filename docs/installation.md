# Installation

## Requirements

Before installing taufactor, [download the most recent version of PyTorch](https://pytorch.org/get-started/locally/). Your exact PyTorch configuration will depend on your operating system and GPU availability. Ensure your PyTorch version is `pytorch>=1.0`.

For example, for a Linux machine with CUDA GPU

```
conda install pytorch pytorch-cuda=11.7 -c pytorch -c nvidia
```

## Stable release

To install TauFactor via PyPI

```
pip install taufactor
```

This is the preferred method to install TauFactor, as it will always install the most recent stable release.

If you don't have [pip](https://pip.pypa.io) installed, this [Python installation guide](http://docs.python-guide.org/en/latest/starting/installation/) can guide you through the process.

## Development

These instructions are for running TauFactor in a development environment (i.e. not using the PyPI package)

To access the most recent development branch, run::

    git clone https://github.com/tldr-group/taufactor.git
    cd taufactor
    git checkout development

If running locally, you must adjust `environment.yml` with appropriate CUDA version. Then follow these steps for setup.

```
conda env create -f environment.yml
conda activate taufactor
pip install -e .
```

Taufactor can be installed from

[PyPI](https://pypi.org/project/taufactor/)

[Github](https://github.com/tldr-group/taufactor)

[tarball](https://github.com/tldr-group/taufactor/tarball/master)
