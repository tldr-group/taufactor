"""Top-level package for TauFactor."""

__version__ = '1.2.0'

from .taufactor import Solver, AnisotropicSolver, PeriodicSolver, \
                       MultiPhaseSolver, ElectrodeSolver

__all__ = ['Solver', 'AnisotropicSolver', 'PeriodicSolver',\
           'MultiPhaseSolver',\
           'ElectrodeSolver']
