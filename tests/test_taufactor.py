#!/usr/bin/env python

"""Tests for `taufactor` package."""

import taufactor as tau
import torch as pt
import numpy as np


#  Testing the main solver

def test_solver_on_uniform_block():
    """Run solver on a block of ones."""
    l = 20
    img = np.ones([l, l, l]).reshape(1, l, l, l)
    img[:, :, 0] = 0
    S = tau.Solver(img, device=pt.device('cpu'))
    S.solve()
    assert np.around(S.tau, decimals=5) == 1.0


def test_solver_on_uniform_rectangular_block_solver_dim():
    """Run solver on a block of ones."""
    l = 20
    img = np.ones([l*2, l, l]).reshape(1, l*2, l, l)
    img[:, :, 0] = 0
    S = tau.Solver(img, device=pt.device('cpu'))
    S.solve()
    assert np.around(S.tau, decimals=5) == 1.0


def test_solver_on_uniform_rectangular_block_non_solver_dim():
    """Run solver on a block of ones."""
    l = 20
    img = np.ones([l, l, l*2]).reshape(1, l, l, l*2)
    img[:, :, 0] = 0
    S = tau.Solver(img, device=pt.device('cpu'))
    S.solve()
    assert np.around(S.tau, decimals=5) == 1.0


def test_solver_on_empty_block():
    """Run solver on a block of zeros."""
    l = 20
    img = np.zeros([l, l, l]).reshape(1, l, l, l)
    img[:, 0] = 1
    S = tau.Solver(img, device=pt.device('cpu'))
    S.solve(verbose='per_iter', iter_limit=1000)
    assert S.tau == pt.inf


def test_solver_on_strip_of_ones():
    """Run solver on a strip of ones, 1/4 volume of total"""
    l = 20
    img = np.zeros([l, l, l]).reshape(1, l, l, l)
    t = 10
    img[:, :, 0:t, 0:t] = 1
    S = tau.Solver(img, device=pt.device('cpu'))
    S.solve()
    assert np.around(S.tau, decimals=5) == 1

def test_solver_on_slanted_strip_of_ones():
    """Run solver on a slanted strip of ones"""
    l = 20
    img = np.zeros([l, l+1, l+1]).reshape(1, l, l+1, l+1)
    # t = 10
    # img[:, :, 0:t, 0:t] = 1
    for i in range(l):
        img[:, i, i:i+2, i:i+2] = 1
    S = tau.Solver(img, device=pt.device('cpu'))
    S.solve()
    assert np.around(S.tau, decimals=5) == 4


#  Testing the periodic solver

def test_deadend():
    """Test deadend pore"""
    solid  = np.zeros((10,50,50))
    solid[:8, 25,25] = 1
    # solve for tau
    S = tau.Solver(solid)
    S.solve()
    assert np.around(S.D_eff, decimals=5) == 0
    assert S.tau == np.inf

def test_periodic_solver_on_uniform_block():
    """Run periodic solver on a block of ones."""
    l = 20
    img = np.ones([l, l, l]).reshape(1, l, l, l)
    img[:, :, 0] = 0
    S = tau.PeriodicSolver(img, device=pt.device('cpu'))
    S.solve()
    assert np.around(S.tau, decimals=5) == 1.0


def test_periodic_solver_on_empty_block():
    """Run periodic solver on a block of zeros."""
    l = 20
    img = np.zeros([l, l, l]).reshape(1, l, l, l)
    img[:, 0] = 1
    S = tau.PeriodicSolver(img, device=pt.device('cpu'))
    S.solve(verbose='per_iter', iter_limit=1000)
    assert S.tau == pt.inf


def test_periodic_solver_on_strip_of_ones():
    """Run periodic solver on a strip of ones, 1/4 volume of total"""
    l = 20
    img = np.zeros([l, l, l]).reshape(1, l, l, l)
    t = 10
    img[:, :, 0:t, 0:t] = 1
    S = tau.PeriodicSolver(img, device=pt.device('cpu'))
    S.solve()
    assert np.around(S.tau, decimals=5) == 1


#  Testing the multiphase solver

def test_multiphase_and_solver_agree():
    """test mph and solver agree when Ds are the same"""
    x = 100
    img = np.ones([x, x, x])
    img[50:] = 2
    img[:, :20] = 0
    img[:, 50:] = 1
    s = tau.MultiPhaseSolver(img, {1: 1, 2: 1*10**-4}, device=pt.device('cpu'))
    mph = s.solve(verbose='per_iter', conv_crit=0.02)
    img[img == 2] = 0
    s = tau.Solver(img, device=pt.device('cpu'))
    s.solve(verbose='per_iter')

    err = (mph-s.tau)

    assert err < 0.02


def test_mphsolver_on_empty_block():
    """Run mpsolver on a block of zeros."""
    l = 20
    img = np.zeros([l, l, l]).reshape(1, l, l, l)
    S = tau.MultiPhaseSolver(img, device=pt.device('cpu'))
    S.solve(iter_limit=1000)
    assert S.tau == pt.inf


def test_mphsolver_on_ones_block():
    """Run mpsolver on a block of ones."""
    l = 20
    img = np.ones([l, l, l]).reshape(1, l, l, l)
    S = tau.MultiPhaseSolver(img, device=pt.device('cpu'))
    S.solve(iter_limit=1000)
    assert np.around(S.tau, 4) == 1.0


def test_mphsolver_on_halves():
    """Run mpsolver on a block of halves."""
    l = 20
    img = np.ones([l, l, l]).reshape(1, l, l, l)
    cond = 0.5
    S = tau.MultiPhaseSolver(img, {1: cond}, device=pt.device('cpu'))
    S.solve(iter_limit=1000)
    print(S.D_eff, S.D_mean)
    assert np.around(S.tau, 4) == 1.0


def test_mphsolver_on_strip_of_ones():
    """Run mpsolver on a strip of ones, 1/4 volume of total"""
    l = 20
    img = np.zeros([l, l, l]).reshape(1, l, l, l)
    x = 10
    img[:, :, 0:x, 0:x] = 1
    S = tau.MultiPhaseSolver(img, device=pt.device('cpu'))
    S.solve()
    assert np.around(S.tau, 4) == 1.0


def test_mphsolver_on_strip_of_ones_and_twos():
    """Run solver on a strip of ones, 1/4 volume of total"""
    l = 20
    img = np.zeros([l, l, l]).reshape(1, l, l, l)
    x = 10
    img[:, :, 0:x, 0:x] = 1
    img[:, :, 0:x, x:l] = 2
    cond = {1: 1, 2: 0.5}
    S = tau.MultiPhaseSolver(img, cond, device=pt.device('cpu'))
    S.solve()
    assert np.around(S.tau, 4) == 1


def test_mphsolver_on_strip_of_ones_and_twos_and_threes():
    """Run solver on a strip of ones, 1/4 volume of total"""
    l = 20
    img = np.ones([l, l, l]).reshape(1, l, l, l)
    x = 10
    img[:, :, 0:x, 0:x] = 2
    img[:, :, 0:x, x:l] = 3
    cond = {1: 1, 2: 0.5, 3: 2}
    S = tau.MultiPhaseSolver(img, cond, device=pt.device('cpu'))
    S.solve()
    assert np.around(S.tau, 4) == 1


#  Testing the tau_e solver

def test_taue_deadend():
    """Run solver on a deadend strip of ones"""

    l = 100
    img = np.zeros((l, l))
    img[:75, 45:55] = 1
    img[img != 1] = 0
    esolver = tau.ElectrodeSolver(img, device=pt.device('cpu'))
    esolver.solve()
    assert np.around(esolver.tau_e, 3) == 0.601


def test_taue_throughpore():
    """Run taue solver on a strip of ones, 1/4 volume of total"""

    l = 100
    img = np.zeros((l, l))
    img[:, 45:55] = 1
    img[img != 1] = 0
    esolver = tau.ElectrodeSolver(img, device=pt.device('cpu'))
    esolver.solve()
    assert np.around(esolver.tau_e, 3) == 1.046
