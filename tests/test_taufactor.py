#!/usr/bin/env python

"""Tests for `taufactor` package."""

import pytest
import taufactor as tau
from taufactor.metrics import volume_fraction, surface_area
from tests.utils import *
import numpy as np
import matplotlib.pyplot as plt

#  Testing the main solver

def test_solver_on_uniform_block():
    """Run solver on a block of ones."""
    l=20
    img = np.ones([l,l, l]).reshape(1, l, l, l)
    S = tau.Solver(img)
    S.solve()
    assert S.tau==1.0

def test_solver_on_empty_block():
    """Run solver on a block of zeros."""
    l=20
    img = np.zeros([l,l, l]).reshape(1, l, l, l)
    S = tau.Solver(img)
    S.solve(verbose='per_iter', iter_limit=1000)
    assert S.tau==cp.inf


def test_solver_on_strip_of_ones():
    """Run solver on a strip of ones, 1/4 volume of total"""
    l=20
    img = np.zeros([l,l, l]).reshape(1, l, l, l)
    t=10
    img[:,:,0:t,0:t]=1
    S = tau.Solver(img)
    S.solve()
    assert S.tau==1

#  Testing the periodic solver

def test_periodic_solver_on_uniform_block():
    """Run solver on a block of ones."""
    l=20
    img = np.ones([l,l, l]).reshape(1, l, l, l)
    S = tau.PeriodicSolver(img)
    S.solve()
    assert S.tau==1.0

def test_periodic_solver_on_empty_block():
    """Run solver on a block of zeros."""
    l=20
    img = np.zeros([l,l, l]).reshape(1, l, l, l)
    S = tau.PeriodicSolver(img)
    S.solve(verbose='per_iter', iter_limit=1000)
    assert S.tau == cp.inf


def test_periodic_solver_on_strip_of_ones():
    """Run solver on a strip of ones, 1/4 volume of total"""
    l=20
    img = np.zeros([l,l, l]).reshape(1, l, l, l)
    t=10
    img[:,:,0:t,0:t]=1
    S = tau.PeriodicSolver(img)
    S.solve()
    assert S.tau==1

# Testing the metrics
# Volume fraction

def test_volume_fraction_on_uniform_block():
    """Run volume fraction on uniform block"""
    l=20
    img = np.ones([l,l, l]).reshape(1, l, l, l)
    vf = volume_fraction(img)

    assert vf==1.0

def test_volume_fraction_on_empty_block():
    """Run volume fraction on empty block"""
    l=20
    img = np.zeros([l,l, l]).reshape(1, l, l, l)
    vf = volume_fraction(img)

    assert vf==1.0

def test_volume_fraction_on_checkerboard():
    """Run volume fraction on checkerboard block"""
    l=20
    img = generate_checkerboard(l)
    vf = volume_fraction(img)

    assert vf==[0.5, 0.5]

def test_volume_fraction_on_strip_of_ones():
    """Run volume fraction on strip of ones"""
    l=20
    img = np.zeros([l,l,l])
    t=10
    img[:,0:t,0:t]=1
    vf = volume_fraction(img, phases={'zeros':0, 'ones':1})

    assert (vf['zeros'],vf['ones'])==(0.75,0.25)

# Surface area

def test_surface_area_on_uniform_block():
    """Run surface area on uniform block"""
    l=20
    img = np.ones([l,l,l])
    sa = surface_area(img, phases=1)

    assert sa==0

def test_surface_area_on_empty_block():
    """Run surface area on empty block"""
    l=20
    img = np.zeros([l,l, l])
    sa = surface_area(img, phases=0)

    assert sa==0

def test_surface_area_on_checkerboard():
    """Run surface area on checkerboard block"""
    l=50
    img = generate_checkerboard(l)
    sa = surface_area(img, phases=[0,1])

    assert sa==1

def test_surface_area_on_strip_of_ones():
    """Run surface area on single one in small 2x2z2 cube"""
    l=2
    img = np.zeros([l,l,l])
    t=1
    img[0,0,0]=1
    sa = surface_area(img, phases=[0,1])

    assert sa==0.25

def test_surface_area_on_non_periodic_2d():
    """Run surface area on a pair of one in small 3x3 square"""
    img = np.array([[0,0,0],[1,1,0],[0,0,0]])
    sa = surface_area(img, phases=[0,1])

    assert sa==5/12

def test_surface_area_on_periodic_2d():
    """Run surface area on a pair of one in small 3x3 square"""
    img = np.array([[0,0,0],[1,1,0],[0,0,0]])
    sa = surface_area(img, phases=[0,1], periodic=[0,1])

    assert sa==6/18


def test_surface_area_interfactial_3ph():
    l = 3
    img = np.zeros([l, l, l])
    img[1] = 1
    img[2] = 2
    sa = surface_area(img, phases=[1, 2])
    assert sa==1/6

def test_multiphase_and_solver_agree():
    x = 100
    img = np.ones([x, x, x])
    img[50:] = 2
    img[:, :20] = 0
    img[:, 50:] = 1
    s = tau.MultiPhaseSolver(img, {1:1, 2:1*10**-4})
    mph = s.solve(verbose = 'per_iter', conv_crit=0.02)
    img[img==2] = 0
    s = tau.Solver(img)
    s.solve(verbose = 'per_iter')

    err = (mph-s.tau)

    assert err< 0.02

def test_mphsolver_on_empty_block():
    """Run solver on a block of zeros."""
    l=20
    img = np.zeros([l,l, l]).reshape(1, l, l, l)
    S = tau.MultiPhaseSolver(img)
    S.solve(iter_limit=1000)
    assert S.tau==cp.inf

def test_mphsolver_on_ones_block():
    """Run solver on a block of ones."""
    l=20
    img = np.ones([l,l, l]).reshape(1, l, l, l)
    S = tau.MultiPhaseSolver(img)
    S.solve(iter_limit=1000)
    assert np.around(S.tau,4)==1.0

def test_mphsolver_on_halves():
    """Run solver on a block of halves."""
    l=20
    img = np.ones([l,l, l]).reshape(1, l, l, l)
    cond = 0.5
    S = tau.MultiPhaseSolver(img, {1:cond})
    S.solve(iter_limit=1000)
    print(S.D_eff, S.D_mean)
    assert np.around(S.tau,4)==1.0

def test_mphsolver_on_strip_of_ones():
    """Run solver on a strip of ones, 1/4 volume of total"""
    l=20
    img = np.zeros([l,l, l]).reshape(1, l, l, l)
    x=10
    img[:,:,0:x,0:x]=1
    S = tau.MultiPhaseSolver(img)
    S.solve()
    assert np.around(S.tau,4)==1.0

def test_mphsolver_on_strip_of_ones_and_twos():
    """Run solver on a strip of ones, 1/4 volume of total"""
    l=20
    img = np.zeros([l,l, l]).reshape(1, l, l, l)
    x=10
    img[:,:,0:x,0:x]=1
    img[:,:,0:x,x:l]=2
    cond = {1:1, 2:0.5}
    S = tau.MultiPhaseSolver(img, cond)
    S.solve()
    assert np.around(S.tau,4)==1


def test_mphsolver_on_strip_of_ones_and_twos_and_threes():
    """Run solver on a strip of ones, 1/4 volume of total"""
    l=20
    img = np.ones([l,l, l]).reshape(1, l, l, l)
    x=10
    img[:,:,0:x,0:x]=2
    img[:,:,0:x,x:l]=3
    cond = {1:1, 2:0.5, 3:2}
    S = tau.MultiPhaseSolver(img, cond)
    S.solve()
    assert np.around(S.tau,4)==1
