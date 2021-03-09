#!/usr/bin/env python

"""Tests for `taufactor` package."""

import pytest
import taufactor as tau
from taufactor.metrics import volume_fraction
import numpy as np

#  Testing the main solver

def test_solver_on_uniform_block():
    """Run solver on a block of ones."""
    l=20
    img = np.ones([l,l, l]).reshape(1, l, l, l)
    S = tau.Solver(img)
    t = S.solve()
    assert t==1.0

def test_solver_on_empty_block():
    """Run solver on a block of zeros."""
    l=20
    img = np.zeros([l,l, l]).reshape(1, l, l, l)
    S = tau.Solver(img, iter_limit=1000)
    t = S.solve()
    assert t==0.0


def test_solver_on_strip_of_ones():
    """Run solver on a strip of ones, 1/4 volume of total"""
    l=20
    img = np.zeros([l,l, l]).reshape(1, l, l, l)
    t=10
    img[:,:,0:t,0:t]=1
    S = tau.Solver(img)
    t = S.solve()
    assert t==0.25


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

def test_volume_fraction_on_strip_of_ones():
    """Run volume fraction on strip of ones"""
    l=20
    img = np.zeros([l,l,l])
    t=10
    img[:,0:t,0:t]=1
    vf = volume_fraction(img, phases={'zeros':0, 'ones':1})

    assert (vf['zeros'],vf['ones'])==(0.75,0.25) 