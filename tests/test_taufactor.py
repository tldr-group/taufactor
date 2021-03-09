#!/usr/bin/env python

"""Tests for `taufactor` package."""

import pytest

# from click.testing import CliRunner

import taufactor as tau
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
    """Run solver on a strip of ones, 1/2 width of total"""
    l=20
    img = np.zeros([l,l, l]).reshape(1, l, l, l)
    t=10
    img[:,:,0:t,0:t]=1
    print(img)
    S = tau.Solver(img)
    t = S.solve()
    assert t==0.25
