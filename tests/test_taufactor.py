"""Tests for `taufactor` package."""

import numpy as np
import pytest
import torch as pt

import taufactor as tau


###  Testing the main solver
def test_solver_on_uniform_block():
    """Run solver on a block of ones."""
    N = 20
    img = np.ones((N, N, N))
    img[:, 0] = 0
    S = tau.Solver(img, device='cpu')
    S.solve()
    assert np.around(S.tau, decimals=5) == 1.0

def test_solver_on_uniform_rectangular_block_solver_dim():
    """Run solver on a block of ones."""
    N = 20
    img = np.ones((N*2, N, N))
    img[:, 0] = 0
    S = tau.Solver(img, device='cpu')
    S.solve()
    assert np.around(S.tau, decimals=5) == 1.0

def test_solver_on_uniform_rectangular_block_non_solver_dim():
    """Run solver on a block of ones."""
    N = 20
    img = np.ones((N, N, N*2))
    img[:, 0] = 0
    S = tau.Solver(img, device='cpu')
    S.solve()
    assert np.around(S.tau, decimals=5) == 1.0

def test_solver_non_percolating():
    """Run solver on a block of zeros."""
    N = 20
    img = np.zeros((N, N, N))
    img[:2] = 1
    S = tau.Solver(img, device='cpu')
    S.solve(verbose='per_iter', iter_limit=1000)
    assert S.tau == pt.inf


def test_solver_records_tau_history_without_debug():
    S = tau.Solver(np.ones((10, 10, 10)), device='cpu')

    S.solve(verbose=False, iter_limit=100)

    assert len(S.tau_t) == 1
    np.testing.assert_allclose(S.tau_t[-1], S.tau)

def test_solver_on_strip_of_ones():
    """Run solver on a strip of ones, 1/4 volume of total"""
    N = 20
    img = np.zeros((N, N, N))
    t = 10
    img[:, 0:t, 0:t] = 1
    S = tau.Solver(img, device='cpu')
    S.solve()
    assert np.around(S.tau, decimals=5) == 1

def test_solver_on_slanted_strip_of_ones():
    """Run solver on a slanted strip of ones"""
    N = 20
    img = np.zeros((N, N+1, N+1))
    for i in range(N):
        img[i, i:i+2, i:i+2] = 1
    S = tau.Solver(img, device='cpu')
    S.solve()
    assert np.around(S.tau, decimals=5) == 7.51668

def test_deadend():
    """Test deadend pore"""
    solid  = np.zeros((10,50,50))
    solid[:8, 25, 25] = 1
    # solve for tau
    S = tau.Solver(solid)
    S.solve()
    assert np.around(S.D_eff, decimals=5) == 0
    assert S.tau == np.inf


###  Testing the periodic solver
def test_periodic_solver_on_uniform_block():
    """Run periodic solver on a block of ones."""
    N = 20
    img = np.ones((N, N, N))
    img[:, 0] = 0
    S = tau.PeriodicSolver(img, device='cpu')
    S.solve()
    assert np.around(S.tau, decimals=5) == 1.0

def test_periodic_solver_non_percolating():
    """Run periodic solver on a block of zeros."""
    N = 20
    img = np.zeros((N, N, N))
    img[:2] = 1
    S = tau.PeriodicSolver(img, device='cpu')
    S.solve(verbose='per_iter', iter_limit=1000)
    assert S.tau == pt.inf

def test_connectivity_settings_match_solver_boundaries():
    """Periodic and electrode solvers use their respective connectivity rules."""
    img = np.zeros((4, 3, 1))
    img[0, 0] = 1
    img[1, (0, 2)] = 1
    img[2, 2] = 1
    img[3, 2] = 1

    assert not tau.Solver(img, device='cpu').percolates[0]
    assert tau.PeriodicSolver(img, device='cpu').percolates[0]

    electrode_img = np.zeros((4, 3, 1))
    electrode_img[:2, 1] = 1
    assert tau.ElectrodeSolver(electrode_img, device='cpu').percolates[0]

def test_periodic_solver_on_strip_of_ones():
    """Run periodic solver on a strip of ones, 1/4 volume of total"""
    N = 20
    img = np.zeros((N, N, N))
    t = 10
    img[:, 0:t, 0:t] = 1
    S = tau.PeriodicSolver(img, device='cpu')
    S.solve()
    assert np.around(S.tau, decimals=5) == 1


###  Testing the multiphase solver
def test_mphsolver_non_percolating():
    """Run mpsolver on a block of zeros."""
    N = 20
    img = np.zeros((N, N, N))
    img[:2] = 1
    S = tau.MultiPhaseSolver(img, device='cpu')
    S.solve(iter_limit=1000)
    assert S.tau == pt.inf

def test_mphsolver_on_ones_block():
    """Run mpsolver on a block of ones."""
    N = 20
    img = np.ones((N, N, N))
    S = tau.MultiPhaseSolver(img, device='cpu')
    S.solve(iter_limit=1000)
    assert np.around(S.tau, 4) == 1.0

def test_mphsolver_on_halves():
    """Run mpsolver on a block of halves."""
    N = 20
    img = np.ones((N, N, N))
    cond = 0.5
    S = tau.MultiPhaseSolver(img, {1: cond}, device='cpu')
    S.solve(iter_limit=1000)
    assert np.around(S.tau, 4) == 1.0

def test_mphsolver_on_strip_of_ones():
    """Run mpsolver on a strip of ones, 1/4 volume of total"""
    N = 20
    img = np.zeros((N, N, N))
    x = 10
    img[:, 0:x, 0:x] = 1
    S = tau.MultiPhaseSolver(img, device='cpu')
    S.solve()
    assert np.around(S.tau, 4) == 1.0

def test_mphsolver_on_strip_of_ones_and_twos():
    """Run solver on a strip of ones, 1/4 volume of total"""
    N = 20
    img = np.zeros((N, N, N))
    x = 10
    img[:, 0:x, 0:x] = 1
    img[:, 0:x, x:N] = 2
    Ds = {0: 0, 1: 1, 2: 0.5}
    S = tau.MultiPhaseSolver(img, Ds, device='cpu')
    S.solve()
    assert np.around(S.tau, 4) == 1

def test_mphsolver_on_strip_of_ones_and_twos_and_threes():
    """Run solver on a strip of ones, 1/4 volume of total"""
    N = 20
    img = np.ones((N, N, N))
    x = 10
    img[:, 0:x, 0:x] = 2
    img[:, 0:x, x:N] = 3
    Ds = {0: 0, 1: 1, 2: 0.5, 3: 2}
    S = tau.MultiPhaseSolver(img, Ds, device='cpu')
    S.solve()
    assert np.around(S.tau, 4) == 1

def test_mphsolver_warns_on_missing_diffusivities():
    """Missing phase labels in diffusivities should be assumed isolating with warning."""
    N = 10
    img = np.zeros([N, N, N])
    img[:, :, : N // 2] = 1
    img[:, :, N // 2 :] = 2
    with pytest.warns(UserWarning, match="assuming these phases are isolating."):
        s = tau.MultiPhaseSolver(img, {1: 1.0}, device='cpu')
    assert s.Ds[2] == 0.0

def test_mphsolver_allows_zero_Ds_and_label_zero_conductive():
    """Label 0 may be conductive and other labels may explicitly be zero conductivity."""
    N = 20
    img = np.zeros((N, N, N))
    img[:, :2] = 1  # explicit non-conductive slab with cond=0
    s = tau.MultiPhaseSolver(img, {0: 1.0, 1: 0.0}, device='cpu')
    s.solve(iter_limit=1000)
    assert np.around(s.tau, 4) == 1

def test_mphsolver_rejects_negative_conductivity():
    """Negative conductivity should raise."""
    img = np.zeros([6, 6, 6])
    with pytest.raises(ValueError):
        tau.MultiPhaseSolver(img, {0: 1.0, 1: -0.1}, device='cpu')

def test_mphsolver_matches_solver_for_binary_case():
    """cond={1:1} should match Solver on a binary structure."""
    N = 20
    img = np.zeros((N, N+1, N+1))
    for i in range(N):
        img[i, i:i+2, i:i+2] = 1
    s_bin = tau.Solver(img, device='cpu')
    s_bin.solve(iter_limit=1000)
    s_mp = tau.MultiPhaseSolver(img, device='cpu')
    s_mp.solve(iter_limit=1000)
    assert np.isclose(float(np.asarray(s_bin.tau)[0]), float(np.asarray(s_mp.tau)[0]), atol=1e-3)

def test_mphsolver_supports_batched_inputs():
    """Batched multiphase solve should match per-sample results."""
    N = 16
    img_a = np.ones((N, N, N))
    img_b = np.ones((N, N, N))
    img_b[:, :, : N // 2] = 2
    imgs = np.stack([img_a, img_b], axis=0)

    Ds = {0: 0.0, 1: 1.0, 2: 0.5}
    s_batch = tau.MultiPhaseSolver(imgs, Ds, device='cpu')
    s_batch.solve(iter_limit=1000)

    s_a = tau.MultiPhaseSolver(img_a, Ds, device='cpu')
    s_a.solve(iter_limit=1000)
    s_b = tau.MultiPhaseSolver(img_b, Ds, device='cpu')
    s_b.solve(iter_limit=1000)

    assert np.allclose(np.asarray(s_batch.tau), np.array([s_a.tau[0], s_b.tau[0]]), atol=1e-3)

def test_periodic_mphsolver_on_uniform_block():
    """Periodic multiphase solver should return tau=1 on a uniform conductive block."""
    N = 20
    img = np.ones((N, N, N))
    s = tau.PeriodicMultiPhaseSolver(img, {1: 1.0}, device='cpu')
    s.solve(iter_limit=1000)
    assert np.around(s.tau, 4) == 1.0

def test_periodic_mphsolver_matches_periodic_solver_for_binary_case():
    """Binary limit of periodic multiphase should match PeriodicSolver."""
    N = 20
    img = np.zeros((N, N + 1, N + 1))
    for i in range(N):
        img[i, i:i + 2, i:i + 2] = 1

    s_bin = tau.PeriodicSolver(img, device='cpu')
    s_bin.solve(iter_limit=1000)

    s_mp = tau.PeriodicMultiPhaseSolver(img, {0: 0.0, 1: 1.0}, device='cpu')
    s_mp.solve(iter_limit=1000)

    assert np.isclose(float(np.asarray(s_bin.tau)[0]), float(np.asarray(s_mp.tau)[0]), atol=1e-3)


###  Testing the tau_e solver
def test_taue_deadend():
    """Run solver on a deadend strip of ones"""
    N = 100
    img = np.zeros((N, N))
    img[:75, 45:55] = 1
    esolver = tau.ImpedanceSolver(img, device='cpu')
    esolver.solve()
    assert np.around(esolver.tau, 3) == 0.593

def test_taue_throughpore():
    """Run taue solver on a strip of ones, 1/4 volume of total"""
    N = 100
    img = np.zeros((N, N))
    img[:, 45:55] = 1
    esolver = tau.ImpedanceSolver(img, device='cpu')
    esolver.solve()
    assert np.around(esolver.tau, 3) == 0.987
