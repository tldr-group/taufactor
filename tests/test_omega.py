"""Coarse spectral estimates and adaptive SOR relaxation."""

import numpy as np
import pytest
import torch
from scipy.linalg import eigh

import taufactor as tau
from taufactor.utils import create_stacked_blocks


@pytest.mark.parametrize('solver_cls, kwargs', [
    (tau.Solver, {}),
    (tau.AnisotropicSolver, {'spacing': (1, 2, 3)}),
    (tau.MultiPhaseSolver, {'diffusivities': {0: 0, 1: 1, 2: 3}}),
    (tau.PeriodicSolver, {}),
    (tau.PeriodicMultiPhaseSolver, {'diffusivities': {0: 0, 1: 1, 2: 3}}),
])
def test_coarse_guess_matches_projected_voxel_operator(solver_cls, kwargs):
    img = np.ones((4, 2, 2), dtype=int)
    img[1, 0, 0] = 0
    if issubclass(solver_cls, tau.MultiPhaseSolver):
        img[2:] = 2
    solver = solver_cls(img, omega='adaptive', device='cpu', **kwargs)

    # Assemble the physical graph independently, including boundary faces.
    coordinates = list(zip(*np.nonzero(img)))
    indices = {coordinate: i for i, coordinate in enumerate(coordinates)}
    operator = np.zeros((len(indices), len(indices)))
    projection = np.zeros((len(indices), img.shape[0]))
    for coordinate, i in indices.items():
        projection[i, coordinate[0]] = 1
        diffusivity = kwargs.get('diffusivities', {}).get(img[coordinate], 1)
        for axis, weight in enumerate((1, getattr(solver, 'Ky', 1), getattr(solver, 'Kz', 1))):
            for direction in (-1, 1):
                neighbour = list(coordinate)
                neighbour[axis] += direction
                if not 0 <= neighbour[axis] < img.shape[axis]:
                    if axis == 0:
                        operator[i, i] += 2 * diffusivity
                        continue
                    if solver.connectivity_periodic[axis]:
                        neighbour[axis] %= img.shape[axis]
                    else:
                        continue
                neighbour = tuple(neighbour)
                if neighbour not in indices:
                    continue
                other_diffusivity = kwargs.get('diffusivities', {}).get(img[neighbour], 1)
                conductance = weight * 2 * diffusivity * other_diffusivity / (diffusivity + other_diffusivity)
                operator[i, i] += conductance
                operator[i, indices[neighbour]] -= conductance

    mass = np.diag(np.diag(operator))
    gap = eigh(projection.T @ operator @ projection, projection.T @ mass @ projection,
               eigvals_only=True)[0]
    expected = 2 / (1 + np.sqrt(1 - (1 - gap)**2))
    assert solver.omega.item() == pytest.approx(expected)
    full_gap = eigh(operator, mass, eigvals_only=True)[0]
    optimum = 2 / (1 + np.sqrt(1 - (1 - full_gap)**2))
    assert 1 <= solver.omega.item() <= optimum + 1e-6


@pytest.mark.parametrize('omega', [None, 1.4])
def test_fixed_omega_preserves_legacy_solve(omega):
    img = create_stacked_blocks(12)
    solver = tau.Solver(img, omega=omega, device='cpu')
    expected = 2 - torch.pi / (1.3 * 12) if omega is None else omega
    reference = tau.Solver(img, omega=expected, device='cpu')
    solver.solve(verbose=False, conv_crit=1e-4)
    reference.solve(verbose=False, conv_crit=1e-4)
    assert solver.omega == expected
    assert solver.iter == reference.iter
    torch.testing.assert_close(solver.field, reference.field, rtol=0, atol=0)


def test_adaptive_estimator_damps_stable_rates_and_resets_samples():
    solver = tau.Solver(create_stacked_blocks(12), omega='adaptive', device='cpu')
    solver._omega_values.fill(1.2)
    solver.omega.fill_(1.2)
    rate = 0.95
    norm = np.array([1.0])
    solver._update_adaptive_omega(norm)
    solver._update_adaptive_omega(norm * rate**20)
    assert solver.omega.item() == pytest.approx(1.2)  # One window is not enough.
    solver._update_adaptive_omega(norm * rate**40)
    rho = (rate + 0.2) / (1.2 * np.sqrt(rate))
    target = 2 / (1 + np.sqrt(1 - rho**2))
    assert solver.omega.item() == pytest.approx((1.2 + target) / 2)
    assert np.isnan(solver._omega_norm).all()
    assert np.isnan(solver._omega_rate).all()


@pytest.mark.parametrize('norms', [
    [1.0, np.nan, 0.5],
    [1.0, 2.0, 4.0],
    [1.0, 0.95**20, 0.95**20 * 0.7**20],
    [1.0, 0.8**20, 0.8**40],  # At the cusp for omega=1.8.
])
def test_adaptive_estimator_skips_unusable_rates(norms):
    solver = tau.Solver(np.ones((4, 2, 2)), omega='adaptive', device='cpu')
    solver._omega_values.fill(1.8)
    solver.omega.fill_(1.8)
    for norm in norms:
        solver._update_adaptive_omega(np.array([norm]))
    assert solver.omega.item() == pytest.approx(1.8)


def test_adaptive_solver_converges_to_fixed_reference():
    img = create_stacked_blocks(24)
    solver = tau.Solver(img, omega='adaptive', device='cpu')
    reference = tau.Solver(img, omega=1.8, device='cpu')
    solver.solve(verbose=False, conv_crit=1e-5)
    reference.solve(verbose=False, conv_crit=1e-5)
    assert solver.converged and reference.converged
    assert len(solver.omega_history) > 1
    assert all(np.all((1 <= omega) & (omega < 2)) for _, omega in solver.omega_history)
    np.testing.assert_allclose(solver.tau, reference.tau, rtol=1e-5)
    torch.testing.assert_close(solver.field, reference.field, rtol=0, atol=1e-5)


@pytest.mark.parametrize('img', [np.ones((8, 2, 2)), np.zeros((8, 2, 2))])
def test_adaptive_handles_zero_updates_and_nonpercolating_domains(img):
    solver = tau.Solver(img, omega='adaptive', device='cpu')
    initial_omega = solver.omega.clone()
    solver.solve(verbose=False, iter_limit=200)
    assert torch.isfinite(solver.omega).all()
    torch.testing.assert_close(solver.omega, initial_omega, rtol=0, atol=0)
    assert solver.converged


def test_single_slice_coarse_guess():
    solver = tau.Solver(np.ones((1, 2, 2)), omega='adaptive', device='cpu')
    # Four boundary conductances / (four boundary + two lateral neighbours).
    assert solver.omega.item() == pytest.approx(2 / (1 + np.sqrt(1 - (1 / 3)**2)))


def test_sampler_combines_both_colours_without_changing_increment():
    solver = tau.Solver(np.ones((4, 2, 2)), omega='adaptive', device='cpu')
    increment = torch.ones_like(solver.factor)
    original = increment.clone()
    solver.iter = 39
    solver._sample_adaptive_omega_update(increment)
    assert np.isnan(solver._omega_norm).all()
    solver.iter = 40
    solver._sample_adaptive_omega_update(2 * increment)
    np.testing.assert_allclose(solver._omega_norm, [np.sqrt(5 * increment.numel())])
    torch.testing.assert_close(increment, original)


def test_adaptive_can_resume_between_colour_samples():
    img = create_stacked_blocks(24)
    uninterrupted = tau.Solver(img, omega='adaptive', device='cpu')
    resumed = tau.Solver(img, omega='adaptive', device='cpu')
    uninterrupted.solve(verbose=False, iter_limit=160, conv_crit=0)
    resumed.solve(verbose=False, iter_limit=39, conv_crit=0)
    resumed.solve(verbose=False, iter_limit=160, conv_crit=0)
    assert [i for i, _ in resumed.omega_history] == [i for i, _ in uninterrupted.omega_history]
    np.testing.assert_array_equal(np.stack([w for _, w in resumed.omega_history]),
                                  np.stack([w for _, w in uninterrupted.omega_history]))
    torch.testing.assert_close(resumed.field, uninterrupted.field, rtol=0, atol=0)


def test_coarse_guess_is_independent_for_each_batch_member():
    img = create_stacked_blocks(12)
    batch = tau.Solver(np.stack([img, np.ones_like(img), np.zeros_like(img)]),
                       omega='adaptive', device='cpu')
    guesses = [tau.Solver(volume, omega='adaptive', device='cpu').omega.item()
               for volume in (img, np.ones_like(img))]
    assert batch.omega.shape == (3, 1, 1, 1)
    np.testing.assert_array_equal(batch.omega.flatten().numpy(), guesses + [1])
    assert guesses[0] != guesses[1]


@pytest.mark.parametrize('bad_norm', [np.nan])
def test_invalid_channel_does_not_block_other_omega_updates(bad_norm):
    img = create_stacked_blocks(12)
    solver = tau.Solver(np.stack([img, img, np.zeros_like(img)]), omega='adaptive', device='cpu')
    solver._omega_values.fill(1.2)
    solver.omega.fill_(1.2)
    solver._update_adaptive_omega(np.ones(3))
    solver._update_adaptive_omega(np.array([0.95**20, bad_norm, 0.95**20]))
    solver._update_adaptive_omega(np.array([0.95**40, bad_norm**2, 0.95**40]))
    assert solver.omega[0].item() > 1.2
    np.testing.assert_allclose(solver.omega[1:].flatten().numpy(), [1.2, 1.2])


def test_updating_one_channel_preserves_other_channels_measurement_windows():
    img = create_stacked_blocks(12)
    solver = tau.Solver(np.stack([img, img]), omega='adaptive', device='cpu')
    solver._omega_values.fill(1.2)
    solver.omega.fill_(1.2)
    initial_history = solver.omega_history[0][1].copy()
    solver._update_adaptive_omega(np.array([1, np.nan]))
    solver._update_adaptive_omega(np.array([0.95**20, 1]))
    solver._update_adaptive_omega(np.array([0.95**40, 0.95**20]))
    first_omega = solver.omega[0].item()
    assert first_omega > 1.2
    assert solver.omega[1].item() == pytest.approx(1.2)
    assert np.isnan(solver._omega_norm[0])
    assert np.isfinite(solver._omega_rate[1])
    first_update = solver.omega_history[-1][1].copy()
    solver._update_adaptive_omega(np.array([0.1, 0.95**40]))
    assert solver.omega[0].item() == first_omega
    assert solver.omega[1].item() > 1.2
    np.testing.assert_array_equal(solver.omega_history[0][1], initial_history)
    np.testing.assert_array_equal(solver.omega_history[-2][1], first_update)


@pytest.mark.parametrize('solver_cls, kwargs, device', [
    (tau.Solver, {}, 'cpu'),
    (tau.AnisotropicSolver, {'spacing': (1, 2, 3)}, 'cpu'),
    (tau.MultiPhaseSolver, {'diffusivities': {0: 0, 1: 1}}, 'cpu'),
    (tau.PeriodicSolver, {}, 'cpu'),
    (tau.PeriodicMultiPhaseSolver, {'diffusivities': {0: 0, 1: 1}}, 'cpu'),
    (tau.Solver, {}, 'cuda'),
])
def test_batched_adaptive_iterations_match_individual_runs(device, solver_cls, kwargs):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    volumes = [create_stacked_blocks(24), create_stacked_blocks(24, features=2)]
    batch = solver_cls(np.stack(volumes + [np.zeros_like(volumes[0])]),
                       omega='adaptive', device=device, **kwargs)
    batch.solve(verbose=False, iter_limit=240, conv_crit=0)
    assert batch.omega.device.type == device
    for b, img in enumerate(volumes):
        individual = solver_cls(img, omega='adaptive', device=device, **kwargs)
        individual.solve(verbose=False, iter_limit=240, conv_crit=0)
        torch.testing.assert_close(batch.field[b], individual.field[0], rtol=0, atol=1e-6)
        torch.testing.assert_close(batch.omega[b], individual.omega[0], rtol=0, atol=1e-6)
        assert all(np.all((1 <= omega) & (omega < 2)) for _, omega in batch.omega_history)
        # Include samples where only another channel changed.
        for step, values in batch.omega_history:
            expected = [w[0] for i, w in individual.omega_history if i <= step][-1]
            assert values[b] == pytest.approx(expected, abs=1e-6)
    assert batch.omega[2].item() == 1


def test_adaptive_omega_broadcasts_only_within_its_batch_channel():
    solver = tau.Solver(np.ones((2, 4, 2, 2)), omega='adaptive', device='cpu')
    solver.omega[0].fill_(1.1)
    solver.omega[1].fill_(1.9)
    increment = torch.ones_like(solver.factor)
    solver._apply_chequerboard(increment)
    torch.testing.assert_close(increment[0], solver.cb.to(torch.float32) * 1.1)
    torch.testing.assert_close(increment[1], solver.cb.to(torch.float32) * 1.9)


@pytest.mark.parametrize('solver_cls', [
    tau.ElectrodeSolver, tau.ImpedanceSolver,
    tau.PeriodicElectrodeSolver, tau.PeriodicImpedanceSolver,
])
def test_non_transport_solvers_reject_adaptive_omega(solver_cls):
    with pytest.raises(TypeError, match='omega must be None or a number'):
        solver_cls(np.ones((4, 2, 2)), omega='adaptive', device='cpu')


@pytest.mark.parametrize('omega', ['online', 'auto'])
def test_unknown_omega_mode_is_rejected(omega):
    with pytest.raises(ValueError, match="omega must be None, a number, or 'adaptive'"):
        tau.Solver(np.ones((4, 2, 2)), omega=omega, device='cpu')
