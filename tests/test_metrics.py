"""Tests for `taufactor` package."""

from taufactor.metrics import (
    estimate_3d_psd_saltykov,
    particle_size_distribution,
    particle_size_distribution_2d,
    specific_surface_area,
    triple_phase_boundary,
    volume_fraction,
    remove_boundary_features,
    relabel_random_order,
    relabel_sequential,
    split_lumped_labels,
)
import numpy as np
import pytest


# Volume fraction
def test_volume_fraction_on_uniform_block():
    """Run volume fraction on uniform block"""
    N = 20
    img = np.ones([N, N, N]).reshape(1, N, N, N)
    vf = volume_fraction(img)['1']

    assert np.around(vf, decimals=5) == 1.0


def test_volume_fraction_on_empty_block():
    """Run volume fraction on empty block"""
    N = 20
    img = np.zeros([N, N, N]).reshape(1, N, N, N)
    vf = volume_fraction(img)['0']

    assert np.around(vf, decimals=5) == 1.0


def test_volume_fraction_on_checkerboard():
    """Run volume fraction on checkerboard block"""
    size = 20
    cb = np.zeros([size, size, size])
    a, b, c = np.meshgrid(range(size), range(size), range(size), indexing='ij')
    cb[(a + b + c) % 2 == 0] = 1
    vf = volume_fraction(cb, phases={'zeros': 0, 'ones': 1})

    assert (vf['zeros'], vf['ones']) == (0.5, 0.5)


def test_volume_fraction_on_strip_of_ones():
    """Run volume fraction on strip of ones"""
    N = 20
    img = np.zeros([N, N, N])
    t = 10
    img[:, 0:t, 0:t] = 1
    vf = volume_fraction(img, phases={'zeros': 0, 'ones': 1})

    assert (vf['zeros'], vf['ones']) == (0.75, 0.25)


def test_volume_fraction_on_multi_cubes():
    """Run surface area on multiple cubes"""
    N = 20
    img = np.zeros([N, N, N])
    img[0:10, 0:10, 0:5] = 1
    img[5:-5, 5:-5, 5:-5] = 2
    img[0:10, 0:10, 15:] = 1
    img[10:, 10:, 15:] = 3
    vf = volume_fraction(img)
    sum = vf['0'] + vf['1'] + vf['2'] + vf['3']

    assert (vf['1'], vf['2'], vf['3'], sum) == (0.125, 0.125, 0.0625, 1.0)

# Surface area
def test_surface_area_on_uniform_block():
    """Run surface area on uniform block"""
    N = 20
    img = np.ones([N, N, N])
    sa_f = specific_surface_area(img, method='face_counting')['1']
    sa_g = specific_surface_area(img, method='gradient')['1']
    # Marchin cubes will not work unless there are non-uniform values
    # sa_m = specific_surface_area(img, method='marching_cubes')

    assert (sa_f, sa_g) == (0, 0)


def test_surface_area_on_floating_cube():
    """Run surface area on floating cube"""
    N = 20
    img = np.zeros([N, N, N])
    x1, x2 = 5, 15
    img[x1:x2, x1:x2, x1:x2] = 1
    sa_f = specific_surface_area(img, method='face_counting')['1']
    sa_m = specific_surface_area(img, method='marching_cubes', smoothing=False, device='cpu')['1']
    sa_g = specific_surface_area(img, method='gradient', smoothing=False)['1']

    # All six sides should be taken into account
    # True value is 0.075
    assert (np.around(sa_f, 5), np.around(sa_m, 5), np.around(sa_g, 5)) == (0.075, 0.07051, 0.07085)


def test_surface_area_on_corner_cube():
    """Run surface area on corner cube"""
    N = 20
    img = np.zeros([N, N, N])
    img[0:10, 0:10, 0:10] = 1
    sa_f = specific_surface_area(img, method='face_counting')['1']

    # Only three inner sides should be taken into account
    # True value is 0.0375
    assert np.around(sa_f, 5) == 0.0375


def test_surface_area_on_sphere():
    """Run surface area on sphere"""
    N = 20
    img = np.zeros([N, N, N])
    radius = N*0.5-3
    x, y, z = np.ogrid[:N, :N, :N]
    distance_squared = (x - N/2 + 0.5)**2 + (y - N/2 + 0.5)**2 + (z - N/2 + 0.5)**2
    mask = distance_squared <= radius**2
    img[mask] = 1
    a_theo = 4*np.pi*radius**2/img.size
    sa_f = np.abs(specific_surface_area(img, method='face_counting')['1']-a_theo)/a_theo*100
    sa_m = np.abs(specific_surface_area(img, method='marching_cubes', device='cpu')['1']-a_theo)/a_theo*100
    sa_g = np.abs(specific_surface_area(img, method='gradient')['1']-a_theo)/a_theo*100

    # Relative errors should be
    # - face_counting: 52.01 %,
    # - marching_cubes: 0.01 %,
    # - gradient:       1.25 %
    assert (np.around(sa_f, 2), np.around(sa_m, 2), np.around(sa_g, 2)) == (52.01, 0.01, 1.25)


def test_surface_area_on_multi_cubes():
    """Run surface area on multiple cubes"""
    N = 20
    img = np.zeros([N, N, N])
    img[0:10, 0:10, 0:5] = 1
    img[5:-5, 5:-5, 5:-5] = 2
    img[0:10, 0:10, 15:] = 1
    img[10:, 10:, 15:] = 3
    sa_f = specific_surface_area(img, method='face_counting')
    sa_m = specific_surface_area(img, method='marching_cubes', smoothing=False, device='cpu')
    sa_g = specific_surface_area(img, method='gradient', smoothing=False)
    results = []
    for sa in [sa_f, sa_m, sa_g]:
        for phase in ['1', '2', '3']:
            results.append(np.around(sa[phase], 4))
    # All six sides should be taken into account
    # True value is 0.075
    reference = [0.0500, 0.0750, 0.0250, \
                 0.0422, 0.0705, 0.0211, \
                 0.0482, 0.0709, 0.0241]
    assert results == reference


# Triple phase boundary
def test_tpb_2d():
    """Run tpb on 3x3"""
    N = 3
    img = np.zeros([N, N])
    img[0] = 1
    img[:, 0] = 2
    tpb = triple_phase_boundary(img)
    assert tpb == 0.25


def test_tpb_3d():
    """Run tpb on 2x2x2"""
    N = 2
    img = np.zeros([N, N, N])
    img[0, 0, 0] = 1
    img[1, 1, 1] = 1
    img[0, 1, 1] = 2
    img[1, 0, 0] = 2
    tpb = triple_phase_boundary(img)
    assert tpb == 1


def test_tpb_3d_corners():
    """Run tpb on 2x2x2 corners"""
    N = 2
    img = np.zeros([N, N, N])
    img[0] = 1
    img[:, 0] = 2
    tpb = triple_phase_boundary(img)
    assert tpb == 1/3


# Relabelling and remove boundary labels
def test_relabel_random_order_preserves_partitions_and_zero_label():
    array = np.array(
        [
            [0, 7, 7, 2],
            [9, 0, 2, 9],
            [2, 7, 9, 0],
        ]
    )

    np.random.seed(0)
    relabelled = relabel_random_order(array)

    assert relabelled.shape == array.shape
    assert np.array_equal(relabelled[array == 0], np.zeros(np.count_nonzero(array == 0), dtype=relabelled.dtype))
    assert np.array_equal(np.unique(relabelled), np.arange(np.unique(array).size))

    for label in np.unique(array):
        mapped = np.unique(relabelled[array == label])
        assert mapped.size == 1


def test_remove_boundary_features_returns_new_array_without_mutating_input():
    labelled = np.zeros((4, 4, 4), dtype=int)
    labelled[0, 1, 1] = 1
    labelled[1:3, 1:3, 1:3] = 2
    original = labelled.copy()

    inner = remove_boundary_features(labelled, verbose=False)

    assert np.array_equal(labelled, original)
    assert inner is not labelled
    assert 1 not in np.unique(inner)
    assert 2 in np.unique(inner)


def test_remove_boundary_features_rejects_binary_masks_with_clear_message():
    labelled = np.zeros((3, 3, 3), dtype=int)
    labelled[1, 1, 1] = 1

    with pytest.raises(ValueError, match="binary mask|connected-component labeling"):
        remove_boundary_features(labelled, verbose=False)


def test_remove_boundary_features_ignores_periodic_boundaries():
    labelled = np.zeros((4, 4, 4), dtype=int)
    labelled[0, 1:3, 1:3] = 1
    labelled[-1, 1:3, 1:3] = 1
    labelled[1:3, 1:3, 1:3] = 2

    inner = remove_boundary_features(labelled, verbose=False, periodic=(True, False, False))

    assert set(np.unique(inner)) == {0, 1, 2}


def test_relabel_sequential_compacts_labels_without_python_loop_per_label():
    labelled = np.array([[0, 10, 10], [7, 0, 3]])

    relabelled = relabel_sequential(labelled)

    assert np.array_equal(relabelled, np.array([[0, 3, 3], [2, 0, 1]]))


# Particle size distribution
def test_particle_size_distribution_removes_boundary_labels():
    img = np.zeros((8, 8, 8), dtype=int)
    img[0, 1:3, 1:3] = 5
    img[2:5, 2:5, 2:5] = 9
    img[3, 3, 3] = 0
    img[5:7, 5:7, 5:7] = 12

    psd = particle_size_distribution(img, warn=False, return_field=True)

    assert psd["n_particles_initial"] == 3
    assert psd["n_particles_kept"] == 2
    assert np.array_equal(psd["particle_labels"], np.array([1, 2]))
    assert np.array_equal(psd["volumes"], np.array([26.0, 8.0]))
    assert np.allclose(psd["equivalent_diameters"], (6.0 * psd["volumes"] / np.pi) ** (1.0 / 3.0))
    assert psd["removed_label_fraction"] == 1 / 3
    assert np.isclose(psd["removed_mass_fraction"], 4 / 38)
    assert psd["labels"][3, 3, 3] == 0


def test_particle_size_distribution_ignores_periodic_boundary_axes():
    img = np.zeros((6, 6, 6), dtype=int)
    img[0, 2:4, 2:4] = 4
    img[-1, 2:4, 2:4] = 4
    img[2:4, 2:4, 2:4] = 7

    non_periodic = particle_size_distribution(img, periodic=(False, False, False), warn=False)
    periodic_x = particle_size_distribution(img, periodic=(True, False, False), warn=False)

    assert non_periodic["n_particles_kept"] == 1
    assert periodic_x["n_particles_kept"] == 2
    assert np.array_equal(periodic_x["volumes"], np.array([8.0, 8.0]))


def test_particle_size_distribution_warns_when_boundary_removal_discards_most_particles_and_mass():
    img = np.zeros((6, 6, 6), dtype=int)
    img[0, 1:3, 1:3] = 1
    img[-1, 1:3, 1:3] = 2
    img[1:5, 0, 1:3] = 3
    img[2:4, 2:4, 2:4] = 4

    with pytest.warns(UserWarning, match="more than half"):
        particle_size_distribution(img)


def test_particle_size_distribution_scales_with_spacing_and_returns_sphericity():
    img = np.zeros((10, 10, 10), dtype=int)
    img[3:7, 3:7, 3:7] = 11

    psd = particle_size_distribution(
        img,
        spacing=(2, 1, 1),
        compute_sphericity=True,
        warn=False,
    )

    assert np.array_equal(psd["particle_labels"], np.array([1]))
    assert np.array_equal(psd["volumes"], np.array([128.0]))
    assert np.allclose(psd["equivalent_diameters"], np.array([(6.0 * 128.0 / np.pi) ** (1.0 / 3.0)]))
    assert psd["surface_areas"].shape == (1,)
    assert psd["surface_areas"][0] > 0
    assert psd["sphericity"].shape == (1,)
    assert np.isfinite(psd["sphericity"][0])


def test_particle_size_distribution_2d_removes_edge_particles_and_computes_circularity():
    img = np.zeros((16, 16), dtype=int)
    yy, xx = np.ogrid[:16, :16]
    disk = (xx - 8) ** 2 + (yy - 8) ** 2 <= 9
    img[disk] = 4
    img[0:3, 0:3] = 7

    psd = particle_size_distribution_2d(img, warn=False, return_field=True)

    expected_area = float(np.count_nonzero(disk))
    assert psd["n_particles_initial"] == 2
    assert psd["n_particles_kept"] == 1
    assert np.array_equal(psd["particle_labels"], np.array([1]))
    assert np.allclose(psd["areas"], np.array([expected_area]))
    assert np.allclose(psd["equivalent_diameters"], np.array([np.sqrt(4.0 * expected_area / np.pi)]))
    assert psd["circularity"].shape == (1,)
    assert 0.7 < psd["circularity"][0] <= 1.1
    assert 7 not in np.unique(psd["labels"])


def test_estimate_3d_psd_saltykov_recovers_peak_bin_for_monodisperse_spheres():
    true_diameter = 10.0
    radius = true_diameter / 2.0
    z = np.linspace(0.0, radius * 0.999, 2000)
    apparent_diameters = 2.0 * np.sqrt(radius**2 - z**2)

    estimate = estimate_3d_psd_saltykov(
        apparent_diameters,
        bin_edges=np.linspace(0.0, true_diameter, 6),
    )

    peak_bin = int(np.argmax(estimate["counts_3d"]))
    assert peak_bin == len(estimate["bin_centers"]) - 1
    assert np.isclose(estimate["fractions_3d"].sum(), 1.0)


def test_split_lumped_labels_splits_components_into_new_labels():
    labelled = np.zeros((8, 8, 8), dtype=int)
    labelled[1:3, 1:3, 1:3] = 5
    labelled[5:7, 5:7, 5:7] = 5
    labelled[2:5, 4:6, 2:5] = 9
    original = labelled.copy()

    fixed, report = split_lumped_labels(labelled, verbose=False, return_report=True)

    assert np.array_equal(labelled, original)
    assert fixed is not labelled
    assert report["n_labels"] == 2
    assert report["n_split_labels"] == 1
    assert report["n_new_labels"] == 1
    assert report["has_splits"] is True
    assert report["split_labels"] == {5: [10]}
    assert set(np.unique(fixed)) == {0, 5, 9, 10}
    assert split_lumped_labels(fixed, verbose=False, return_report=True)[1]["has_splits"] is False


def test_split_lumped_labels_returns_clean_report_for_connected_labels():
    labelled = np.zeros((8, 8, 8), dtype=int)
    labelled[1:4, 1:4, 1:4] = 5
    labelled[4:7, 4:7, 4:7] = 9

    fixed, report = split_lumped_labels(labelled, verbose=False, return_report=True)

    assert np.array_equal(fixed, labelled)
    assert report["n_labels"] == 2
    assert report["n_split_labels"] == 0
    assert report["n_new_labels"] == 0
    assert report["has_splits"] is False
    assert report["split_labels"] == {}
