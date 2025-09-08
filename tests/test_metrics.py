"""Tests for `taufactor` package."""

from taufactor.metrics import volume_fraction, specific_surface_area, triple_phase_boundary
import numpy as np


# Volume fraction
def test_volume_fraction_on_uniform_block():
    """Run volume fraction on uniform block"""
    l = 20
    img = np.ones([l, l, l]).reshape(1, l, l, l)
    vf = volume_fraction(img)['1']

    assert np.around(vf, decimals=5) == 1.0


def test_volume_fraction_on_empty_block():
    """Run volume fraction on empty block"""
    l = 20
    img = np.zeros([l, l, l]).reshape(1, l, l, l)
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
    l = 20
    img = np.zeros([l, l, l])
    t = 10
    img[:, 0:t, 0:t] = 1
    vf = volume_fraction(img, phases={'zeros': 0, 'ones': 1})

    assert (vf['zeros'], vf['ones']) == (0.75, 0.25)


def test_volume_fraction_on_multi_cubes():
    """Run surface area on multiple cubes"""
    l = 20
    img = np.zeros([l, l, l])
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
    l = 20
    img = np.ones([l, l, l])
    sa_f = specific_surface_area(img, method='face_counting')['1']
    sa_g = specific_surface_area(img, method='gradient')['1']
    # Marchin cubes will not work unless there are non-uniform values
    # sa_m = specific_surface_area(img, method='marching_cubes')

    assert (sa_f, sa_g) == (0, 0)


def test_surface_area_on_floating_cube():
    """Run surface area on floating cube"""
    l = 20
    img = np.zeros([l, l, l])
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
    l = 20
    img = np.zeros([l, l, l])
    img[0:10, 0:10, 0:10] = 1
    sa_f = specific_surface_area(img, method='face_counting')['1']

    # Only three inner sides should be taken into account
    # True value is 0.0375
    assert np.around(sa_f, 5) == 0.0375


def test_surface_area_on_sphere():
    """Run surface area on sphere"""
    l = 20
    img = np.zeros([l, l, l])
    radius = l*0.5-3
    x, y, z = np.ogrid[:l, :l, :l]
    distance_squared = (x - l/2 + 0.5)**2 + (y - l/2 + 0.5)**2 + (z - l/2 + 0.5)**2
    mask = distance_squared <= radius**2
    img[mask] = 1
    a_theo = 4*np.pi*radius**2/img.size
    sa_f = np.abs(specific_surface_area(img, method='face_counting')['1']-a_theo)/a_theo*100
    sa_m = np.abs(specific_surface_area(img, method='marching_cubes', device='cpu')['1']-a_theo)/a_theo*100
    sa_g = np.abs(specific_surface_area(img, method='gradient')['1']-a_theo)/a_theo*100

    # Relative errors should be
    # - face_counting: 52.01 %,
    # - marching_cubes: 0.82 %,
    # - gradient:       0.95 %
    assert (np.around(sa_f, 2), np.around(sa_m, 2), np.around(sa_g, 2)) == (52.01, 0.82, 0.95)


def test_surface_area_on_multi_cubes():
    """Run surface area on multiple cubes"""
    l = 20
    img = np.zeros([l, l, l])
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


def test_tpb_2d():
    """Run tpb on 3x3"""
    l = 3
    img = np.zeros([l, l])
    img[0] = 1
    img[:, 0] = 2
    tpb = triple_phase_boundary(img)
    assert tpb == 0.25


def test_tpb_3d_corners():
    """Run tpb on 2x2"""

    l = 2
    img = np.zeros([l, l, l])
    img[0, 0, 0] = 1
    img[1, 1, 1] = 1
    img[0, 1, 1] = 2
    img[1, 0, 0] = 2
    tpb = triple_phase_boundary(img)
    assert tpb == 1


def test_tpb_3d_corners():
    """Run tpb on 2x2 corners"""

    l = 2
    img = np.zeros([l, l, l])
    img[0] = 1
    img[:, 0] = 2
    tpb = triple_phase_boundary(img)
    assert tpb == 1/3