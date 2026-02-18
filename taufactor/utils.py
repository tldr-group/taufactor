import numpy as np
import matplotlib.pyplot as plt


def flux_direction(im, outpath=None):
    """
    Plots the flux direction of the image and provides code for transposing the image to change the flux direction
    :param im: segmented input image with n phases
    :return: None
    """
    im = np.asarray(im)
    if im.ndim not in (2, 3):
        raise ValueError("im must be 2D or 3D")

    if im.ndim == 3:
        orders = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
        titles = [
            "img",
            "np.transpose(img, (1,2,0))",
            "np.transpose(img, (2,0,1))",
        ]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
        for ax, order, title in zip(axes, orders, titles):
            vol = np.transpose(im, order)
            sl = vol[vol.shape[0] // 2]  # middle slice after permutation
            ax.imshow(sl, cmap="gray")
            ax.set_title(title, fontfamily="monospace")
            ax.axis("off")
        fig.suptitle(
            "Direction of flux is into the page.\n"
            "Use the shown np.transpose to change the flux direction.",
            fontsize=12,
        )
    else:  # 2D
        orders = [(0, 1), (1, 0)]
        titles = ["img", "np.transpose(img, (1,0))"]
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
        for ax, order, title in zip(axes, orders, titles):
            arr = np.transpose(im, order)
            ax.imshow(arr, cmap="gray")
            ax.set_title(title, fontfamily="monospace")
            ax.axis("off")
            # optional: annotate 'Flux' at the side
            h, w = arr.shape
            ax.annotate(
                "Flux", xy=(w * 1.02, h / 2), xytext=(w * 1.02, h / 2),
                textcoords="data", rotation=90, va="center", ha="left",
                bbox=dict(boxstyle="rarrow,pad=0.3", fc="white", ec="black", lw=1),
                fontsize=12,
            )
        fig.suptitle(
            "Direction of flux is down.\n"
            "The transformations required to change direction of flux are shown.",
            fontsize=12,
        )
    if outpath:
        fig.savefig(outpath, bbox_inches="tight")
    else:
        plt.show()
    return fig, axes

def compute_impedance(R, C, freq):
    """
    Compute impedance given resistor and capacitor arrays.
    This is under the assumption of an excitation with amplitude 1 at the left boundary
    and zero flux at the right boundary (closed Warburg)
    """
    Z = np.full_like(freq, 1e50, dtype=complex)
    for R_i, C_i in zip(R[::-1], C[::-1]):
        Z = R_i + 1.0 / (1j * freq * C_i + 1.0 / Z)
    return Z

def compute_impedance_batched(R, C, freq):
    """
    R, C: arrays of shape (bs, Nx)
    freq: array of shape (F,) or (bs, F)
    Returns: Z of shape (bs, F)
    """
    if R.shape != C.shape:
        raise ValueError(f"R and C must have same shape; got {R.shape} vs {C.shape}")
    bs, Nx = R.shape

    # Broadcast frequencies to (bs, F)
    if freq.ndim == 1:
        w = freq[None, :].repeat(bs, axis=0)
    elif freq.ndim == 2 and freq.shape[0] == bs:
        w = freq
    else:
        raise ValueError("freq must be (F,) or (bs, F)")

    Z = np.full_like(w, 1e50, dtype=complex)
    for k in range(Nx - 1, -1, -1):
        Rk = R[:, k][:, None]  # (bs, 1)
        Ck = C[:, k][:, None]  # (bs, 1)
        Z = Rk + 1.0 / (1j * w * Ck + 1.0 / Z)
    return Z

def add_voxel_sphere(array, center_x, center_y, center_z, radius):
    """
    Create a voxelized representation of a sphere in 3D array based on
    given midpoint and radius in terms of pixel resolution.
    """
    nx, ny, nz = array.shape
    x, y, z = np.ogrid[:nx, :ny, :nz]

    distance_squared = (x - center_x + 0.5)**2 + (y - center_y + 0.5)**2 + (z - center_z + 0.5)**2
    mask = distance_squared <= radius**2
    array[mask] = 1

def create_fcc_cube(pixels, overlap=0.0):
    """
    Create a voxelized FCC unit cell structure in a cube with given
    pixel resolution and overlap of the spheres.

    Parameters:
        pixels (int): Cube/Array side length given in pixels.
        overlap (float): Overlap of neighbouring spheres given in percent.
                     1 corresponds to radius = distance between midpoints.

    Returns:
        numpy.ndarray: 3D array where values of 1 represent the FCC structure.
    """
    # Initialize a 3D numpy array filled with zeros
    cube = np.zeros((pixels, pixels, pixels), dtype=int)

    # Calculate the center and radius
    center = 0.5*pixels
    radius = 0.25*np.sqrt(2)*pixels/(1-0.5*overlap)

    # Add half-spheres centered on each face of the cube
    # We have 6 centers, a list of three center positions with a pos and neg sign
    for axis in range(3):
        for sign in [-1, 1]:
            center_pos = [center] * 3
            center_pos[axis] = center + sign * (center)
            add_voxel_sphere(cube, *center_pos, radius)

    # Add quarter-spheres at each corner of the cube
    for corner in [(0, 0, 0), (0, 0, pixels), (0, pixels, 0), (0, pixels, pixels),
                   (pixels, 0, 0), (pixels, 0, pixels), (pixels, pixels, 0), (pixels, pixels, pixels)]:
        add_voxel_sphere(cube, *corner, radius)

    return cube

def theoretical_fcc_metrics(a, overlap):
    """
    Return theoretical metrics of FCC unit cell structure.
    Notation consistent with https://en.wikipedia.org/wiki/Spherical_cap
    """
    if overlap < (1-np.cos(np.pi/6))*2:
        radius = 0.25*np.sqrt(2)*a/(1-0.5*overlap)
        h = 0.5*radius*overlap
        cap_radius = np.sqrt(2*radius*h - h*h)
        cap_volume = np.pi/3*h*h*(3*radius-h)
        cap_area = 2*np.pi*radius*h

        volume = 4*4/3*np.pi*radius**3 - 48*cap_volume
        volume_fraction = volume/(a**3)

        surface = 4*4*np.pi*radius**2 - 48*cap_area
        specific_surface = surface/(a**3)
    else:
        raise ValueError("Overlap must be smaller than 26.8%!")

    return volume_fraction, specific_surface, cap_radius

def create_stacked_blocks(Nx, features=1):
    """Create the shifted block/checkerboard structure used for solver benchmarks."""
    if Nx % (2*features) != 0:
        raise ValueError(f"Nx must be a multiple of 2*features; got Nx={Nx} and features={features}")
    feature_size = Nx // (2*features)
    x, y, z = np.ogrid[:Nx, :Nx, :Nx]
    shift = ((x // feature_size) % 2 ) * Nx // 4
    pattern = (
        ((x // feature_size) + ((y + shift) // feature_size) + ((z + shift) // feature_size)) % 2
    )
    return pattern.astype(int)

def create_2d_diagonals(Nx, features=1):
    """Create a 2D diagonal pattern extruded in z."""
    if Nx % (2*features) != 0:
        raise ValueError(f"Nx must be a multiple of 2*features; got Nx={Nx} and features={features}")
    feature_size = Nx // (2*features)
    x, y, z = np.ogrid[:Nx, :Nx, :Nx]
    pattern = (((x + y) // feature_size) % 2) + z - z
    return pattern.astype(int)

def create_2d_zigzag(Nx, features=1):
    """Create mirrored 2D diagonal channels (zigzag) extruded in z."""
    pattern = create_2d_diagonals(Nx, features)
    half = Nx // 2
    pattern[half:] = pattern[:half][::-1]
    return pattern

def create_3d_diagonals(Nx, features=1):
    """Create a 3D diagonal pattern."""
    if Nx % (2*features) != 0:
        raise ValueError(f"Nx must be a multiple of 2*features; got Nx={Nx} and features={features}")
    feature_size = Nx // (2*features)
    x, y, z = np.ogrid[:Nx, :Nx, :Nx]
    pattern = (((x + y + z) // feature_size) % 2)
    return pattern.astype(int)

def extract_inner_features(labelled_array, verbose=True):
    initial_labels = np.unique(labelled_array).size
    if initial_labels < 3:
        raise ValueError("Input array should be labelled array with more than 3 phases!")

    # Find all features which are in contact with domain boundary
    boundary_labels = np.unique(labelled_array[0,:,:])
    boundary_labels = np.concatenate((boundary_labels, np.unique(labelled_array[-1,:,:])))
    boundary_labels = np.concatenate((boundary_labels, np.unique(labelled_array[:,0,:])))
    boundary_labels = np.concatenate((boundary_labels, np.unique(labelled_array[:,-1,:])))
    boundary_labels = np.concatenate((boundary_labels, np.unique(labelled_array[:,:,0])))
    boundary_labels = np.concatenate((boundary_labels, np.unique(labelled_array[:,:,-1])))
    boundary_labels = np.unique(boundary_labels)

    mask_boundary_labels = np.isin(labelled_array, boundary_labels)
    labelled_array[mask_boundary_labels] = 0
    if verbose:
        print(f"{np.unique(labelled_array).size} of initial {initial_labels} labels remaining.")
    return labelled_array

def relabel_random_order(array):
    remaining_labels = np.unique(array)
    new_labels = np.arange(len(remaining_labels))
    # Zero should be kept where it is
    np.random.shuffle(new_labels[1:])

    # Create a mapping from old labels to new shuffled labels
    label_mapping = dict(zip(remaining_labels, new_labels))
    relabel_function = np.vectorize(lambda x: label_mapping[x])

    return relabel_function(array)
