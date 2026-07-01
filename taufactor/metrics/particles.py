import numpy as np
import warnings

from skimage import measure
from scipy.ndimage import find_objects, generate_binary_structure, label

from .surfaces import specific_surface_area


def relabel_sequential(array):
    """Relabel an array with consecutive integer labels."""
    remaining_labels, inverse = np.unique(array, return_inverse=True)
    new_labels = np.arange(remaining_labels.size, dtype=np.intp)
    return new_labels[inverse].reshape(array.shape)


def relabel_random_order(array):
    """Relabel an array with shuffled consecutive integer labels.

    Args:
        array: Input labeled array.

    Returns:
        Array with labels remapped to consecutive integers in random order.
        If label ``0`` is present, it remains ``0``.
    """
    remaining_labels, inverse = np.unique(array, return_inverse=True)
    new_labels = np.arange(remaining_labels.size, dtype=np.intp)
    # Zero should be kept where it is
    np.random.shuffle(new_labels[1:])
    return new_labels[inverse].reshape(array.shape)


def remove_boundary_features(labelled_array, verbose=True, periodic=(False, False, False)):
    """Remove labeled features that touch the domain boundary.

    Args:
        labelled_array: 3D array of connected-component labels, where ``0`` is
            background and each feature has its own positive integer label.
        verbose: If ``True``, print how many labels remain after filtering.
        periodic: Periodicity flags for ``(x, y, z)``. Periodic axes are ignored
            when checking boundary contact.

    Returns:
        A new array where all boundary-touching labels have been set to ``0``.

    Raises:
        ValueError: If the input does not look like a labeled array with
            background plus at least two feature labels.
    """
    initial_labels = np.unique(labelled_array).size
    if initial_labels < 3:
        raise ValueError(
            "Input must be a labeled array with background label 0 and at least "
            "two feature labels. A binary mask such as {0, 1} is not sufficient; "
            "run connected-component labeling first."
        )

    boundary_labels = _find_boundary_labels(labelled_array, tuple(bool(p) for p in periodic))

    inner_features = labelled_array.copy()
    mask_boundary_labels = np.isin(inner_features, list(boundary_labels))
    inner_features[mask_boundary_labels] = 0
    if verbose:
        print(f"{np.unique(inner_features).size-1} of initial {initial_labels-1} labels remaining.")
    return inner_features


def split_lumped_labels(labelled_array, connectivity=1, background=0, verbose=True, return_report=False):
    """Split groups of particles with one lumped label into new labels.

    Args:
        labelled_array: 2D or 3D labeled array.
        connectivity: Connectivity passed to ``scipy.ndimage.label``.
        background: Background label to ignore. Defaults to ``0``.
        verbose: If ``True``, print a short summary.
        return_report: If ``True``, also return a summary of the applied splits.

    Returns:
        numpy.ndarray | tuple[numpy.ndarray, dict]: A relabeled copy of the
        input array, optionally with a report describing the performed splits.
    """
    fixed = np.asarray(labelled_array).copy()
    if fixed.ndim not in (2, 3):
        raise ValueError(f"Expected a 2D or 3D labeled array, got {fixed.ndim}D input.")
    if connectivity < 1 or connectivity > fixed.ndim:
        raise ValueError(f"connectivity must be between 1 and {fixed.ndim} for a {fixed.ndim}D array.")

    next_label = int(np.max(fixed)) + 1 if fixed.size else 1
    split_labels = {}
    unique_labels, inverse = np.unique(fixed, return_inverse=True)
    non_background_mask = unique_labels != background
    non_background_labels = unique_labels[non_background_mask]
    compact_lookup = np.zeros(unique_labels.size, dtype=np.int32)
    compact_lookup[non_background_mask] = np.arange(1, non_background_labels.size + 1, dtype=np.int32)
    compact = compact_lookup[inverse].reshape(fixed.shape)
    structure = generate_binary_structure(fixed.ndim, connectivity)
    n_labels = int(non_background_labels.size)

    for compact_label, bbox in enumerate(find_objects(compact), start=1):
        if bbox is None:
            continue
        component_labels, n_components = label(compact[bbox] == compact_label, structure=structure)
        if n_components <= 1:
            continue
        label_value = int(non_background_labels[compact_label - 1])
        region = fixed[bbox]
        split_labels[label_value] = []
        for component_id in range(2, n_components + 1):
            region[component_labels == component_id] = next_label
            split_labels[label_value].append(next_label)
            next_label += 1

    report = {
        "n_labels": int(n_labels),
        "n_split_labels": int(len(split_labels)),
        "n_new_labels": int(sum(len(new_labels) for new_labels in split_labels.values())),
        "split_labels": split_labels,
        "has_splits": bool(split_labels),
    }
    if verbose:
        if split_labels:
            print(
                f"Split {report['n_split_labels']} labels and created "
                f"{report['n_new_labels']} new labels."
            )
        else:
            print(f"All {report['n_labels']} labels were already connected.")
    if return_report:
        return fixed, report
    return fixed


def particle_size_distribution(
    labelled_array,
    spacing=(1, 1, 1),
    periodic=(False, False, False),
    compute_sphericity=False,
    surface_area_method='gradient',
    return_field=False,
    relabel=True,
    warn=True,
):
    """Measure 3D particle volumes, equivalent sphere diameters, and sphericity.

    Args:
        labelled_array: 3D labeled particle array with background label ``0``.
        spacing: Voxel spacing ``(dx, dy, dz)``. Defaults to ``(1, 1, 1)``.
        periodic: Periodicity flags for ``(x, y, z)``. Periodic axes are ignored
            when removing boundary-touching labels.
        compute_sphericity: If ``True``, also compute particle surface areas and
            sphericity values.
        surface_area_method: Surface-area method passed to
            :func:`specific_surface_area`. Defaults to ``'gradient'``.
        relabel: If ``True``, relabel surviving particles consecutively.
        warn: If ``True``, warn when boundary removal discards more than half of
            the particles or more than half of the particle mass.

    Returns:
        dict: Particle analysis results with cleaned labels, per-particle volumes,
        equivalent diameters, and optional surface areas and sphericity.
    """
    array = _validate_particle_labels(labelled_array, ndim=3, spacing=spacing, periodic=periodic)
    filtered, removed_label_fraction, removed_mass_fraction, total_particles = _remove_boundary_particles(
        array,
        periodic=periodic,
        warn=warn,
    )

    if relabel:
        filtered = relabel_sequential(filtered)

    kept_labels, counts = np.unique(filtered[filtered > 0], return_counts=True)
    voxel_volume = float(np.prod(spacing))
    volumes = counts.astype(float) * voxel_volume
    equivalent_diameters = (6.0 * volumes / np.pi) ** (1.0 / 3.0)

    result = {
        'particle_labels': kept_labels.astype(int),
        'volumes': volumes,
        'equivalent_diameters': equivalent_diameters,
        'removed_label_fraction': removed_label_fraction,
        'removed_mass_fraction': removed_mass_fraction,
        'n_particles_initial': total_particles,
        'n_particles_kept': int(kept_labels.size),
    }
    if return_field:
        result['labels'] = filtered

    if compute_sphericity:
        phases = {str(int(lbl)): int(lbl) for lbl in kept_labels}
        if phases:
            specific_areas = specific_surface_area(
                filtered,
                spacing=spacing,
                phases=phases,
                method=surface_area_method,
                periodic=list(periodic),
            )
            total_volume = filtered.size * voxel_volume
            surface_areas = np.array(
                [specific_areas[str(int(lbl))] * total_volume for lbl in kept_labels],
                dtype=float,
            )
            with np.errstate(divide='ignore', invalid='ignore'):
                sphericity = np.divide(
                    np.pi * equivalent_diameters**2,
                    surface_areas,
                    out=np.full_like(equivalent_diameters, np.nan),
                    where=surface_areas > 0,
                )
        else:
            surface_areas = np.array([], dtype=float)
            sphericity = np.array([], dtype=float)
        result['surface_areas'] = surface_areas
        result['sphericity'] = sphericity

    return result


def particle_size_distribution_2d(
    labelled_array,
    spacing=(1, 1),
    return_field=False,
    relabel=True,
    warn=True,
    perimeter_method='crofton',
):
    """Measure 2D particle areas, equivalent circle diameters, and circularity.

    Args:
        labelled_array: 2D labeled particle array with background label ``0``.
        spacing: Pixel spacing ``(dx, dy)``. Defaults to ``(1, 1)``.
        return_field: If ``True``, include the filtered labeled image.
        relabel: If ``True``, relabel surviving particles consecutively.
        warn: If ``True``, warn when edge removal discards more than half of the
            particles or more than half of the particle area.
        perimeter_method: Either ``'crofton'`` or ``'standard'``.

    Returns:
        dict: Particle analysis results with per-particle areas, equivalent
        diameters, perimeters, and circularity.
    """
    array = _validate_particle_labels(labelled_array, ndim=2, spacing=spacing)
    if not np.isclose(spacing[0], spacing[1]):
        raise ValueError("particle_size_distribution_2d requires equal in-plane spacing for circularity.")

    filtered, removed_label_fraction, removed_mass_fraction, total_particles = _remove_boundary_particles(
        array,
        periodic=(False, False),
        warn=warn,
    )

    if relabel:
        filtered = relabel_sequential(filtered)

    if perimeter_method == 'crofton':
        perimeter_property = 'perimeter_crofton'
    elif perimeter_method == 'standard':
        perimeter_property = 'perimeter'
    else:
        raise ValueError("perimeter_method must be 'crofton' or 'standard'.")

    props = measure.regionprops_table(
        filtered,
        properties=('label', 'area', perimeter_property),
    )

    kept_labels = np.asarray(props['label'], dtype=int)
    areas = np.asarray(props['area'], dtype=float) * float(np.prod(spacing))
    equivalent_diameters = np.sqrt(4.0 * areas / np.pi)
    perimeters = np.asarray(props[perimeter_property], dtype=float) * float(spacing[0])
    with np.errstate(divide='ignore', invalid='ignore'):
        circularity = np.divide(
            4.0 * np.pi * areas,
            perimeters**2,
            out=np.full_like(areas, np.nan),
            where=perimeters > 0,
        )

    result = {
        'particle_labels': kept_labels,
        'areas': areas,
        'equivalent_diameters': equivalent_diameters,
        'perimeters': perimeters,
        'circularity': circularity,
        'removed_label_fraction': removed_label_fraction,
        'removed_mass_fraction': removed_mass_fraction,
        'n_particles_initial': total_particles,
        'n_particles_kept': int(kept_labels.size),
    }
    if return_field:
        result['labels'] = filtered
    return result


def estimate_3d_psd_saltykov(
    apparent_diameters,
    bins='auto',
    bin_edges=None,
    clip_negative=True,
    normalize=True,
):
    """Estimate a 3D sphere diameter distribution from 2D section diameters.

    Uses a Saltykov-style unfolding under the assumption of spherical particles
    cut by a random plane.

    Args:
        apparent_diameters: 1D array of apparent 2D section diameters.
        bins: Histogram bin spec passed to ``numpy.histogram_bin_edges`` when
            ``bin_edges`` is not provided.
        bin_edges: Explicit histogram bin edges.
        clip_negative: If ``True``, clip negative unfolded counts to zero.
        normalize: If ``True``, also return normalized bin fractions.

    Returns:
        dict: Bin edges, centers, 2D histogram counts, and unfolded 3D counts.
    """
    diameters = np.asarray(apparent_diameters, dtype=float)
    diameters = diameters[np.isfinite(diameters)]
    diameters = diameters[diameters > 0]
    if diameters.size == 0:
        raise ValueError("apparent_diameters must contain at least one positive finite value.")

    if bin_edges is None:
        bin_edges = np.histogram_bin_edges(diameters, bins=bins)
    else:
        bin_edges = np.asarray(bin_edges, dtype=float)
    if bin_edges.ndim != 1 or bin_edges.size < 2:
        raise ValueError("bin_edges must be a 1D array with at least two entries.")
    if np.any(np.diff(bin_edges) <= 0):
        raise ValueError("bin_edges must be strictly increasing.")

    counts_2d, _ = np.histogram(diameters, bins=bin_edges)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    n_bins = bin_centers.size
    transfer = np.zeros((n_bins, n_bins), dtype=float)

    def cdf(section_diameter, true_diameter):
        if section_diameter <= 0:
            return 0.0
        if section_diameter >= true_diameter:
            return 1.0
        return 1.0 - np.sqrt(1.0 - (section_diameter / true_diameter) ** 2)

    for j, true_diameter in enumerate(bin_centers):
        for i in range(j + 1):
            lower = bin_edges[i]
            upper = min(bin_edges[i + 1], true_diameter)
            if lower >= true_diameter:
                continue
            probability = cdf(upper, true_diameter) - cdf(lower, true_diameter)
            transfer[i, j] = true_diameter * probability

    counts_3d = np.zeros(n_bins, dtype=float)
    for j in range(n_bins - 1, -1, -1):
        residual = counts_2d[j] - np.dot(transfer[j, j + 1:], counts_3d[j + 1:])
        if transfer[j, j] > 0:
            counts_3d[j] = residual / transfer[j, j]
        if clip_negative and counts_3d[j] < 0:
            counts_3d[j] = 0.0

    result = {
        'bin_edges': bin_edges,
        'bin_centers': bin_centers,
        'counts_2d': counts_2d.astype(float),
        'counts_3d': counts_3d,
    }
    if normalize:
        counts_2d_sum = counts_2d.sum()
        counts_3d_sum = counts_3d.sum()
        result['fractions_2d'] = counts_2d / counts_2d_sum if counts_2d_sum else np.zeros_like(counts_2d, dtype=float)
        result['fractions_3d'] = counts_3d / counts_3d_sum if counts_3d_sum else np.zeros_like(counts_3d, dtype=float)
    return result


def _validate_particle_labels(array, ndim, spacing, periodic=None):
    array = np.asarray(array)
    if array.ndim != ndim:
        raise ValueError(f"Expected a {ndim}D labeled array, got {array.ndim}D input.")
    if len(spacing) != ndim:
        raise ValueError(f"spacing must have {ndim} elements.")
    if periodic is not None and len(periodic) != ndim:
        raise ValueError(f"periodic must have {ndim} elements.")
    if 0 not in np.unique(array):
        raise ValueError("Input must use 0 as the background label.")
    if any(s <= 0 for s in spacing):
        raise ValueError("spacing values must be positive.")
    return array


def _find_boundary_labels(labelled_array, periodic=None):
    """Return labels that touch any non-periodic domain boundary."""
    labelled_array = np.asarray(labelled_array)
    if periodic is None:
        periodic = (False,) * labelled_array.ndim
    if len(periodic) != labelled_array.ndim:
        raise ValueError(
            f"periodic must have {labelled_array.ndim} elements for a {labelled_array.ndim}D array."
        )
    boundary_labels = set()
    for axis, is_periodic in enumerate(periodic):
        if is_periodic:
            continue
        lower = [slice(None)] * labelled_array.ndim
        upper = [slice(None)] * labelled_array.ndim
        lower[axis] = 0
        upper[axis] = -1
        boundary_labels.update(np.unique(labelled_array[tuple(lower)]))
        boundary_labels.update(np.unique(labelled_array[tuple(upper)]))
    boundary_labels.discard(0)
    return boundary_labels


def _remove_boundary_particles(array, periodic, warn):
    particle_labels = np.unique(array)
    particle_labels = particle_labels[particle_labels > 0]
    total_particles = particle_labels.size
    total_mass = np.count_nonzero(array)

    boundary_labels = _find_boundary_labels(array, tuple(bool(p) for p in periodic))
    filtered = array.copy()
    if boundary_labels:
        filtered[np.isin(filtered, list(boundary_labels))] = 0

    removed_particles = len(boundary_labels)
    removed_mass = np.count_nonzero(np.isin(array, list(boundary_labels))) if boundary_labels else 0
    removed_label_fraction = removed_particles / total_particles if total_particles else 0.0
    removed_mass_fraction = removed_mass / total_mass if total_mass else 0.0

    if warn and removed_label_fraction > 0.5:
        warnings.warn(
            "Boundary removal discarded more than half of the particle labels "
            f"({removed_particles}/{total_particles}).",
            UserWarning,
        )
    if warn and removed_mass_fraction > 0.5:
        warnings.warn(
            "Boundary removal discarded more than half of the particle mass "
            f"({removed_mass}/{total_mass} voxels).",
            UserWarning,
        )

    return filtered, removed_label_fraction, removed_mass_fraction, int(total_particles)
