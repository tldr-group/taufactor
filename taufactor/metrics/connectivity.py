import cc3d
import numpy as np
from scipy.ndimage import generate_binary_structure

# TauFactor rank 1/2/3 (faces / +edges / +corners) → cc3d 4/8 (2D) or 6/18/26 (3D).
_CC3D_CONNECTIVITY = {
    2: {1: 4, 2: 8, 3: 8},
    3: {1: 6, 2: 18, 3: 26},
}


def _phase_mask(array, phase_labels):
    """Return the binary mask for one or more phase labels."""
    labels = np.asarray(phase_labels)
    if labels.ndim == 0:
        return array == labels
    if labels.ndim != 1:
        raise ValueError("phase_labels must be an integer or a one-dimensional sequence of integers.")
    mask = np.zeros(array.shape, dtype=bool)  # array, mask: array.shape
    for value in labels:  # labels: (n_phase_labels,)
        mask |= array == value
    return mask


def _periodic_label_pairs(labeled_mask, neighbour_structure, periodic):
    """Return connected label pairs separated only by a periodic boundary."""
    centre = np.array(neighbour_structure.shape) // 2
    pairs = []

    for offset in np.argwhere(neighbour_structure) - centre:
        nonzero_axes = np.flatnonzero(offset)
        if not nonzero_axes.size or offset[nonzero_axes[0]] < 0:
            continue

        wrap_axes = [axis for axis in nonzero_axes if periodic[axis]]
        for wrapped in range(1, 1 << len(wrap_axes)):
            source_slices = []
            target_slices = []
            for axis, step in enumerate(offset):
                if step == 0:
                    source_slices.append(slice(None))
                    target_slices.append(slice(None))
                    continue

                wraps = axis in wrap_axes and wrapped & (1 << wrap_axes.index(axis))
                if step > 0:
                    source_slices.append(slice(-1, None) if wraps else slice(None, -1))
                    target_slices.append(slice(0, 1) if wraps else slice(1, None))
                else:
                    source_slices.append(slice(0, 1) if wraps else slice(1, None))
                    target_slices.append(slice(-1, None) if wraps else slice(None, -1))

            source = labeled_mask[tuple(source_slices)]
            target = labeled_mask[tuple(target_slices)]
            connected = (source != 0) & (target != 0) & (source != target)
            if connected.any():
                pairs.append(np.column_stack((source[connected], target[connected])))

    if not pairs:
        return np.empty((0, 2), dtype=labeled_mask.dtype)
    return np.unique(np.sort(np.concatenate(pairs), axis=1), axis=0)


def _merge_periodic_labels(labeled_mask, num_labels, pairs):
    """Merge periodic label pairs in place and return the merged label count."""
    parent = np.arange(num_labels + 1, dtype=labeled_mask.dtype)

    def find(label_value):
        while parent[label_value] != label_value:
            parent[label_value] = parent[parent[label_value]]
            label_value = parent[label_value]
        return label_value

    merges = 0
    for first, second in pairs:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[second_root] = first_root
            merges += 1

    for label_value in np.unique(pairs):
        parent[label_value] = find(label_value)

    chunk_depth = max(1, (16 * 1024**2) // labeled_mask[0].nbytes)
    for start in range(0, labeled_mask.shape[0], chunk_depth):
        chunk = labeled_mask[start:start + chunk_depth]
        labeled_mask[start:start + chunk_depth] = parent[chunk]
    return num_labels - merges


def label_periodic(
    field,
    phase_labels,
    periodic,
    connectivity=1,
    debug=False,
    phase_mask=None,
):
    """Label connected components with periodic boundary conditions.

    Labels the unpadded phase mask, then merges labels connected across periodic boundaries.

    Args:
        field (numpy.ndarray): Input array (2D or 3D).
        phase_labels (int | Sequence[int]): Label value(s) forming the connected phase.
        periodic (Sequence[bool]): Periodicity flags per axis (e.g. ``(True, False, True)``).
        connectivity (int, optional): TauFactor rank ``1``, ``2``, or ``3``.
            Defaults to ``1``.
        debug (bool, optional): Print simple diagnostics. Defaults to ``False``.
        phase_mask (numpy.ndarray, optional): Precomputed mask for ``phase_labels``.
            Defaults to ``None``.

    Returns:
        tuple[numpy.ndarray, int]: Tuple ``(labels, num_labels)`` where ``labels`` is the
        cropped labeled array and ``num_labels`` is the number of connected components
        after periodic merging.
    """
    if phase_mask is None:
        phase_mask = _phase_mask(field, phase_labels)

    labeled_mask, num_labels = cc3d.connected_components(
        phase_mask,
        connectivity=_CC3D_CONNECTIVITY[phase_mask.ndim][connectivity],
        return_N=True,
        binary_image=True,
    )
    neighbour_structure = generate_binary_structure(phase_mask.ndim, connectivity)
    pairs = _periodic_label_pairs(labeled_mask, neighbour_structure, periodic)
    if pairs.size:
        num_labels = _merge_periodic_labels(labeled_mask, num_labels, pairs)
    if debug:
        print(f"Merged {pairs.shape[0]} periodic label pairs.")
    return labeled_mask, num_labels


def find_spanning_labels(labelled_array, axis):
    """Find labels that span the domain along an axis.

    A label is considered spanning if it appears on both opposing faces
    along the specified axis; background label ``0`` is ignored.

    Args:
        labelled_array (numpy.ndarray): Labeled 3D array.
        axis (str): One of ``'x'``, ``'y'``, or ``'z'``.

    Returns:
        set[int]: Set of labels that appear on both faces along ``axis``.

    Raises:
        ValueError: If ``axis`` is not one of ``'x'``, ``'y'``, ``'z'``.
    """
    if axis == "x":
        front = np.s_[0,:,:]
        end   = np.s_[-1,:,:]
    elif axis == "y":
        front = np.s_[:,0,:]
        end   = np.s_[:,-1,:]
    elif axis == "z":
        front = np.s_[:,:,0]
        end   = np.s_[:,:,-1]
    else:
        raise ValueError("Axis should be x, y or z!")

    first_slice_labels = np.unique(labelled_array[front])
    last_slice_labels = np.unique(labelled_array[end])
    spanning_labels = set(first_slice_labels) & set(last_slice_labels)
    spanning_labels.discard(0)  # Remove the background label if it exists
    return spanning_labels


def find_front_labels(labelled_array, axis):
    """Find features that are connected to the front of given axis

    Returns:
        set: Labels that appear in the first slice of the given axis.
    """
    if axis == "x":
        front = np.s_[0,:,:]
    elif axis == "y":
        front = np.s_[:,0,:]
    elif axis == "z":
        front = np.s_[:,:,0]
    else:
        raise ValueError("Axis should be x, y or z!")

    first_slice_labels = set(np.unique(labelled_array[front]))
    first_slice_labels.discard(0)  # Remove the background label if it exists
    return first_slice_labels


def extract_connected_network(
    array,
    phase_labels,
    axis,
    periodic=None,
    connectivity=1,
    open_end=True,
    debug=False
):
    """Extract a connected network and connectivity metrics for a phase.

    For the given ``phase_labels``, labels connected components in their union at one
    or more neighbor connectivities, detects which labels span the domain
    along ``axis``, and returns connectivity results keyed by neighbourhood. With
    ``open_end=True``, the network percolates along ``axis``; otherwise it is
    connected to the first face along that axis.

    Args:
        array (numpy.ndarray): 3D segmented image.
        phase_labels (int | Sequence[int]): Label value(s) forming the phase whose
            spanning network is evaluated.
        axis (str): One of ``'x'``, ``'y'``, or ``'z'`` along which spanning is checked.
        periodic (Sequence[bool], optional): Periodicity flags per axis (e.g.
            ``(True, False, False)``). Defaults to ``[False, False, False]``.
        connectivity (int | None, optional): If ``1``, ``2``, or ``3``, evaluate that
            connectivity only. If ``None``, evaluates all (1, 2, 3). Defaults to ``1``.
        open_end (bool, optional): If ``True``, returns the network percolating along
            ``axis``; if ``False``, returns the network connected to the inlet face.
        debug (bool, optional): Print simple diagnostics. Defaults to ``False``.

    Returns:
        dict[int, dict[str, numpy.ndarray | float]]: Results keyed by connectivity.
            Each entry contains ``connected_mask`` (a boolean 3D mask),
            ``connected_fraction`` (the fraction of the selected phase in the
            connected network), and ``disconnected_volume_fraction`` (the local
            disconnected-phase volume fraction along ``axis``). Returns an empty
            dictionary if none of ``phase_labels`` are present.

    Notes:
        Connectivity meanings in 3D:
        - 1: faces (6-neighborhood),
        - 2: faces + edges (18-neighborhood),
        - 3: faces + edges + corners (26-neighborhood).
    """
    if periodic is None:
        periodic = [False, False, False]

    if array.ndim != 3:
        print(f"Expected a 3D array, but got an array with {array.ndim} dimension(s).")
        return None

    try:
        axis_index = "xyz".index(axis)
    except ValueError as error:
        raise ValueError("Axis should be x, y or z!") from error

    # Build the binary phase image once; it is also the direct input to ``label``.
    phase_mask = _phase_mask(array, phase_labels)
    vol_phase = np.count_nonzero(phase_mask) / phase_mask.size

    # Define a list of connectivities to loop over
    connectivities_to_loop_over = [connectivity] if connectivity else range(1, 4)
    if vol_phase == 0:
        return {}

    results = {}
    transverse_axes = tuple(index for index in range(array.ndim) if index != axis_index)
    volume_fraction_all = phase_mask.mean(axis=transverse_axes)

    # Compute the largest interconnected features depending on given connectivity
    for conn in connectivities_to_loop_over:
        if any(periodic):
            labeled_mask, num_labels = label_periodic(
                array,
                phase_labels,
                periodic,
                connectivity=conn,
                debug=debug,
                phase_mask=phase_mask,
            )
        else:
            labeled_mask, num_labels = cc3d.connected_components(
                phase_mask,
                connectivity=_CC3D_CONNECTIVITY[phase_mask.ndim][conn],
                return_N=True,
                binary_image=True,
            )
        if(debug):
            print(f"Found {num_labels} labelled regions. For connectivity {conn} and phase labels {phase_labels}.")

        if open_end:
            through_labels = find_spanning_labels(labeled_mask,axis)
        else:
            through_labels = find_front_labels(labeled_mask,axis)
        is_spanning = np.zeros(int(labeled_mask.max()) + 1, dtype=bool)  # (max_label + 1,)
        if through_labels:
            is_spanning[list(through_labels)] = True  # through_labels: set of label ids
        spanning_network = is_spanning[labeled_mask]  # labeled_mask, spanning_network: array.shape
        volume_fraction_conn = spanning_network.mean(axis=transverse_axes)

        results[conn] = {
            "connected_mask": spanning_network,
            "connected_fraction": spanning_network.mean() / vol_phase,
            "spatial_vol_frac_all": volume_fraction_all,
            "spatial_vol_frac_connected": volume_fraction_conn,
            "spatial_vol_frac_disconnected": volume_fraction_all - volume_fraction_conn,
        }
    return results
