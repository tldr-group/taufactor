import numpy as np

from scipy.ndimage import label, generate_binary_structure

from .base import volume_fraction


def label_periodic(field, grayscale_value, neighbour_structure, periodic, debug=False):
    """Label connected components with periodic boundary conditions.

    Wraps the image in periodic directions, labels connected components
    equal to ``grayscale_value``, then merges labels that touch across
    periodic boundaries. Finally, crops back to the original shape.

    Args:
        field (numpy.ndarray): Input array (2D or 3D).
        grayscale_value (int | float): Target value to label.
        neighbour_structure (numpy.ndarray): Structuring element as from
            ``scipy.ndimage.generate_binary_structure``.
        periodic (Sequence[bool]): Periodicity flags per axis (e.g. ``(True, False, True)``).
        debug (bool, optional): Print simple diagnostics. Defaults to ``False``.

    Returns:
        tuple[numpy.ndarray, int]: Tuple ``(labels, num_labels)`` where ``labels`` is the
        cropped labeled array and ``num_labels`` is the number of connected components
        after periodic merging.
    """
    padx = int(periodic[0])
    pady = int(periodic[1])
    padz = int(periodic[2])
    mask = np.pad(field, ((padx, padx), (pady, pady), (padz, padz)), mode='wrap')
    labeled_mask, num_labels = label(mask==grayscale_value, structure=neighbour_structure)
    count = 1
    for k in range(100):
        # Find indices where labels are different at the boundaries and create swaplist
        swap_list = np.zeros((1,2))
        if periodic[0]:
            # right x
            indices = np.where((labeled_mask[0,:,:]!=labeled_mask[-2,:,:]) & (labeled_mask[0,:,:]!=0) & (labeled_mask[-2,:,:]!=0))
            additional_swaps = np.column_stack((labeled_mask[0,:,:][indices], labeled_mask[-2,:,:][indices]))
            swap_list = np.vstack((swap_list,additional_swaps))
            # left x
            indices = np.where((labeled_mask[1,:,:]!=labeled_mask[-1,:,:]) & (labeled_mask[1,:,:]!=0) & (labeled_mask[-1,:,:]!=0))
            additional_swaps = np.column_stack((labeled_mask[1,:,:][indices], labeled_mask[-1,:,:][indices]))
            swap_list = np.vstack((swap_list,additional_swaps))
        if periodic[1]:
            # top y
            indices = np.where((labeled_mask[:,0,:]!=labeled_mask[:,-2,:]) & (labeled_mask[:,0,:]!=0) & (labeled_mask[:,-2,:]!=0))
            additional_swaps = np.column_stack((labeled_mask[:,0,:][indices], labeled_mask[:,-2,:][indices]))
            swap_list = np.vstack((swap_list,additional_swaps))
            # bottom y
            indices = np.where((labeled_mask[:,1,:]!=labeled_mask[:,-1,:]) & (labeled_mask[:,1,:]!=0) & (labeled_mask[:,-1,:]!=0))
            additional_swaps = np.column_stack((labeled_mask[:,1,:][indices], labeled_mask[:,-1,:][indices]))
            swap_list = np.vstack((swap_list,additional_swaps))
        if periodic[2]:
            # front z
            indices = np.where((labeled_mask[:,:,0]!=labeled_mask[:,:,-2]) & (labeled_mask[:,:,0]!=0) & (labeled_mask[:,:,-2]!=0))
            additional_swaps = np.column_stack((labeled_mask[:,:,0][indices], labeled_mask[:,:,-2][indices]))
            swap_list = np.vstack((swap_list,additional_swaps))
            # back z
            indices = np.where((labeled_mask[:,:,1]!=labeled_mask[:,:,-1]) & (labeled_mask[:,:,1]!=0) & (labeled_mask[:,:,-1]!=0))
            additional_swaps = np.column_stack((labeled_mask[:,:,1][indices], labeled_mask[:,:,-1][indices]))
            swap_list = np.vstack((swap_list,additional_swaps))
        swap_list = swap_list[1:,:]
        # Sort swap list columns to ensure consistent ordering
        swap_list = np.sort(swap_list, axis=1)

        # Remove duplicates from swap_list
        swap_list = np.unique(swap_list, axis=0)
        # print(f"swap_list contains {swap_list.shape[0]} elements.")
        if (swap_list.shape[0]==0):
            break
        for i in range(swap_list.shape[0]):
            index = swap_list.shape[0] - i -1
            labeled_mask[labeled_mask == swap_list[index][1]] = swap_list[index][0]
        count += 1
    if(debug):
        print(f"Did {count} iterations for periodic labelling.")
    dim = labeled_mask.shape
    return labeled_mask[padx:dim[0]-padx,pady:dim[1]-pady,padz:dim[2]-padz], np.unique(labeled_mask).size-1


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


def extract_through_feature(
    array,
    grayscale_value,
    axis,
    periodic=[False,False,False],
    connectivity=1,
    open_end=True,
    debug=False
):
    """Extract spanning features and their fractions for a phase.

    For the given ``grayscale_value``, labels connected components at one
    or more neighbor connectivities, detects which labels span the domain
    along ``axis``, and returns boolean masks plus the fraction of the
    phase volume that is spanning.

    Args:
        array (numpy.ndarray): 3D segmented image.
        grayscale_value (int): Target label value whose spanning network is evaluated.
        axis (str): One of ``'x'``, ``'y'``, or ``'z'`` along which spanning is checked.
        periodic (Sequence[bool], optional): Periodicity flags per axis (e.g.
            ``(True, False, False)``). Defaults to ``[False, False, False]``.
        connectivity (int | None, optional): If ``1``, ``2``, or ``3``, evaluate that
            connectivity only. If ``None``, evaluates all (1, 2, 3). Defaults to ``1``.
        debug (bool, optional): Print simple diagnostics. Defaults to ``False``.

    Returns:
        tuple[list[numpy.ndarray], numpy.ndarray] | tuple[int, int]:
            - If the phase is present: a list of boolean masks (one per connectivity)
              indicating the spanning network, and a 1D array of spanning fractions
              (per connectivity) relative to the phase volume.
            - If the phase volume is zero: ``(0, 0)``.

    Notes:
        Connectivity meanings in 3D:
        - 1: faces (6-neighborhood),
        - 2: faces + edges (18-neighborhood),
        - 3: faces + edges + corners (26-neighborhood).
    """
    if array.ndim != 3:
        print(f"Expected a 3D array, but got an array with {array.ndim} dimension(s).")
        return None

    # Compute volume fraction of given grayscale value
    vol_phase = volume_fraction(array, phases={'1': grayscale_value})['1']
    if vol_phase == 0:
        return 0, 0

    # Define a list of connectivities to loop over
    connectivities_to_loop_over = [connectivity] if connectivity else range(1, 4)
    through_feature = []
    through_feature_fraction = np.zeros(len(connectivities_to_loop_over))

    # Compute the largest interconnected features depending on given connectivity
    count = 0
    for conn in connectivities_to_loop_over:
        neighbour_structure = generate_binary_structure(3, conn)
        # Label connected components in the mask with given neighbour structure
        if any(periodic):
            labeled_mask, num_labels = label_periodic(array, grayscale_value, neighbour_structure, periodic, debug=debug)
        else:
            labeled_mask, num_labels = label(array == grayscale_value, structure=neighbour_structure)
        if(debug):
            print(f"Found {num_labels} labelled regions. For connectivity {conn} and grayscale {grayscale_value}.")

        if open_end:
            through_labels = find_spanning_labels(labeled_mask,axis)
        else:
            through_labels = find_front_labels(labeled_mask,axis)
        spanning_network = np.isin(labeled_mask, list(through_labels))

        through_feature.append(spanning_network)
        through_feature_fraction[count] = volume_fraction(spanning_network, phases={'1': 1})['1']/vol_phase
        count += 1

    return through_feature, through_feature_fraction
