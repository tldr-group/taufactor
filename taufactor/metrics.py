import numpy as np
import psutil
import warnings

try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None
    F = None

from scipy.ndimage import label, generate_binary_structure, convolve
from skimage import measure


def volume_fraction(img, phases={}):
    """Compute volume fractions for labels in a segmented image.

    Calculates the fraction of voxels belonging to each phase. If
    ``phases`` is empty, all unique labels in ``img`` are measured.
    Otherwise, uses the provided mapping of phase names to label values.

    Args:
        img (torch.Tensor | numpy.ndarray): Segmented image. If not a
            ``torch.Tensor``, it will be converted to one.
        phases (dict[str, int], optional): Mapping from phase name to the
            integer label in ``img``. If empty (default), all labels in
            the image are measured and names are derived from the label
            values.

    Returns:
        dict[str, float]: Mapping from phase name to volume fraction in
        the range ``[0, 1]``.

    Raises:
        ImportError: If PyTorch is not available.
    """
    if torch is None:
        raise ImportError("PyTorch is required.")

    if type(img) is not type(torch.tensor(1)):
        img = torch.tensor(img)

    if phases=={}:
        volume = torch.numel(img)
        labels, counts = torch.unique(img, return_counts=True)
        labels = labels.int()
        counts = counts.float()
        counts /= volume
        vf_out = {}
        for i, label in enumerate(labels):
            vf_out[str(label.item())] = counts[i].item()
    else:
        vf_out={}
        for p in phases:
            vf_out[p]=(img==phases[p]).to(torch.float).mean().item()

    return vf_out


def _crop_area_of_interest_torch(tensor, labels):
    """Return a tight crop around specified labels in a tensor.

    Computes the bounding box of all voxels whose value is in
    ``labels`` and returns the sub-tensor expanded by a margin of 3
    voxels in each dimension (clipped to bounds).

    Args:
        tensor (torch.Tensor): Input tensor (3D).
        labels (int | Sequence[int]): Label value(s) to include.

    Returns:
        torch.Tensor: Cropped sub-tensor containing the region of
        interest plus a 3-voxel margin.
    """
    indices = torch.nonzero(torch.isin(tensor, labels), as_tuple=True)
    min_idx = [torch.min(idx).item() for idx in indices]
    max_idx = [torch.max(idx).item() for idx in indices]

    # Slice the tensor to the bounding box
    # Make sure to stay inside the bounds of total array
    sub_tensor = tensor[max(min_idx[0]-3,0):min(max_idx[0]+4,tensor.shape[0]),
                        max(min_idx[1]-3,0):min(max_idx[1]+4,tensor.shape[1]),
                        max(min_idx[2]-3,0):min(max_idx[2]+4,tensor.shape[2])]
    return sub_tensor


def _crop_area_of_interest_numpy(array, labels):
    """Return a tight crop around specified labels in an array.

    Computes the bounding box of all voxels whose value is in
    ``labels`` and returns the sub-array expanded by a margin of 3
    voxels in each dimension (clipped to bounds).

    Args:
        array (numpy.ndarray): Input array (3D).
        labels (int | Sequence[int]): Label value(s) to include.

    Returns:
        numpy.ndarray: Cropped sub-array containing the region of
        interest plus a 3-voxel margin.
    """
    indices = np.nonzero(np.isin(array, labels))
    min_idx = [np.min(idx) for idx in indices]
    max_idx = [np.max(idx) for idx in indices]

    # Slice the array to the bounding box
    # Make sure to stay inside the bounds of the total array
    sub_array = array[max(min_idx[0]-3, 0):min(max_idx[0]+4, array.shape[0]),
                      max(min_idx[1]-3, 0):min(max_idx[1]+4, array.shape[1]),
                      max(min_idx[2]-3, 0):min(max_idx[2]+4, array.shape[2])]
    return sub_array


def _gaussian_kernel_3d_torch(device, size=3, sigma=1.0):
    """Create normalized 3D Gaussian kernel using PyTorch"""
    ax = torch.linspace(-(size // 2), size // 2, size)
    xx, yy, zz = torch.meshgrid(ax, ax, ax, indexing="ij")

    # Calculate Gaussian function for each point in the grid
    kernel = torch.exp(-(xx**2 + yy**2 + zz**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    kernel = kernel.to(device)
    return kernel.unsqueeze(0).unsqueeze(0)


def _gaussian_kernel_3d_numpy(size=3, sigma=1.0):
    """Create normalized 3D Gaussian kernel using NumPy"""
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy, zz = np.meshgrid(ax, ax, ax)

    # Calculate Gaussian function for each point in the grid
    kernel = np.exp(-(xx**2 + yy**2 + zz**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)
    return kernel


def specific_surface_area(
    img,
    spacing=(1,1,1),
    phases={},
    method='gradient',
    device='cuda',
    smoothing=True,
    verbose=False
):
    """Compute specific surface area per phase.

    Supports three methods:
    - ``'gradient'`` (default): Smooth binary masks and integrate gradient magnitude.
    - ``'face_counting'``: Count voxel face changes between neighboring cells.
    - ``'marching_cubes'``: Extract surfaces and compute mesh area (CPU-only; assumes
      ``dx == dy == dz``).

    Args:
        img (torch.Tensor | numpy.ndarray): Labeled microstructure; integer values
            represent phases.
        spacing (tuple[float, float, float], optional): Voxel spacing ``(dx, dy, dz)``.
            Defaults to ``(1, 1, 1)``.
        phases (dict[str, int], optional): Mapping from phase name to the integer label
            in ``img``. If empty, all labels are processed. Defaults to ``{}``.
        method (str, optional): One of ``'gradient'``, ``'face_counting'``, or
            ``'marching_cubes'``. Defaults to ``'gradient'``.
        device (str | torch.device, optional): Device for GPU-accelerated methods.
            Only used for ``'gradient'``/``'face_counting'``. Defaults to ``'cuda'``.
        smoothing (bool, optional): Apply light Gaussian smoothing to the binary mask
            prior to measurement (used in ``'gradient'`` and ``'marching_cubes'``).
            Defaults to ``True``.
        verbose (bool, optional): Print simple memory usage diagnostics. Defaults to ``False``.

    Returns:
        dict[str, float]: Mapping from phase name to specific surface area
        (surface per unit volume).

    Raises:
        ImportError: If PyTorch is not available for ``'gradient'`` or
            ``'face_counting'`` methods.
        ValueError: If ``method`` is invalid, or if ``'marching_cubes'`` is used
            with anisotropic spacing (requires ``dx == dy == dz``).
    """
    if torch is None:
        raise ImportError("PyTorch is required.")

    [dx,dy,dz] = spacing
    surface_areas = {}

    device = torch.device(device)
    if torch.device(device).type.startswith('cuda') and not torch.cuda.is_available():
        device = torch.device('cpu')
        warnings.warn("CUDA not available, defaulting device to cpu.")

    if (method == 'gradient') | (method == 'face_counting'):
        if type(img) is not type(torch.tensor(1)):
            tensor = torch.tensor(img)
        else:
            tensor = img
        tensor = tensor.to(device)
    
    if (verbose) and (device.type == 'cuda'):
        torch.cuda.reset_peak_memory_stats(device=device)

    if method == 'gradient':
        if phases=={}:
            labels = torch.unique(tensor)
            labels = labels.int()
            phases = {str(label.item()): label.item() for label in labels}
        gaussian = _gaussian_kernel_3d_torch(device)

        volume = torch.numel(tensor)
        for name, label in phases.items():
            sub_tensor = _crop_area_of_interest_torch(tensor, label)
            # Create binary mask for the label within the slice
            mask = (sub_tensor == label).float()
            if smoothing:
                mask = mask.unsqueeze(0).unsqueeze(0)
                mask = F.pad(mask, (1,1,1,1,1,1), mode='reflect')
                mask = F.conv3d(mask, gaussian, padding='valid')
                mask = mask.squeeze()

            grad = torch.gradient(mask, spacing=(dx,dy,dz))
            norm2 = grad[0].pow(2) + grad[1].pow(2) + grad[2].pow(2)
            surface_area = torch.sum(torch.sqrt(norm2)).item()

            surface_areas[name] = surface_area / volume

    elif method == 'face_counting':
        # TODO: treat dimensions such that dx!=dz is accounted for
        tensor = tensor.to(torch.int32)
        if len(torch.unique(tensor)) == 1:
            surface_areas = {str(tensor[0,0,0].int().item()): 0.0}
        else:
            volume = torch.numel(tensor)*dx
            phasepairs = torch.tensor([[0,0]], device=device)

            neighbour_idx = torch.nonzero(tensor[:-1,:,:] != tensor[1:,:,:], as_tuple=True)
            neighbour_list = torch.stack([tensor[:-1,:,:][neighbour_idx], tensor[1:,:,:][neighbour_idx]])
            phasepairs = torch.cat((phasepairs,torch.transpose(neighbour_list,0,1)), 0)
            neighbour_idx = torch.nonzero(tensor[:,:-1,:] != tensor[:,1:,:], as_tuple=True)
            neighbour_list = torch.stack([tensor[:,:-1,:][neighbour_idx], tensor[:,1:,:][neighbour_idx]])
            phasepairs = torch.cat((phasepairs,torch.transpose(neighbour_list,0,1)), 0)
            neighbour_idx = torch.nonzero(tensor[:,:,:-1] != tensor[:,:,1:], as_tuple=True)
            neighbour_list = torch.stack([tensor[:,:,:-1][neighbour_idx], tensor[:,:,1:][neighbour_idx]])
            phasepairs = torch.cat((phasepairs,torch.transpose(neighbour_list,0,1)), 0)

            # Crop initial dummy values
            phasepairs = phasepairs[1:]

            if phases=={}:
                if phasepairs == {}:
                    surface_areas
                labels, counts = torch.unique(phasepairs, return_counts=True)
                labels = labels.int()
                counts = counts.float()
                counts /= volume
                surface_areas = {str(label.item()): counts[i].item() for i, label in enumerate(labels)}
            else:
                for name, label in phases.items():
                    count = torch.sum((phasepairs == label).int()).item()
                    surface_areas[name] = count / volume

    elif method == 'marching_cubes':
        if device.type != 'cpu':
            warnings.warn("The marching cubes algorithm is performed on the CPU based on scikit-image package.")
        if (dx != dy) or (dx!= dz) or (dy!=dz):
            raise ValueError("Surface area computation based on marching cubes assumes dx=dy=dz.")

        if type(img) is type(torch.tensor(1)):
            array = np.array(img.cpu())
        else:
            array = img

        if phases=={}:
            labels = np.unique(array).astype(int)
            phases = {str(label): label for label in labels}

        volume = array.size*dx
        gaussian = _gaussian_kernel_3d_numpy(size=3, sigma=1.0)
        for name, label in phases.items():
            sub_array = _crop_area_of_interest_numpy(array, label)
            sub_array = (sub_array == label).astype(float)
            if smoothing:
                sub_array = convolve(sub_array, gaussian, mode='nearest')
            vertices, faces, _, _ = measure.marching_cubes(sub_array, 0.5, method='lewiner')
            surface_area = measure.mesh_surface_area(vertices, faces)
            surface_areas[name] = surface_area/volume

    else:
        raise ValueError("Choose method\n 'gradient' for fast phase-field approach\n 'face_counting' for face counting or\n 'marching_cubes' for marching cubes method.")
    
    if verbose:
        if device.type == 'cuda':
            print(f"GPU-RAM currently allocated {torch.cuda.memory_allocated(device=device) / 1e6:.2f} MB ({torch.cuda.memory_reserved(device=device) / 1e6:.2f} MB reserved)")
            print(f"GPU-RAM maximally allocated {torch.cuda.max_memory_allocated(device=device) / 1e6:.2f} MB ({torch.cuda.max_memory_reserved(device=device) / 1e6:.2f} MB reserved)")
        elif device.type == 'cpu':
            memory_info = psutil.virtual_memory()
            print(f"CPU total memory: {memory_info.total / 1e6:.2f} MB")
            print(f"CPU available memory: {memory_info.available / 1e6:.2f} MB")
            print(f"CPU used memory: {memory_info.used / 1e6:.2f} MB")

    return surface_areas


def triple_phase_boundary(img):
    """Compute triple-phase boundary (TPB) density.

    Calculates the fraction of voxel vertices/edges that are shared by
    at least three distinct phases. The input image must contain exactly
    three unique labels.

    Args:
        img (numpy.ndarray | torch.Tensor): Segmented 2D or 3D image with exactly
            three phase labels.

    Returns:
        float: Triple-phase boundary density (normalized by the number of candidate
        vertices/edges).

    Raises:
        ImportError: If PyTorch is not available.
        ValueError: If the image does not contain exactly three phases.
    """
    if torch is None:
        raise ImportError("PyTorch is required.")

    phases = torch.unique(torch.tensor(img))
    if len(phases)!=3:
        raise ValueError('Image must have exactly 3 phases')
    shape = img.shape
    dim = len(shape)
    ph_maps = []
    img = F.pad(torch.tensor(img), (1,)*dim*2, 'constant', value=-1)
    if dim==2:
        x, y = shape
        total_edges = (x-1)*(y-1)
        for ph in phases:
            ph_map = torch.zeros_like(img)
            ph_map_temp = torch.zeros_like(img)
            ph_map_temp[img==ph] = 1
            for i in [0, 1]:
                for j in [0, 1]:
                    ph_map += torch.roll(torch.roll(ph_map_temp, i, 0), j, 1)
            ph_maps.append(ph_map)
        tpb_map = torch.ones_like(img)
        for ph_map in ph_maps:
            tpb_map *= ph_map
        tpb_map[tpb_map>1] = 1
        tpb_map = tpb_map[1:-1, 1:-1]
        tpb = torch.sum(tpb_map)
    else:
        tpb = 0
        x, y, z = shape
        total_edges = z*(x-1)*(y-1) + x*(y-1)*(z-1) + y*(x-1)*(z-1)
        print(total_edges)
        for d in range(dim):
            ph_maps = []
            for ph in phases:
                ph_map = torch.zeros_like(img)
                ph_map_temp = torch.zeros_like(img)
                ph_map_temp[img==ph] = 1
                for i in [0, 1]:
                    for j in [0, 1]:
                        d1 =( d + 1) % 3
                        d2 = (d + 2) % 3
                        ph_map += torch.roll(torch.roll(ph_map_temp, i, d1), j, d2)
                ph_maps.append(ph_map)
            tpb_map = torch.ones_like(img)
            for ph_map in ph_maps:
                tpb_map *= ph_map
            tpb_map[tpb_map>1] = 1
            tpb_map = tpb_map[1:-1, 1:-1, 1:-1]
            tpb += torch.sum(tpb_map)

    return tpb/total_edges


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


def extract_through_feature(
    array,
    grayscale_value,
    axis,
    periodic=[False,False,False],
    connectivity=1,
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

        through_labels = find_spanning_labels(labeled_mask,axis)
        spanning_network = np.isin(labeled_mask, list(through_labels))

        through_feature.append(spanning_network)
        through_feature_fraction[count] = volume_fraction(spanning_network, phases={'1': 1})['1']/vol_phase
        count += 1

    return through_feature, through_feature_fraction
