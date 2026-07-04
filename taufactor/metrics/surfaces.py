import numpy as np
import psutil
import warnings

try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None
    F = None

from scipy.ndimage import convolve
from skimage import measure


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


def _gaussian_kernel_3d_torch(size=3, sigma=1.0, device='cuda'):
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
    spacing = (1,1,1),
    phases = {},
    method = 'gradient',
    periodic = [False,False,False],
    device = 'cuda',
    smoothing = True,
    sigma = 0.8,
    verbose = False
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

    dx, dy, dz = spacing
    nx, ny, nz = img.shape
    surface_areas = {}

    device = torch.device(device)
    if torch.device(device).type.startswith('cuda') and not torch.cuda.is_available():
        device = torch.device('cpu')
        warnings.warn("CUDA not available, defaulting device to cpu.")

    if (method == 'gradient') | (method == 'face_counting'):
        tensor = (img if isinstance(img, torch.Tensor)
                  else torch.as_tensor(img)).to(device)
        if len(torch.unique(tensor)) == 1:
            return {str(tensor[0,0,0].int().item()): 0.0}
    
    if (verbose) and (device.type == 'cuda'):
        torch.cuda.reset_peak_memory_stats(device=device)

    if method == 'gradient':
        if phases=={}:
            labels = torch.unique(tensor)
            labels = labels.int()
            phases = {str(label.item()): label.item() for label in labels}
        gaussian = _gaussian_kernel_3d_torch(size=3, sigma=sigma, device=device)

        volume = torch.numel(tensor)
        for name, label in phases.items():
            sub_tensor = _crop_area_of_interest_torch(tensor, label)
            # Create binary mask for the label within the slice
            mask = (sub_tensor == label).float()
            if smoothing:
                mask = mask.unsqueeze(0).unsqueeze(0)
                mask = F.pad(mask, (1,1,1,1,1,1), mode='replicate')
                mask = F.conv3d(mask, gaussian, padding='valid')
                mask = mask.squeeze()

            grad = torch.gradient(mask, spacing=(dx,dy,dz))
            norm2 = grad[0].pow(2) + grad[1].pow(2) + grad[2].pow(2)
            surface_area = torch.sum(torch.sqrt(norm2)).item()

            surface_areas[name] = surface_area / volume

    elif method == 'face_counting':
        tensor = tensor.to(torch.int32)
        max_label = int(tensor.max().item())
        area_per_label = torch.zeros(max_label + 1, dtype=torch.float64, device=tensor.device)
        volume = (nx * ny * nz) * (dx * dy * dz)

        def sum_facing_neighbours(a: torch.Tensor, b: torch.Tensor, face_area: float) -> None:
            idx = (a != b)
            if idx.any():
                counts = torch.bincount(a[idx], minlength=max_label + 1)
                counts += torch.bincount(b[idx], minlength=max_label + 1)
                area_per_label[: counts.shape[0]] += face_area * counts.to(torch.float64)

        sum_facing_neighbours(tensor[1:, :, :], tensor[:-1, :, :], dy*dz)
        sum_facing_neighbours(tensor[:, 1:, :], tensor[:, :-1, :], dx*dz)
        sum_facing_neighbours(tensor[:, :, 1:], tensor[:, :, :-1], dx*dy)

        # Periodic wrap faces (compare the two boundary slabs)
        if periodic[0]:
            sum_facing_neighbours(tensor[0, :, :], tensor[-1, :, :], dy*dz)
        if periodic[1]:
            sum_facing_neighbours(tensor[:, 0, :], tensor[:, -1, :], dx*dz)
        if periodic[2]:
            sum_facing_neighbours(tensor[:, :, 0], tensor[:, :, -1], dx*dy)

        if phases=={}:
            labels = torch.unique(tensor).tolist()
            surface_areas = {str(int(lbl)): (area_per_label[int(lbl)].item() / volume) for lbl in labels}
        else:
            for name, label in phases.items():
                surface_areas[name] = area_per_label[int(label)].item()  / volume

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
        gaussian = _gaussian_kernel_3d_numpy(size=3, sigma=sigma)
        for name, label in phases.items():
            sub_array = _crop_area_of_interest_numpy(array, label)
            sub_array = (sub_array == label).astype(float)
            if smoothing:
                sub_array = convolve(sub_array, gaussian, mode='nearest')
            vertices, faces, _, _ = measure.marching_cubes(sub_array, 0.5, method='lewiner')
            surface_area = measure.mesh_surface_area(vertices, faces)
            surface_areas[name] = float(surface_area/volume)

    else:
        raise ValueError("Choose method\n "
                         "'gradient' for fast phase-field approach\n "
                         "'face_counting' for face counting or\n "
                         "'marching_cubes' for marching cubes method.")
    
    if verbose:
        if device.type == 'cuda':
            print(f"GPU-RAM currently {torch.cuda.memory_allocated(device=device) / 1e6:.2f} MB "
                  f"(max allocated {torch.cuda.max_memory_allocated(device=device) / 1e6:.2f} MB; "
                  f"{torch.cuda.max_memory_reserved(device=device) / 1e6:.2f} MB reserved)")
        elif device.type == 'cpu':
            memory_info = psutil.virtual_memory()
            print(f"CPU total memory: {memory_info.total / 1e6:.2f} MB")
            print(f"CPU available memory: {memory_info.available / 1e6:.2f} MB")
            print(f"CPU used memory: {memory_info.used / 1e6:.2f} MB")

    return surface_areas


def interfacial_areas(
    img,
    spacing = (1,1,1),
    method = 'face_counting',
    periodic = [False,False,False],
    normalize = True,
    device = 'cuda',
    smoothing = True,
    verbose = False
):
    if torch is None:
        raise ImportError("PyTorch is required.")

    dx, dy, dz = spacing
    nx, ny, nz = img.shape

    device = torch.device(device)
    if torch.device(device).type.startswith('cuda') and not torch.cuda.is_available():
        device = torch.device('cpu')
        warnings.warn("CUDA not available, defaulting device to cpu.")


    tensor = (img if isinstance(img, torch.Tensor)
                  else torch.as_tensor(img)).to(device)
    if len(torch.unique(tensor)) == 1:
        return {str(tensor[0,0,0].int().item()): 0.0}
    
    if (verbose) and (device.type == 'cuda'):
        torch.cuda.reset_peak_memory_stats(device=device)

    if method == 'face_counting':

        def add_phasepairs(a: torch.Tensor, b: torch.Tensor, interfacial_areas, face_area: float):
            idx = (a != b)
            if idx.any():
                neighbours = torch.stack([a[idx], b[idx]])
                phasepairs = torch.transpose(neighbours, 0, 1)
                phasepairs, _ = torch.sort(phasepairs, dim=1)
                pairs, counts = torch.unique(phasepairs, return_counts=True, dim=0)
                counts = face_area * counts.to(torch.float64)
                for i in range(pairs.shape[0]):
                    pair = (int(pairs[i][0]), int(pairs[i][1]))
                    interfacial_areas[pair] = interfacial_areas.get(pair, 0.0) + float(counts[i])
            return interfacial_areas

        tensor = tensor.to(torch.int32)
        interfacial_areas = {}
        interfacial_areas = add_phasepairs(tensor[:-1, :, :], tensor[1:, :, :], interfacial_areas, dy*dz)
        interfacial_areas = add_phasepairs(tensor[: ,:-1, :], tensor[:, 1:, :], interfacial_areas, dx*dz)
        interfacial_areas = add_phasepairs(tensor[:, :, :-1], tensor[:, :, 1:], interfacial_areas, dx*dy)

        # Periodic wrap faces (compare the two boundary slabs)
        if periodic[0]:
            interfacial_areas = add_phasepairs(tensor[0, :, :], tensor[-1, :, :], interfacial_areas, dy*dz)
        if periodic[1]:
            interfacial_areas = add_phasepairs(tensor[:, 0, :], tensor[:, -1, :], interfacial_areas, dx*dz)
        if periodic[2]:
            interfacial_areas = add_phasepairs(tensor[:, :, 0], tensor[:, :, -1], interfacial_areas, dx*dy)

        if normalize:
            volume = (nx * ny * nz) * (dx * dy * dz)
            interfacial_areas = {
                pair: area / volume for pair, area in interfacial_areas.items()
            }
    
    else:
        raise ValueError("Only method='face_counting' implemented for now.")
    
    if verbose:
        if device.type == 'cuda':
            print(f"GPU-RAM currently {torch.cuda.memory_allocated(device=device) / 1e6:.2f} MB "
                      f"(max allocated {torch.cuda.max_memory_allocated(device=device) / 1e6:.2f} MB; "
                      f"{torch.cuda.max_memory_reserved(device=device) / 1e6:.2f} MB reserved)")
        elif device.type == 'cpu':
            memory_info = psutil.virtual_memory()
            print(f"CPU total memory: {memory_info.total / 1e6:.2f} MB")
            print(f"CPU available memory: {memory_info.available / 1e6:.2f} MB")
            print(f"CPU used memory: {memory_info.used / 1e6:.2f} MB")

    return interfacial_areas
