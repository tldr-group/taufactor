try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None
    F = None


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
