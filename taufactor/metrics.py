import numpy as np
import torch
import torch.nn.functional as F
import warnings

from scipy.ndimage import label, generate_binary_structure, convolve
from skimage import measure

def volume_fraction(img, phases={}):
    """
    Calculates volume fractions of phases in an image
    :param img: segmented input image with n phases
    :param phases: a dictionary of phases to be calculated with keys as labels and phase values as values, default empty
    :return: list of volume fractions if no labels, dictionary of label: volume fraction pairs if labelled
    """

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

def crop_area_of_interest_torch(tensor, labels):
    indices = torch.nonzero(torch.isin(tensor, labels), as_tuple=True)
    min_idx = [torch.min(idx).item() for idx in indices]
    max_idx = [torch.max(idx).item() for idx in indices]

    # Slice the tensor to the bounding box
    # Make sure to stay inside the bounds of total array
    sub_tensor = tensor[max(min_idx[0]-3,0):min(max_idx[0]+4,tensor.shape[0]),
                        max(min_idx[1]-3,0):min(max_idx[1]+4,tensor.shape[1]),
                        max(min_idx[2]-3,0):min(max_idx[2]+4,tensor.shape[2])]
    return sub_tensor

def crop_area_of_interest_numpy(array, labels):
    indices = np.nonzero(np.isin(array, labels))
    min_idx = [np.min(idx) for idx in indices]
    max_idx = [np.max(idx) for idx in indices]

    # Slice the array to the bounding box
    # Make sure to stay inside the bounds of the total array
    sub_array = array[max(min_idx[0]-3, 0):min(max_idx[0]+4, array.shape[0]),
                      max(min_idx[1]-3, 0):min(max_idx[1]+4, array.shape[1]),
                      max(min_idx[2]-3, 0):min(max_idx[2]+4, array.shape[2])]
    return sub_array

def gaussian_kernel_3d_torch(size=3, sigma=1.0, device=torch.device('cuda')):
    """Creates a 3D Gaussian kernel using PyTorch"""
    ax = torch.linspace(-(size // 2), size // 2, size)
    xx, yy, zz = torch.meshgrid(ax, ax, ax, indexing="ij")

    # Calculate Gaussian function for each point in the grid
    kernel = torch.exp(-(xx**2 + yy**2 + zz**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    kernel = kernel.to(device)
    return kernel.unsqueeze(0).unsqueeze(0)

def gaussian_kernel_3d_numpy(size=3, sigma=1.0):
    """Creates a 3D Gaussian kernel using NumPy"""
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy, zz = np.meshgrid(ax, ax, ax)

    # Calculate Gaussian function for each point in the grid
    kernel = np.exp(-(xx**2 + yy**2 + zz**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)
    return kernel

def specific_surface_area(img, spacing=(1,1,1), phases={}, method='gradient', device=torch.device('cuda'), smoothing=True):
    """
    Calculate the specific surface area of all (specified) phases
    :param img: labelled microstructure where each integer value represents a phase
    :param spacing: voxel size in each dimension [dx,dy,dz]
    :param phases: dictionary of phases {'name': label, ...}. If empty do all by default.
    :param method: string to indicate preferred method (face_counting, marching_cubes or gradient)
    :return: the surface area per unit volume
    """
    [dx,dy,dz] = spacing
    surface_areas = {}

    if (method == 'gradient') | (method == 'face_counting'):
        if type(img) is not type(torch.tensor(1)):
            tensor = torch.tensor(img)
        else:
            tensor = img
        tensor = tensor.to(device)

    if method == 'gradient':
        if phases=={}:
            labels = torch.unique(tensor)
            labels = labels.int()
            phases = {str(label.item()): label.item() for label in labels}
        gaussian = gaussian_kernel_3d_torch(device=device)

        volume = torch.numel(tensor)
        for name, label in phases.items():
            sub_tensor = crop_area_of_interest_torch(tensor, label)
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
        if device != 'cpu':
            warnings.warn("The marching cubes algorithm is performed on the CPU based on scikit-image package.")
        if dx != dy | dx!= dz | dy!=dz:
            raise ValueError("Surface area computation based on marching cubes assumes dx=dy=dz.")

        if type(img) is type(torch.tensor(1)):
            array = np.array(img.cpu())
        else:
            array = img

        if phases=={}:
            labels = np.unique(array).astype(int)
            phases = {str(label): label for label in labels}

        volume = array.size*dx
        gaussian = gaussian_kernel_3d_numpy(size=3, sigma=1.0)
        for name, label in phases.items():
            sub_array = crop_area_of_interest_numpy(array, label)
            sub_array = (sub_array == label).astype(float)
            if smoothing:
                sub_array = convolve(sub_array, gaussian, mode='nearest')
            vertices, faces, _, _ = measure.marching_cubes(sub_array, 0.5, method='lewiner')
            surface_area = measure.mesh_surface_area(vertices, faces)
            surface_areas[name] = surface_area/volume

    else:
        raise ValueError("Choose method\n 'gradient' for fast phase-field approach\n 'face_counting' for face counting or\n 'marching_cubes' for marching cubes method.")

    return surface_areas

def triple_phase_boundary(img):
    """Calculate triple phase boundary density i.e. fraction of voxel verticies that touch at least 3 phases

    Args:
        img (numpy array): image to calculate metric on
    Returns:
        float: triple phase boundary density 
    """
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
    # Initialize phi field whith enlarged dimensions in periodic directions. Boundary values of
    # array are copied into ghost cells which are necessary to impose boundary conditions.
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
            swap_list = np.row_stack((swap_list,additional_swaps))
            # left x
            indices = np.where((labeled_mask[1,:,:]!=labeled_mask[-1,:,:]) & (labeled_mask[1,:,:]!=0) & (labeled_mask[-1,:,:]!=0))
            additional_swaps = np.column_stack((labeled_mask[1,:,:][indices], labeled_mask[-1,:,:][indices]))
            swap_list = np.row_stack((swap_list,additional_swaps))
        if periodic[1]:
            # top y
            indices = np.where((labeled_mask[:,0,:]!=labeled_mask[:,-2,:]) & (labeled_mask[:,0,:]!=0) & (labeled_mask[:,-2,:]!=0))
            additional_swaps = np.column_stack((labeled_mask[:,0,:][indices], labeled_mask[:,-2,:][indices]))
            swap_list = np.row_stack((swap_list,additional_swaps))
            # bottom y
            indices = np.where((labeled_mask[:,1,:]!=labeled_mask[:,-1,:]) & (labeled_mask[:,1,:]!=0) & (labeled_mask[:,-1,:]!=0))
            additional_swaps = np.column_stack((labeled_mask[:,1,:][indices], labeled_mask[:,-1,:][indices]))
            swap_list = np.row_stack((swap_list,additional_swaps))
        if periodic[2]:
            # front z
            indices = np.where((labeled_mask[:,:,0]!=labeled_mask[:,:,-2]) & (labeled_mask[:,:,0]!=0) & (labeled_mask[:,:,-2]!=0))
            additional_swaps = np.column_stack((labeled_mask[:,:,0][indices], labeled_mask[:,:,-2][indices]))
            swap_list = np.row_stack((swap_list,additional_swaps))
            # back z
            indices = np.where((labeled_mask[:,:,1]!=labeled_mask[:,:,-1]) & (labeled_mask[:,:,1]!=0) & (labeled_mask[:,:,-1]!=0))
            additional_swaps = np.column_stack((labeled_mask[:,:,1][indices], labeled_mask[:,:,-1][indices]))
            swap_list = np.row_stack((swap_list,additional_swaps))
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
    """
    Find labels that appear on both ends along given axis

    Returns:
        set: Labels that appear on both ends of the first axis.
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

def extract_through_feature(array, grayscale_value, axis, periodic=[False,False,False], connectivity=1, debug=False):
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
        # connectivity 1 = cells connected by sides (6 neighbours)
        # connectivity 2 = cells connected by sides & edges (14 neighbours)
        # connectivity 3 = cells connected by sides & edges & corners (26 neighbours)
        neighbour_structure = generate_binary_structure(3,conn)
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
